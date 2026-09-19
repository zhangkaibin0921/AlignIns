import copy
import time

import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, backdoor_train_dataset=None):
        self.id = id
        self.args = args
        self.error = 0
        self.hessian_metrix = []
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if self.args.data != "tinyimagenet":
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)

            # Adaptive MDF uses a clean local update as an attacker-side proxy for
            # the unavailable current-round median update.  Preserve this split
            # before poisoning rather than using the whole training dataset.
            adaptive_attacks = {"adaptive", "adaptive_mdf", "adaptive_pdc", "adaptive_joint"}
            if self.id < args.num_corrupt and self.args.attack in adaptive_attacks:
                self.adaptive_clean_train_loader = DataLoader(
                    copy.deepcopy(self.train_dataset), batch_size=self.args.bs,
                    shuffle=True, num_workers=args.num_workers, pin_memory=False,
                    drop_last=True,
                )
                self._adaptive_reference_cache = {}

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data != 'sen140':
                self.data_idxs = data_idxs
                
                # For SoDa attack, backup clean dataset split before poisoning
                # This ensures clean data for self-reference training
                if self.args.attack == 'soda':
                    # Backup the clean dataset split (before poisoning)
                    self.clean_backup_dataset = copy.deepcopy(self.train_dataset)
                    self.clean_train_loader = DataLoader(
                        self.clean_backup_dataset,
                        batch_size=self.args.bs,
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=True
                    )
                else:
                    # For other attacks, backup the original dataset
                    self.clean_backup_dataset = copy.deepcopy(train_dataset)
                
                utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
            elif self.id < args.num_corrupt and self.args.attack != 'non' and self.args.data == 'sen140':
                self.clean_backup_dataset = copy.deepcopy(train_dataset)
                self.data_idxs = data_idxs
                benign_part = data_idxs[:int(len(data_idxs) * (1 - self.args.poison_frac))]
                malicious_part = data_idxs[int(len(data_idxs) * (1 - self.args.poison_frac)):]

                self.train_dataset = utils.DatasetSplit_new(train_dataset, backdoor_train_dataset, benign_part, malicious_part, data_idxs)
        else:
            self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def get_model_parameters(self, model):
        """Extract all model parameters as a flattened vector"""
        return torch.cat([param.view(-1) for param in model.parameters()])

    @staticmethod
    def get_model_state(model):
        """Flatten the exact state uploaded to the server (parameters + buffers)."""
        return parameters_to_vector(
            [value for value in model.state_dict().values()]
        )

    @staticmethod
    def _differentiable_state_update(model, initial_state):
        """Build an upload-shaped update while retaining gradients for parameters.

        ``state_dict`` also contains BatchNorm buffers.  Those buffers are not
        differentiable, but including them here makes the attack-side cosine and
        top-k masks operate on exactly the vector observed by the defense.
        """
        named_params = dict(model.named_parameters())
        chunks = []
        pointer = 0
        for name, value in model.state_dict().items():
            current = named_params[name] if name in named_params else value
            numel = value.numel()
            baseline = initial_state[pointer:pointer + numel].view_as(value)
            chunks.append((current - baseline).reshape(-1))
            pointer += numel
        return torch.cat(chunks)
    
    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def _adaptive_weights(self):
        """返回 (w_cos, w_sign, w_div)，对应三种自适应攻击变体：
        - adaptive / adaptive_mdf : 只规避 MDF（匹配正常余弦/符号分数）
        - adaptive_pdc            : 只规避 PDC（匹配干净客户端对统计）
        - adaptive_joint          : 同时规避两阶段
        """
        attack = self.args.attack
        lambda_cos = float(getattr(self.args, "lambda_cos", 1.0))
        lambda_sign = float(getattr(self.args, "lambda_sign", 1.0))
        lambda_div = float(getattr(self.args, "lambda_div", 1.0))
        if attack == "adaptive" or attack == "adaptive_mdf":
            return lambda_cos, lambda_sign, 0.0
        elif attack == "adaptive_pdc":
            return 0.0, 0.0, lambda_div
        elif attack == "adaptive_joint":
            return lambda_cos, lambda_sign, lambda_div
        return 0.0, 0.0, 0.0

    def _adaptive_clean_update(self, global_model, criterion, round, initial_state):
        """Return a cached clean local update used as the MDF proxy direction."""
        if not hasattr(self, "adaptive_clean_train_loader"):
            return None
        if round in self._adaptive_reference_cache:
            return self._adaptive_reference_cache[round]

        reference_model = copy.deepcopy(global_model)
        reference_model.train()
        optimizer = torch.optim.SGD(
            reference_model.parameters(),
            lr=self.args.client_lr * (self.args.lr_decay) ** round,
            weight_decay=self.args.wd,
            momentum=self.args.momentum,
        )
        for _ in range(self.args.local_ep):
            for inputs, labels in self.adaptive_clean_train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device=self.args.device, non_blocking=True)
                labels = labels.to(device=self.args.device, non_blocking=True)
                criterion(reference_model(inputs), labels).backward()
                optimizer.step()

        clean_update = (
            self.get_model_state(reference_model).detach() - initial_state
        )
        # Candidate and refinement share only the current round's reference.
        # Keeping every historical parameter vector would grow GPU memory linearly
        # with the number of communication rounds.
        self._adaptive_reference_cache = {round: clean_update}
        return clean_update

    def adaptive_reference_update(self, round):
        """Return the cached clean proxy for same-round colluding attackers."""
        return self._adaptive_reference_cache.get(round)

    def prepare_adaptive_reference(self, global_model, criterion, round):
        """Prepare this client's clean same-round update without mutating the model."""
        initial_state = self.get_model_state(global_model).detach().clone()
        return self._adaptive_clean_update(
            global_model, criterion, round, initial_state
        )

    @staticmethod
    def _topk_mask(update, ratio):
        """Hard top-k support used by the defense; support selection is detached."""
        flat = update.detach().abs()
        k = max(1, min(flat.numel(), int(flat.numel() * float(ratio))))
        indices = torch.topk(flat, k=k).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices] = True
        return mask

    def _soft_sign_agreement(self, current, target, mask, eps=1e-8):
        """Bounded-gradient approximation of hard sign agreement.

        A tanh temperature based on ``current`` makes the normalized input nearly
        constant and saturates on wrong signs.  This rational approximation uses
        a detached target-based temperature and retains a polynomial gradient on
        both correctly and incorrectly signed coordinates.
        """
        if not torch.any(mask):
            return torch.zeros((), device=current.device)
        current_values = current[mask]
        target_values = target[mask].detach()
        target_sign = torch.sign(target_values)
        nonzero = target_sign.ne(0)
        base_scale = target_values[nonzero].abs().mean() if torch.any(nonzero) else target.abs().mean()
        temperature_ratio = max(
            1e-4, float(getattr(self.args, "adaptive_sign_temperature", 0.1))
        )
        temperature = (base_scale * temperature_ratio).clamp_min(eps)

        soft_sign = current_values / (current_values.abs() + temperature)
        scores = torch.empty_like(current_values)
        scores[nonzero] = (1.0 + soft_sign[nonzero] * target_sign[nonzero]) * 0.5
        # The defense regards sign(0) == sign(0) as agreement.  Smoothly reward
        # coordinates whose target sign is zero for remaining close to zero.
        scores[~nonzero] = temperature / (current_values[~nonzero].abs() + temperature)
        return scores.mean()

    def _stable_cosine(self, current, target, eps=1e-8):
        """Cosine proxy with a reference-scaled floor near a zero current update."""
        target = target.detach()
        target_norm = torch.norm(target)
        if target_norm <= eps:
            return torch.zeros((), device=current.device)
        floor_ratio = max(
            1e-4, float(getattr(self.args, "adaptive_cosine_floor", 0.05))
        )
        radius = (target_norm * floor_ratio).clamp_min(eps)
        current_norm = torch.sqrt(torch.sum(current * current) + radius * radius)
        return torch.clamp(
            torch.dot(current, target) / (current_norm * target_norm), -1.0, 1.0
        )

    @staticmethod
    def _hard_cosine(update_a, update_b, eps=1e-8):
        norm_a = torch.norm(update_a)
        norm_b = torch.norm(update_b)
        if norm_a <= eps or norm_b <= eps:
            return torch.zeros((), device=update_a.device)
        return torch.clamp(
            torch.dot(update_a, update_b) / (norm_a * norm_b), -1.0, 1.0
        ).detach()

    def _mdf_losses(self, current_update, clean_update, median_reference):
        """Match the scores of a clean client against the MedianGuard proxy.

        MedianGuard applies a relative one-sided MZ filter by default.  Driving
        every attacker score to one is unnecessary and destroys benign client
        heterogeneity.  The clean local update supplies a normal target score;
        the coordinate median of colluding clean updates approximates the hidden
        server-side median direction.
        """
        zero = torch.zeros((), device=current_update.device)
        if clean_update is None or median_reference is None:
            return zero, zero

        current_cos = self._stable_cosine(current_update, median_reference)
        target_cos = self._hard_cosine(clean_update, median_reference)

        ratio = float(getattr(self.args, "sparsity", 0.3))
        current_mask = self._topk_mask(current_update, ratio)
        clean_mask = self._topk_mask(clean_update, ratio)
        current_sign = self._soft_sign_agreement(
            current_update, median_reference, current_mask
        )
        target_sign = (
            torch.sign(clean_update[clean_mask])
            == torch.sign(median_reference[clean_mask])
        ).float().mean().detach()

        margin = float(getattr(self.args, "adaptive_mdf_margin", 0.0))
        if bool(getattr(self.args, "median_guard_two_sided", False)):
            cos_loss = (current_cos - target_cos).pow(2)
            sign_loss = (current_sign - target_sign).pow(2)
        else:
            cos_loss = torch.relu(target_cos + margin - current_cos).pow(2)
            sign_loss = torch.relu(target_sign + margin - current_sign).pow(2)
        return cos_loss, sign_loss

    @staticmethod
    def _hard_pdc_features(update_a, update_b, ratio, eps=1e-8):
        """Exact AvgAlign2 pair features, used as detached clean targets."""
        mask_a = Agent._topk_mask(update_a, ratio)
        mask_b = Agent._topk_mask(update_b, ratio)
        common = mask_a & mask_b
        zero = torch.zeros((), device=update_a.device)
        if not torch.any(common):
            return zero, zero
        a = update_a[common]
        b = update_b[common]
        denom = torch.norm(a) * torch.norm(b)
        cosine = torch.dot(a, b) / denom.clamp_min(eps)
        alignment = (torch.sign(a) == torch.sign(b)).float().mean()
        return cosine.detach(), alignment.detach()

    def _soft_pdc_features(self, current_update, target_update, ratio, eps=1e-8):
        """Differentiable AvgAlign2 pair features on the same top-k intersection."""
        current_mask = self._topk_mask(current_update, ratio)
        target_mask = self._topk_mask(target_update, ratio)
        common = current_mask & target_mask
        zero = torch.zeros((), device=current_update.device)
        if not torch.any(common):
            return zero, zero
        current = current_update[common]
        target = target_update[common]
        cosine = self._stable_cosine(current, target, eps=eps)
        alignment = self._soft_sign_agreement(
            current_update, target_update, common, eps=eps
        )
        return cosine, alignment

    def _pdc_adaptive_loss(self, current_update, reference_update,
                           peer_updates, peer_reference_updates):
        """Make poisoned pairs statistically resemble their clean counterparts.

        Blindly forcing malicious updates apart creates outliers and can make the
        benign cluster even easier to select.  Instead, each attacker follows its
        own clean update, while malicious--malicious pair features are matched to
        the features of the corresponding clean-client pair.
        """
        zero = torch.zeros((), device=current_update.device)
        if reference_update is None:
            return zero
        ratio = float(getattr(self.args, "avg_align_topk_ratio", 0.3))

        # The attack should first look like this client's clean local update.
        own_cos, own_align = self._soft_pdc_features(
            current_update, reference_update, ratio
        )
        total = (1.0 - own_cos).pow(2) + (1.0 - own_align).pow(2)

        if not peer_updates or not peer_reference_updates:
            return total

        valid = 0
        for peer_update, peer_reference in zip(peer_updates, peer_reference_updates):
            peer_update = peer_update.to(current_update.device).detach()
            peer_reference = peer_reference.to(current_update.device).detach()
            if (peer_update.numel() != current_update.numel()
                    or peer_reference.numel() != current_update.numel()):
                continue
            current_cos, current_align = self._soft_pdc_features(
                current_update, peer_update, ratio
            )
            target_cos, target_align = self._hard_pdc_features(
                reference_update, peer_reference, ratio
            )
            total = total + (current_cos - target_cos).pow(2)
            total = total + (current_align - target_align).pow(2)
            valid += 1
        return total / (valid + 1)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None,
                    adaptive_peer_updates=None, adaptive_mode="single",
                    adaptive_peer_references=None,
                    adaptive_mdf_reference=None,
                    return_state_update=False):
        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        initial_state = initial_global_model_params.detach().clone()
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)

        is_adaptive = self.is_malicious and self.args.attack in (
            "adaptive", "adaptive_mdf", "adaptive_pdc", "adaptive_joint")
        if is_adaptive:
            w_cos, w_sign, w_div = self._adaptive_weights()
            reference_update = self._adaptive_clean_update(
                global_model, criterion, round, initial_state
            ) if w_cos > 0 or w_sign > 0 or w_div > 0 else None
            if adaptive_mdf_reference is None:
                adaptive_mdf_reference = reference_update
        
        # SoDa attack: Self-reference training phase
        if self.is_malicious and self.args.attack == 'soda' and hasattr(self, 'clean_train_loader'):
            print('Self-reference training')
            temp_model = copy.deepcopy(global_model)
            temp_model.train()
            # SoDa-BNGuard uses fixed learning rate for self-reference training (no decay)
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=self.args.client_lr,
                                        weight_decay=self.args.wd, momentum=self.args.momentum)

            for local_epoch in range(self.args.local_ep):
                start = time.time()
                for i, (inputs, labels) in enumerate(self.clean_train_loader):
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                    labels.to(device=self.args.device, non_blocking=True)

                    outputs = temp_model(inputs)
                    minibatch_loss = criterion(outputs, labels)
                    
                    minibatch_loss.backward()
                    
                    optimizer.step()

                end = time.time()
                train_time = end - start
                print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                        minibatch_loss, train_time))
            print('Self-reference finished')

            fixed_params = self.get_model_parameters(temp_model)
        else:
            fixed_params = None
        
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** round,
                                    weight_decay=self.args.wd, momentum=self.args.momentum)

        regular_loss = 0.0
        for local_epoch in range(self.args.local_ep):
            start = time.time()
            old_gradient = {}
            old_gradient_mine = {}
            old_params = {}
            for i, (inputs, labels) in enumerate(self.train_loader):
                # if i == 0 and self.is_malicious:
                #     save_image(torch.cat([inputs[labels == self.args.target_class][:10]]), '%s_image.png' % self.id, normalize=True, nrow=10)
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                 labels.to(device=self.args.device, non_blocking=True)
                outputs = global_model(inputs)
                # outputs = outputs[:, :]
                minibatch_loss = criterion(outputs, labels)
                # print(minibatch_loss)
                
                # SoDa attack: add regularization loss to keep model close to self-reference model
                if self.is_malicious and self.args.attack == 'soda' and fixed_params is not None:
                    current_params = self.get_model_parameters(global_model)
                    l2_loss = torch.norm(current_params - fixed_params, p=2)
                    cos_loss = torch.nn.functional.cosine_similarity(current_params, fixed_params, dim=0)
                    minibatch_loss = minibatch_loss + 0.1 * l2_loss + 100 * (1 - cos_loss)
                
                # Adaptive attack against MedianGuard + AvgAlign2: both stages
                # operate on client updates, not on complete model parameters.
                if is_adaptive and initial_global_model_params is not None:
                    current_update = self._differentiable_state_update(
                        global_model, initial_state
                    )
                    extra_loss = 0.0

                    if w_cos > 0 or w_sign > 0:
                        cos_loss, sign_loss = self._mdf_losses(
                            current_update, reference_update, adaptive_mdf_reference
                        )
                        extra_loss = extra_loss + w_cos * cos_loss + w_sign * sign_loss

                    if w_div > 0 and adaptive_mode == "refine":
                        extra_loss = extra_loss + w_div * self._pdc_adaptive_loss(
                            current_update, reference_update, adaptive_peer_updates,
                            adaptive_peer_references,
                        )

                    minibatch_loss = minibatch_loss + extra_loss
                
                minibatch_loss.backward()
                if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                    for name, param in global_model.named_parameters():
                        param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
                optimizer.step()

                if self.args.attack == 'pgd' and self.id < self.args.num_corrupt and (i == len(self.train_loader) - 1):
                    if self.args.data == 'cifar10':
                        eps = torch.norm(initial_global_model_params) * 0.1
                    else:
                        eps = torch.norm(initial_global_model_params)

                    current_local_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
                    norm_diff = torch.norm(current_local_model_params - initial_global_model_params)
                    print('clip before: ', norm_diff)
                    if norm_diff > eps:
                        w_proj_vec = eps * (current_local_model_params - initial_global_model_params) / norm_diff + initial_global_model_params

                        print('clip after: ', torch.norm(w_proj_vec - initial_global_model_params))

                        new_state_dict = utils.vector_to_model_wo_load(w_proj_vec, global_model)    
                        global_model.load_state_dict(new_state_dict)

            end = time.time()
            train_time = end - start
            print("local epoch %d \t client: %d \t mal: %s \t loss: %.8f \t time: %.2f" % (local_epoch, self.id, str(self.is_malicious),
                                                                     minibatch_loss, train_time))

        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params
            if return_state_update:
                return self.update
            return self.update

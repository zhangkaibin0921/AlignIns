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
    
    def check_poison_timing(self, round):
        if round > self.args.cease_poison:
            self.train_dataset = utils.DatasetSplit(self.clean_backup_dataset, self.data_idxs)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)

    def local_train(self, global_model, criterion, round=None, neurotoxin_mask=None):
        # print(len(self.train_dataset))
        """ Do a local training over the received global model, return the update """
        # start = time.time()
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.id < self.args.num_corrupt:
            self.check_poison_timing(round)
        
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

            return self.update

import copy
import bisect

import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import logging
from utils import vector_to_model, vector_to_name_param

import sklearn.metrics.pairwise as smp
from geom_median.torch import compute_geometric_median

try:
    from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
except ImportError:  # pragma: no cover
    KMeans = SpectralClustering = DBSCAN = None

try:
    import hdbscan
except ImportError:  # pragma: no cover
    hdbscan = None

try:
    import finch
except ImportError:  # pragma: no cover
    finch = None


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = args.server_lr
        self.n_params = n_params
        
        if self.args.aggr == 'foolsgold':
            self.memory_dict = dict()
            self.wv_history = []
        
        if self.args.aggr == 'scopemm':
            self.tpr_history = []
            self.fpr_history = []
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):


        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.aggr != "rlr":
            lr_vector = lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg' or self.args.aggr == 'rlr' or self.args.aggr == 'lockdown':          
            aggregated_updates = self.agg_avg(agent_updates_dict)
        elif self.args.aggr == 'median':
            aggregated_updates = self.agg_median(agent_updates_dict)
        elif self.args.aggr == 'avg_align':
            aggregated_updates = self.agg_avg_alignment(agent_updates_dict)
        elif self.args.aggr == 'alignins':
            aggregated_updates = self.agg_alignins(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'tda_only':
            aggregated_updates = self.agg_tda_only(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'median_guard':
            aggregated_updates = self.agg_median_guard(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'median_guard_align':
            aggregated_updates = self.agg_median_guard_avg_align(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'median_guard_align2':
            aggregated_updates = self.agg_median_guard_avg_align2(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'trust_clip':
            aggregated_updates = self.agg_trust_clip(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'mpsa':
            aggregated_updates = self.agg_mpsa(agent_updates_dict, cur_global_params)
        elif self.args.aggr == 'mmetric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'scopemm':
            aggregated_updates = self.agg_scope_multimetric(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'scope':
            aggregated_updates = self.agg_scope(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'mpsaguard':
            aggregated_updates = self.agg_mpsa_guard(agent_updates_dict, global_model, cur_global_params)
        elif self.args.aggr == 'foolsgold':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict)
        elif self.args.aggr == 'signguard':
            aggregated_updates = self.agg_signguard(agent_updates_dict)
        elif self.args.aggr == "mkrum":
            aggregated_updates = self.agg_mkrum(agent_updates_dict)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        neurotoxin_mask = {}
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return updates_dict, neurotoxin_mask

    def agg_rfa(self, agent_updates_dict):
        local_updates = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)

        n = len(local_updates)
        temp_updates = torch.stack(local_updates, dim=0)
        weights = torch.ones(n, dtype=torch.float64).to(self.args.device)
        # compute_geometric_median 需要 weights 是 torch tensor，并且需要和 local_updates 在同一设备上
        # 如果 local_updates 在 CUDA，需要先移到 CPU（因为 geom_median 库可能不支持 CUDA）
        if self.args.device != 'cpu':
            local_updates_cpu = [up.cpu() for up in local_updates]
            weights_cpu = weights.cpu()
            temp_updates_cpu = temp_updates.cpu()
        else:
            local_updates_cpu = local_updates
            weights_cpu = weights
            temp_updates_cpu = temp_updates
        gw = compute_geometric_median(local_updates_cpu, weights_cpu).median
        for i in range(2):
            weights_cpu = torch.mul(weights_cpu, torch.exp(-1.0*torch.norm(temp_updates_cpu-gw, dim=1)))
            gw = compute_geometric_median(local_updates_cpu, weights_cpu).median
        # 将结果移回原始设备
        aggregated_model = gw.to(self.args.device)
        return aggregated_model

    def agg_alignins(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []  # 实际干净的客户端ID（真实标签）
        malicious_id = []  # 实际恶意的客户端ID（真实标签）

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id  # 所有客户端ID列表（恶意+干净）
        num_chosen_clients = len(chosen_clients)
        inter_model_updates = torch.stack(local_updates, dim=0)

        # 获取模型层结构信息（用于归一化和统计）
        state_dict = global_model.state_dict()
        layer_ranges = []
        start = 0
        for name, param in state_dict.items():
            length = param.numel()
            layer_ranges.append((name, start, start + length))
            start += length
        layer_starts = [s for (_, s, _) in layer_ranges]

        # 分层归一化（Layer-wise normalization）
        use_layer_norm = getattr(self.args, "alignins_layer_norm", False)
        if use_layer_norm:
            # 对每个客户端的更新进行分层归一化
            normalized_updates = []
            eps = 1e-8
            for update in local_updates:
                normalized_update = torch.zeros_like(update)
                for name, start_idx, end_idx in layer_ranges:
                    layer_grad = update[start_idx:end_idx]
                    layer_norm = torch.norm(layer_grad)
                    if layer_norm > eps:
                        normalized_update[start_idx:end_idx] = layer_grad / layer_norm
                    else:
                        normalized_update[start_idx:end_idx] = layer_grad
                normalized_updates.append(normalized_update)
            inter_model_updates = torch.stack(normalized_updates, dim=0)
            logging.info("[AlignIns] 已应用分层归一化（Layer-wise normalization）")

        # 如果启用了分层归一化，也对全局模型进行归一化（用于 TDA 计算）
        normalized_flat_global_model = flat_global_model
        if use_layer_norm:
            normalized_flat_global_model = torch.zeros_like(flat_global_model)
            eps = 1e-8
            for name, start_idx, end_idx in layer_ranges:
                layer_grad = flat_global_model[start_idx:end_idx]
                layer_norm = torch.norm(layer_grad)
                if layer_norm > eps:
                    normalized_flat_global_model[start_idx:end_idx] = layer_grad / layer_norm
                else:
                    normalized_flat_global_model[start_idx:end_idx] = layer_grad

        # 仅在 VGG9（当前用于 CIFAR-10）下统计 Top-k 梯度在各层的占比
        is_vgg9 = getattr(self.args, "data", "") == "cifar10"
        layer_hit_counts = None
        if is_vgg9:
            layer_hit_counts = {}
            for name, _, _ in layer_ranges:
                layer_hit_counts[name] = 0

        tda_list = []
        mpsa_list = []
        # 计算主符号（基于归一化后的更新，如果启用了归一化）
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        total_topk = 0
        for i in range(len(inter_model_updates)):
            # Top-k 选择（基于归一化后的更新，如果启用了归一化）
            _, init_indices = torch.topk(
                torch.abs(inter_model_updates[i]),
                int(len(inter_model_updates[i]) * self.args.sparsity)
            )
            if is_vgg9 and layer_ranges is not None and len(init_indices) > 0:
                idx_np = init_indices.detach().cpu().numpy().astype(int)
                total_topk += idx_np.size
                for flat_idx in idx_np:
                    # 使用二分查找定位所属参数张量
                    layer_idx = bisect.bisect_right(layer_starts, flat_idx) - 1
                    if 0 <= layer_idx < len(layer_ranges):
                        layer_name = layer_ranges[layer_idx][0]
                        layer_hit_counts[layer_name] += 1
            # 计算MPSA（使用归一化后的更新，如果启用了归一化）
            mpsa = (torch.sum(
                torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]) / torch.numel(
                inter_model_updates[i][init_indices])).item()
            mpsa_list.append(mpsa)
            # 计算TDA（使用归一化后的值）
            tda = cos(inter_model_updates[i], normalized_flat_global_model).item()
            tda_list.append(tda)

        logging.info('TDA: %s' % [round(i, 4) for i in tda_list])
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])

        # 打印 Top-k 梯度在各层的占比（仅 VGG9）
        if is_vgg9 and layer_hit_counts is not None and total_topk > 0:
            layer_ratios = {
                name: count / total_topk for name, count in layer_hit_counts.items() if count > 0
            }
            # 按占比从大到小排序，便于观察主要来源层
            sorted_items = sorted(layer_ratios.items(), key=lambda x: x[1], reverse=True)
            pretty_ratios = {name: round(ratio, 4) for name, ratio in sorted_items}
            logging.info(f"[AlignIns][VGG9] Top-{int(self.args.sparsity * 100)}%% 梯度在各层的占比: {pretty_ratios}")

        ######## MZ-score calculation ########
        # MPSA的MZ-score
        mpsa_std = np.std(mpsa_list) if len(mpsa_list) > 1 else 1e-12
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = [np.abs(m - mpsa_med) / mpsa_std for m in mpsa_list]
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])

        # TDA的MZ-score
        tda_std = np.std(tda_list) if len(tda_list) > 1 else 1e-12
        tda_med = np.median(tda_list)
        mzscore_tda = [np.abs(t - tda_med) / tda_std for t in tda_list]
        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        ######## Anomaly detection with MZ score ########
        # MPSA筛选出的良性索引（chosen_clients的索引）
        benign_idx1 = set(range(num_chosen_clients))
        benign_idx1.intersection_update(
            set(np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s).flatten().astype(int)))

        # TDA筛选出的良性索引（chosen_clients的索引）
        benign_idx2 = set(range(num_chosen_clients))
        benign_idx2.intersection_update(
            set(np.argwhere(np.array(mzscore_tda) < self.args.lambda_c).flatten().astype(int)))

        ######## 计算MPSA和TDA的Precision并打印 ########
        def _calc_precision(selected_indices, chosen_ids, actual_benign_ids):
            """
            计算Precision：真正干净数 / 选中数
            selected_indices：筛选出的索引（对应chosen_ids的索引）
            chosen_ids：所有客户端ID列表（malicious_id + benign_id）
            actual_benign_ids：实际干净的客户端ID列表（真实标签）
            """
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            # 统计选中的客户端中实际干净的数量
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = chosen_ids[idx]  # 索引对应的实际客户端ID
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        # MPSA的Precision
        mpsa_precision, mpsa_selected, mpsa_true_clean = _calc_precision(benign_idx1, chosen_clients, benign_id)
        if mpsa_precision is not None:
            logging.info(
                f"[MPSA] 识别的干净客户端准确率(Precision): {mpsa_precision:.4f}  |  选中数: {mpsa_selected}  真正干净数: {mpsa_true_clean}")
        else:
            logging.info(f"[MPSA] 无选中客户端，无法计算干净客户端准确率")

        # TDA的Precision
        tda_precision, tda_selected, tda_true_clean = _calc_precision(benign_idx2, chosen_clients, benign_id)
        if tda_precision is not None:
            logging.info(
                f"[TDA] 识别的干净客户端准确率(Precision): {tda_precision:.4f}  |  选中数: {tda_selected}  真正干净数: {tda_true_clean}")
        else:
            logging.info(f"[TDA] 无选中客户端，无法计算干净客户端准确率")

        ######## 后续原有逻辑 ########
        benign_set = benign_idx2.intersection(benign_idx1)
        benign_idx = list(benign_set)
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        ######## Post-filtering model clipping ########
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        norm_clip = updates_norm.median(dim=0)[0].item()
        benign_updates = torch.stack(local_updates, dim=0)
        updates_norm = torch.norm(benign_updates, dim=1).reshape((-1, 1))
        updates_norm_clipped = torch.clamp(updates_norm, 0, norm_clip, out=None)
        benign_updates = (benign_updates / updates_norm) * updates_norm_clipped

        ######## 原有TPR/FPR计算 ########
        correct = 0
        for idx in benign_idx:
            if chosen_clients[idx] in benign_id:  # 修正：用实际ID判断是否干净
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for idx in benign_idx:
                if chosen_clients[idx] in malicious_id:  # 修正：用实际ID判断是否恶意
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f' % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {chosen_clients[idx]: benign_updates[idx] for idx in benign_idx}
        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_tda_only(self, agent_updates_dict, global_model, flat_global_model):
        """
        仅使用TDA（Top-k Direction Alignment）的防御方法，参考AlignIns。
        
        特点：
        1. 只计算TDA，不使用MPSA
        2. 通过开关控制TDA的参考向量：
           - use_median_anchor=False（默认）：对照上一轮的global model（类似AlignIns）
           - use_median_anchor=True：对照中位数聚合结果（类似MedianGuard）
        3. 双侧MZ-score过滤，过滤偏离中位数过远的客户端（过高或过低）
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(chosen_clients)
        if num_chosen_clients == 0:
            return torch.zeros_like(flat_global_model)
        if num_chosen_clients == 1:
            return local_updates[0]

        inter_model_updates = torch.stack(local_updates, dim=0)

        # 计算三个关键向量用于分析
        eps = 1e-8
        
        # g_true: 所有良性客户端更新的平均值（Ground Truth）
        # 注意：local_updates 的顺序与 agent_updates_dict 的键顺序一致
        # 需要根据 benign_id 从 agent_updates_dict 中提取对应的更新
        if len(benign_id) > 0:
            benign_dict = {bid: agent_updates_dict[bid] for bid in benign_id if bid in agent_updates_dict}
            if len(benign_dict) > 0:
                g_true = self.agg_avg(benign_dict)
            else:
                g_true = torch.zeros_like(flat_global_model)
                logging.warning("[TDA-Only] 良性客户端不在 agent_updates_dict 中，无法计算 g_true")
        else:
            g_true = torch.zeros_like(flat_global_model)
            logging.warning("[TDA-Only] 无良性客户端，无法计算 g_true")
        
        # g_history: 上一轮的全局更新（AlignIns Anchor）
        g_history = flat_global_model
        
        # g_median: 当前轮次计算出的坐标中位数向量（FedCODA Anchor）
        g_median = torch.median(inter_model_updates, dim=0).values
        
        # 计算余弦相似度
        def _cosine_sim(vec1, vec2):
            norm1 = torch.norm(vec1)
            norm2 = torch.norm(vec2)
            if norm1 > eps and norm2 > eps:
                return float(torch.dot(vec1, vec2).item() / (norm1 * norm2))
            else:
                return 0.0
        
        sim_alignins = _cosine_sim(g_history, g_true)
        sim_fedcoda = _cosine_sim(g_median, g_true)
        
        logging.info(f"[TDA-Only] Sim_AlignIns (Cosine(g_history, g_true)): {sim_alignins:.6f}")
        logging.info(f"[TDA-Only] Sim_FedCODA (Cosine(g_median, g_true)): {sim_fedcoda:.6f}")

        # 确定TDA的参考向量（锚点）
        use_median_anchor = getattr(self.args, "tda_use_median_anchor", False)
        
        if use_median_anchor:
            # 使用中位数聚合作为锚点
            anchor_vector = g_median
            logging.info("[TDA-Only] 使用中位数聚合作为TDA锚点")
        else:
            # 使用上一轮的global model作为锚点（默认，类似AlignIns）
            anchor_vector = g_history
            logging.info("[TDA-Only] 使用上一轮global model作为TDA锚点")

        # 计算TDA（余弦相似度），仅在每个客户端自身的 Top-k 重要坐标上计算
        # Top-k 比例沿用 self.args.sparsity（默认 0.3），与 AlignIns 的 MPSA Top-k 一致
        tda_list = []
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        topk_dim = max(1, int(inter_model_updates.shape[1] * float(getattr(self.args, "sparsity", 0.3))))
        for i in range(len(inter_model_updates)):
            _, topk_idx = torch.topk(torch.abs(inter_model_updates[i]), k=topk_dim)
            vec_i = inter_model_updates[i][topk_idx]
            vec_anchor = anchor_vector[topk_idx]
            tda = cos(vec_i, vec_anchor).item()
            tda_list.append(tda)

        logging.info('[TDA-Only] TDA scores: %s' % [round(i, 4) for i in tda_list])

        # MZ-score计算（双侧过滤）
        tda_std = np.std(tda_list) if len(tda_list) > 1 else 1e-12
        tda_med = np.median(tda_list)
        tda_std = tda_std if tda_std > eps else 1.0
        
        # 双侧MZ-score：计算与中位数的绝对偏差
        mzscore_tda = [np.abs(t - tda_med) / tda_std for t in tda_list]
        
        logging.info('MZ-score of TDA: %s' % [round(i, 4) for i in mzscore_tda])

        # TDA筛选出的良性索引（双侧过滤）
        lambda_c = float(getattr(self.args, "lambda_c", 1.0))
        benign_idx = set(range(num_chosen_clients))
        benign_idx.intersection_update(
            set(np.argwhere(np.array(mzscore_tda) < lambda_c).flatten().astype(int))
        )
        benign_idx = sorted(list(benign_idx))

        # 计算TDA的Precision
        def _calc_precision(selected_indices, chosen_ids, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = chosen_ids[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        tda_precision, tda_selected, tda_true_clean = _calc_precision(benign_idx, chosen_clients, benign_id)
        if tda_precision is not None:
            logging.info(
                f"[TDA-Only] 识别的干净客户端准确率(Precision): {tda_precision:.4f}  |  选中数: {tda_selected}  真正干净数: {tda_true_clean}")
        else:
            logging.info(f"[TDA-Only] 无选中客户端，无法计算干净客户端准确率")

        if len(benign_idx) == 0:
            logging.info("[TDA-Only] 过滤后无客户端被保留，返回零更新")
            return torch.zeros_like(local_updates[0])

        # 直接使用过滤后的更新
        benign_updates = torch.stack([local_updates[i] for i in benign_idx], dim=0)

        # TPR/FPR计算
        correct = 0
        for idx in benign_idx:
            if chosen_clients[idx] in benign_id:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for idx in benign_idx:
                if chosen_clients[idx] in malicious_id:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('[TDA-Only] benign update index:   %s' % str(benign_id))
        logging.info('[TDA-Only] selected update index: %s' % str([chosen_clients[idx] for idx in benign_idx]))
        logging.info('[TDA-Only] FPR:       %.4f' % FPR)
        logging.info('[TDA-Only] TPR:       %.4f' % TPR)

        # 加权平均聚合
        current_dict = {chosen_clients[idx]: benign_updates[i] for i, idx in enumerate(benign_idx)}
        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def _median_guard_select(self, agent_updates_dict, flat_global_model):
        """
        核心筛选逻辑：返回通过 MedianGuard 过滤后的客户端字典以及可选的回退值。
        回退值用于处理 degenerate 情况（例如 tmp 近似 0 向量）。
        """
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        client_ids = malicious_id + benign_id
        num_clients = len(client_ids)
        if num_clients == 0:
            return {}, torch.zeros_like(flat_global_model)
        ordered_updates = [agent_updates_dict[cid] for cid in client_ids]
        if num_clients == 1:
            # 只有一个客户端，直接保留该客户端
            return {client_ids[0]: ordered_updates[0]}, None

        # 按照 client_ids 的顺序重新堆叠 updates（保持与 malicious+benign 顺序一致）
        stacked = torch.stack(ordered_updates, dim=0)  # [N, D]

        # 1) 中位数聚合得到 tmp
        tmp = torch.median(stacked, dim=0).values  # [D]

        # 2) 以 tmp 的符号作为“主符号”，计算 MPSA-like & TDA-like
        sparsity = float(getattr(self.args, "sparsity", 0.3))
        lambda_s = float(getattr(self.args, "lambda_s", 1.0))
        lambda_c = float(getattr(self.args, "lambda_c", 1.0))
        eps = float(getattr(self.args, "eps", 1e-12))

        major_sign = torch.sign(tmp)
        tmp_norm = torch.norm(tmp).item()
        if tmp_norm <= eps:
            # 锚点几乎为零向量时，直接退化为纯中位数聚合
            logging.info("[MedianGuard] tmp 近似零向量，退化为纯中位数聚合")
            return {}, tmp

        dim = tmp.numel()
        topk_dim = max(1, int(dim * sparsity))

        mpsa_scores = []
        tda_scores = []
        for i in range(num_clients):
            vec = stacked[i]

            # Top-k 重要坐标（按绝对值）
            _, topk_idx = torch.topk(torch.abs(vec), k=topk_dim)

            # MPSA-like：与 tmp 主符号的一致性
            sign_vec = torch.sign(vec[topk_idx])
            agree = torch.sum(sign_vec == major_sign[topk_idx]).item()
            mpsa = agree / float(topk_dim)
            mpsa_scores.append(mpsa)

            # TDA-like：与 tmp 的余弦相似度
            vec_norm = torch.norm(vec).item()
            if vec_norm > eps:
                tda = float(torch.dot(vec, tmp).item() / (vec_norm * tmp_norm))
            else:
                tda = 0.0
            tda_scores.append(tda)

        logging.info('[MedianGuard] MPSA-like scores: %s' % [round(x, 4) for x in mpsa_scores])
        logging.info('[MedianGuard] TDA-like scores: %s' % [round(x, 4) for x in tda_scores])

        # 3) MZ-score 过滤
        use_two_sided_mz = bool(getattr(self.args, "median_guard_two_sided", False))

        def _mz_filter(scores, lam, tag):
            """
            MZ 过滤：
            - 默认单侧：仅过滤相似度过低（比分布中心显著更低）的客户端
            - 若启用 two-sided：相似度过高或过低都会被过滤
            """
            arr = np.array(scores, dtype=np.float64)
            if arr.size <= 1:
                # 只有 1 个或 0 个客户端时，直接全部保留
                return set(range(len(scores)))
            std = np.std(arr)
            med = np.median(arr)
            std = std if std > 1e-12 else 1.0

            if use_two_sided_mz:
                mz = np.abs(arr - med) / std
                logging.info(f"[MedianGuard] Two-sided MZ-score of {tag}: %s" %
                             [round(v, 4) for v in mz])
                keep_mask = mz < lam
            else:
                # 单侧 MZ-score：只看“低于中位数”的幅度
                mz_low = (med - arr) / std
                logging.info(f"[MedianGuard] One-sided MZ-score(low) of {tag}: %s" %
                             [round(v, 4) for v in mz_low])
                # 保留条件：不过低 => mz_low < lam （或 score >= med - lam*std）
                keep_mask = mz_low < lam

            keep = set(np.argwhere(keep_mask).flatten().astype(int).tolist())
            return keep

        keep_mpsa = _mz_filter(mpsa_scores, lambda_s, "MPSA")
        keep_tda = _mz_filter(tda_scores, lambda_c, "TDA")

        # 交集作为最终保留的客户端索引（相对于 client_ids 的索引）
        keep_idx_set = keep_mpsa.intersection(keep_tda)
        keep_idx = sorted(list(keep_idx_set))

        if len(keep_idx) == 0:
            logging.info("[MedianGuard] 过滤后无客户端被保留，返回零更新")
            return {}, torch.zeros_like(flat_global_model)

        # 统计 TPR / FPR
        correct = 0
        for idx in keep_idx:
            if client_ids[idx] in benign_id:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for idx in keep_idx:
                if client_ids[idx] in malicious_id:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('[MedianGuard] benign update index:   %s' % str(benign_id))
        logging.info('[MedianGuard] selected update index: %s' % str([client_ids[i] for i in keep_idx]))
        logging.info('[MedianGuard] FPR:       %.4f' % FPR)
        logging.info('[MedianGuard] TPR:       %.4f' % TPR)

        selected_updates = {client_ids[i]: ordered_updates[i] for i in keep_idx}
        return selected_updates, None

    def agg_median_guard(self, agent_updates_dict, flat_global_model):
        """
        Median-anchored defense (参考 AlignIns 思路，但以中位数聚合结果为“主符号”与锚点):
        1. 先对所有客户端更新做逐坐标中位数聚合，得到 tmp（全局中位数更新）。
        2. 以 tmp 的符号作为“主符号”，对每个客户端计算：
           - MPSA-like：在 top-k 重要坐标上，与 tmp 主符号的一致比例（主符号相似度）
           - TDA-like：整体向量与 tmp 的余弦相似度
        3. 分别对这两个分数做 MZ-score，低于阈值 lambda_s / lambda_c 的视为异常，剔除。
        4. 对剩余客户端做加权平均聚合。
        """
        selected_updates, fallback = self._median_guard_select(agent_updates_dict, flat_global_model)
        if fallback is not None:
            return fallback
        if len(selected_updates) == 0:
            return torch.zeros_like(flat_global_model)
        aggregated_update = self.agg_avg(selected_updates)
        return aggregated_update

    def agg_median_guard_avg_align(self, agent_updates_dict, global_model, flat_global_model):
        """
        组合防御：先运行 MedianGuard 过滤客户端，再对保留下来的客户端执行 avg_align。
        """
        selected_updates, fallback = self._median_guard_select(agent_updates_dict, flat_global_model)
        if fallback is not None:
            return fallback
        if len(selected_updates) == 0:
            return torch.zeros_like(flat_global_model)
        # 在过滤后的客户端集合上运行 avg_align（内部已包含加权聚合逻辑）
        # aggregated_update = self.agg_avg_alignment(selected_updates)

        selected_updates = self.agg_avg_alignment2(selected_updates)
        use_trust_clip = bool(getattr(self.args, "trust_clip_after_mg2", False))
        if use_trust_clip:
            logging.info("[MedianGuard+AvgAlign2] 已开启 trust_clip_after_mg2，对过滤后的客户端执行 trust_clip")
            aggregated_update = self.agg_trust_clip(selected_updates, global_model, flat_global_model)
        else:
            aggregated_update = self.agg_avg(selected_updates)

        return aggregated_update

    def agg_median_guard_avg_align2(self, agent_updates_dict, global_model, flat_global_model):
        """
        组合防御2：先运行 avg_align，再运行 MedianGuard 过滤。
        - 先用 avg_align 得到基线聚合结果 baseline_update
        - 再用 MedianGuard 过滤客户端；如无客户端通过，回退到 baseline_update
        - 如果开启 trust_clip_after_mg2 开关，则对保留客户端再执行 trust_clip，否则直接 agg_avg
        """
        baseline_update = self.agg_avg_alignment(agent_updates_dict)

        # 先用 avg_align2 得到筛选出的“良性”客户端更新集合
        selected_updates = self.agg_avg_alignment2(agent_updates_dict)

        if len(selected_updates) == 0:
            logging.info("[MedianGuard+AvgAlign2] avg_align2 未筛出客户端，回退使用 avg_align 聚合结果")
            return baseline_update

        selected_updates, fallback = self._median_guard_select(selected_updates, flat_global_model)
        if fallback is not None:
            return fallback
        if len(selected_updates) == 0:
            logging.info("[MedianGuard+AvgAlign2] 过滤后无客户端通过，回退使用 avg_align 聚合结果")
            return baseline_update

        use_trust_clip = bool(getattr(self.args, "trust_clip_after_mg2", False) )
        if use_trust_clip:
            logging.info("[MedianGuard+AvgAlign2] 已开启 trust_clip_after_mg2，对过滤后的客户端执行 trust_clip")
            aggregated_update = self.agg_trust_clip(selected_updates, global_model, flat_global_model)
        else:
            aggregated_update = self.agg_avg(selected_updates)

        return aggregated_update

    def agg_trust_clip(self, agent_updates_dict, global_model, flat_global_model):
        """
        Trust-Clip 防御：
        分层自适应裁剪，防止攻击集中于某一层被全局范数掩盖：
        1) 对每一层计算可信方向 g_trust_layer（该层更新的坐标中位数）
        2) 对每一层计算基准幅度 R_base_layer（该层更新 L2 范数的中位数）
        3) 对每个客户端、每一层按与 g_trust_layer 的余弦相似度自适应裁剪：
           alpha_k = max(0, cos_sim)
           R_k = R_base_layer * (beta + (1 - beta) * alpha_k)
           若 ||g_layer|| > R_k，则按比例缩放到 R_k
        4) 将各层裁剪后拼回向量，再按样本量加权平均
        """
        if len(agent_updates_dict) == 0:
            return torch.zeros_like(flat_global_model)

        beta = float(getattr(self.args, "trust_clip_beta", 0.5))

        client_ids = list(agent_updates_dict.keys())
        local_updates = [agent_updates_dict[cid] for cid in client_ids]
        stacked = torch.stack(local_updates, dim=0)  # [N, D]

        # 构建层范围
        state_dict = global_model.state_dict()
        layer_ranges = []
        start = 0
        for name, param in state_dict.items():
            length = param.numel()
            layer_ranges.append((name, start, start + length))
            start += length

        eps = 1e-12

        # 预计算每层的可信方向与基准幅度
        layer_trust = {}
        layer_rbase = {}
        for name, s, e in layer_ranges:
            layer_stack = stacked[:, s:e]  # [N, layer_dim]
            g_trust_layer = torch.median(layer_stack, dim=0).values
            norms_layer = torch.norm(layer_stack, dim=1)
            r_base_layer = torch.median(norms_layer).item()
            layer_trust[name] = g_trust_layer
            layer_rbase[name] = r_base_layer

        # 对每个客户端分层裁剪
        clipped_updates = []
        for idx in range(len(local_updates)):
            g = local_updates[idx]
            clipped_chunks = []
            for name, s, e in layer_ranges:
                g_layer = g[s:e]
                trust_layer = layer_trust[name]
                r_base_layer = layer_rbase[name]

                # 余弦相似度
                denom = (torch.norm(g_layer) * torch.norm(trust_layer) + eps)
                cos_sim = torch.dot(g_layer, trust_layer).item() / denom
                alpha_k = max(0.0, cos_sim)

                # 动态阈值
                R_k = r_base_layer * (beta + (1.0 - beta) * alpha_k)

                g_norm = torch.norm(g_layer).item()
                if g_norm > R_k and R_k > 0:
                    scaling = R_k / (g_norm + eps)
                    clipped_layer = g_layer * scaling
                else:
                    clipped_layer = g_layer
                clipped_chunks.append(clipped_layer)

            clipped_g = torch.cat(clipped_chunks)
            clipped_updates.append(clipped_g)

        # 加权平均
        clipped_dict = {cid: clipped_updates[i] for i, cid in enumerate(client_ids)}
        aggregated_update = self.agg_avg(clipped_dict)
        return aggregated_update

    def agg_mpsa(self, agent_updates_dict, flat_global_model):
        """
        仅保留 AlignIns 中的 MPSA 检测逻辑，去掉 TDA 相关计算以及后续裁剪步骤。
        """
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(chosen_clients)
        inter_model_updates = torch.stack(local_updates, dim=0)

        mpsa_list = []
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(
                torch.abs(inter_model_updates[i]),
                int(len(inter_model_updates[i]) * self.args.sparsity)
            )
            mpsa = (
                torch.sum(
                    torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]
                ) / torch.numel(inter_model_updates[i][init_indices])
            ).item()
            mpsa_list.append(mpsa)

        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])

        mpsa_std = np.std(mpsa_list) if len(mpsa_list) > 1 else 1e-12
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = [np.abs(m - mpsa_med) / mpsa_std for m in mpsa_list]
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])

        benign_idx = set(range(num_chosen_clients))
        benign_idx.intersection_update(
            set(np.argwhere(np.array(mzscore_mpsa) < self.args.lambda_s).flatten().astype(int))
        )
        benign_idx = sorted(list(benign_idx))
        if len(benign_idx) == 0:
            return torch.zeros_like(local_updates[0])

        def _calc_precision(selected_indices, chosen_ids, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = chosen_ids[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        mpsa_precision, mpsa_selected, mpsa_true_clean = _calc_precision(benign_idx, chosen_clients, benign_id)
        if mpsa_precision is not None:
            logging.info(
                f"[MPSA-Only] 识别的干净客户端准确率(Precision): {mpsa_precision:.4f}  |  选中数: {mpsa_selected}  真正干净数: {mpsa_true_clean}"
            )
        else:
            logging.info(f"[MPSA-Only] 无选中客户端，无法计算干净客户端准确率")

        correct = 0
        for idx in benign_idx:
            if chosen_clients[idx] in benign_id:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for idx in benign_idx:
                if chosen_clients[idx] in malicious_id:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f' % FPR)
        logging.info('TPR:       %.4f' % TPR)

        current_dict = {chosen_clients[idx]: local_updates[idx] for idx in benign_idx}
        aggregated_update = self.agg_avg(current_dict)
        return aggregated_update

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    def agg_avg_alignment(self, agent_updates_dict):
        """
        FedAvg-style aggregation with sign-alignment diagnostics.
        When clustering is enabled, the largest cluster is treated as the benign set
        and only those client updates are aggregated.
        """
        if len(agent_updates_dict) == 0:
            raise ValueError("agent_updates_dict must not be empty for avg_align aggregation")

        client_ids = list(agent_updates_dict.keys())
        local_updates = [agent_updates_dict[cid] for cid in client_ids]
        stacked = torch.stack(local_updates, dim=0)
        n_clients, dim = stacked.shape

        topk_ratio = float(getattr(self.args, "avg_align_topk_ratio", 0.3))
        topk_dim = max(1, int(dim * topk_ratio))
        abs_updates = torch.abs(stacked)
        topk_indices = torch.topk(abs_updates, k=topk_dim, dim=1).indices
        topk_mask = torch.zeros_like(stacked, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)

        sign_updates = torch.sign(stacked)
        align_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        cosine_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        # 二维特征矩阵: [n_clients, n_clients, 2] - 每个元素是 [cosine_sim, align_score]
        feature_matrix = np.zeros((n_clients, n_clients, 2), dtype=np.float64)

        benign_pairs = []
        mixed_pairs = []
        malicious_pairs = []
        benign_cosine_pairs = []
        mixed_cosine_pairs = []
        malicious_cosine_pairs = []
        num_corrupt = int(getattr(self.args, "num_corrupt", 0))
        eps = 1e-8
        
        # 用于统计选择的梯度比例
        gradient_ratio_list = []

        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j:
                    # 对角线：符号对齐率为1，余弦相似度为0（不考虑客户端与自身）
                    align_matrix[i, j] = 1.0
                    cosine_matrix[i, j] = 0.0
                    feature_matrix[i, j] = [0.0, 1.0]
                    continue

                common_mask = topk_mask[i] & topk_mask[j]
                intersect_count = int(common_mask.sum().item())
                
                # 计算选择的梯度占全部梯度的比例
                gradient_ratio = intersect_count / dim if dim > 0 else 0.0
                gradient_ratio_list.append(gradient_ratio)
                
                if intersect_count == 0:
                    align_score = 0.0
                    cosine_sim = 0.0
                else:
                    # 计算符号一致性
                    same_sign = torch.sum(sign_updates[i][common_mask] == sign_updates[j][common_mask]).item()
                    align_score = float(same_sign / intersect_count)
                    
                    # 计算余弦相似度（使用相同的 top30% 梯度）
                    vec_i = stacked[i][common_mask]
                    vec_j = stacked[j][common_mask]
                    norm_i = torch.norm(vec_i).item()
                    norm_j = torch.norm(vec_j).item()
                    if norm_i > eps and norm_j > eps:
                        cosine_sim = float(torch.dot(vec_i, vec_j).item() / (norm_i * norm_j))
                    else:
                        cosine_sim = 0.0

                align_matrix[i, j] = align_matrix[j, i] = align_score
                cosine_matrix[i, j] = cosine_matrix[j, i] = cosine_sim
                feature_matrix[i, j] = feature_matrix[j, i] = [cosine_sim, align_score]

                id_i = client_ids[i]
                id_j = client_ids[j]
                if id_i >= num_corrupt and id_j >= num_corrupt:
                    benign_pairs.append(align_score)
                    benign_cosine_pairs.append(cosine_sim)
                elif id_i < num_corrupt and id_j < num_corrupt:
                    malicious_pairs.append(align_score)
                    malicious_cosine_pairs.append(cosine_sim)
                else:
                    mixed_pairs.append(align_score)
                    mixed_cosine_pairs.append(cosine_sim)

        # 打印选择的梯度占全部梯度的比例统计
        if len(gradient_ratio_list) > 0:
            gradient_ratio_arr = np.array(gradient_ratio_list)
            logging.info(
                f"[AvgAlign] 选择的梯度占全部梯度的比例统计: "
                f"mean={gradient_ratio_arr.mean():.6f}, "
                f"std={gradient_ratio_arr.std():.6f}, "
                f"min={gradient_ratio_arr.min():.6f}, "
                f"max={gradient_ratio_arr.max():.6f}, "
                f"median={np.median(gradient_ratio_arr):.6f}"
            )

        logging.info("[AvgAlign] Pairwise sign-alignment matrix: %s", np.round(align_matrix, 3).tolist())
        logging.info("[AvgAlign] Pairwise cosine similarity matrix: %s", np.round(cosine_matrix, 3).tolist())

        def _log_stats(tag, values):
            if len(values) == 0:
                logging.info(f"[AvgAlign][{tag}] 无对应客户端对")
                return
            arr = np.array(values, dtype=np.float64)
            logging.info(
                f"[AvgAlign][{tag}] count={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}, "
                f"min={arr.min():.4f}, max={arr.max():.4f}"
            )

        _log_stats("Benign-Benign (Align)", benign_pairs)
        _log_stats("Benign-Malicious (Align)", mixed_pairs)
        _log_stats("Malicious-Malicious (Align)", malicious_pairs)
        _log_stats("Benign-Benign (Cosine)", benign_cosine_pairs)
        _log_stats("Benign-Malicious (Cosine)", mixed_cosine_pairs)
        _log_stats("Malicious-Malicious (Cosine)", malicious_cosine_pairs)


         # Min-Max normalization of the feature matrix (二维特征归一化)
        # 归一化时排除对角线数据（客户端与自身的数据）
        eps = 1e-12
        # 分别对余弦相似度和符号一致性进行归一化，只使用上三角（不含对角线，k=1）
        triu_indices = np.triu_indices_from(cosine_matrix, k=1)
        cosine_vals = cosine_matrix[triu_indices]
        align_vals = align_matrix[triu_indices]
        
        # 归一化余弦相似度（基于非对角线数据）
        if len(cosine_vals) > 0:
            cosine_min = np.min(cosine_vals)
            cosine_max = np.max(cosine_vals)
        else:
            cosine_min = 0.0
            cosine_max = 1.0
        
        if abs(cosine_max - cosine_min) < eps:
            cosine_matrix_normalized = cosine_matrix.copy()
            logging.info("[AvgAlign] 余弦相似度矩阵所有非对角线值相同，跳过标准化")
        else:
            # 归一化整个矩阵（包括对角线），但归一化参数基于非对角线数据
            cosine_matrix_normalized = (cosine_matrix - cosine_min) / (cosine_max - cosine_min)
            # 对角线保持为0（客户端与自身的余弦相似度不考虑，设为0）
            np.fill_diagonal(cosine_matrix_normalized, 0.0)
            logging.info(f"[AvgAlign] 余弦相似度矩阵 Min-Max 标准化（排除对角线）: min={cosine_min:.4f}, max={cosine_max:.4f}")
        
        # 归一化符号一致性（基于非对角线数据）
        if len(align_vals) > 0:
            align_min = np.min(align_vals)
            align_max = np.max(align_vals)
        else:
            align_min = 0.0
            align_max = 1.0
        
        if abs(align_max - align_min) < eps:
            align_matrix_normalized = align_matrix.copy()
            logging.info("[AvgAlign] 对齐矩阵所有非对角线值相同，跳过标准化")
        else:
            # 归一化整个矩阵（包括对角线），但归一化参数基于非对角线数据
            align_matrix_normalized = (align_matrix - align_min) / (align_max - align_min)
            # 对角线保持为1（客户端与自身的符号对齐率）
            np.fill_diagonal(align_matrix_normalized, 1.0)
            logging.info(f"[AvgAlign] 对齐矩阵 Min-Max 标准化（排除对角线）: min={align_min:.4f}, max={align_max:.4f}")
        
        # 构建归一化后的二维特征矩阵
        feature_matrix_normalized = np.zeros((n_clients, n_clients, 2), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                feature_matrix_normalized[i, j] = [cosine_matrix_normalized[i, j], align_matrix_normalized[i, j]]
        
        logging.info("[AvgAlign] Normalized pairwise sign-alignment matrix: %s", np.round(align_matrix_normalized, 3).tolist())
        logging.info("[AvgAlign] Normalized pairwise cosine similarity matrix: %s", np.round(cosine_matrix_normalized, 3).tolist())

        # Extract normalized pairs from the normalized matrix
        benign_pairs_normalized = []
        mixed_pairs_normalized = []
        malicious_pairs_normalized = []
        benign_cosine_normalized = []
        mixed_cosine_normalized = []
        malicious_cosine_normalized = []
        for i in range(n_clients):
            for j in range(i + 1, n_clients):  # Skip diagonal (i == j)
                id_i = client_ids[i]
                id_j = client_ids[j]
                normalized_align = align_matrix_normalized[i, j]
                normalized_cosine = cosine_matrix_normalized[i, j]
                if id_i >= num_corrupt and id_j >= num_corrupt:
                    benign_pairs_normalized.append(normalized_align)
                    benign_cosine_normalized.append(normalized_cosine)
                elif id_i < num_corrupt and id_j < num_corrupt:
                    malicious_pairs_normalized.append(normalized_align)
                    malicious_cosine_normalized.append(normalized_cosine)
                else:
                    mixed_pairs_normalized.append(normalized_align)
                    mixed_cosine_normalized.append(normalized_cosine)

        logging.info("[AvgAlign] Normalized statistics:")
        _log_stats("Benign-Benign (Align Normalized)", benign_pairs_normalized)
        _log_stats("Benign-Malicious (Align Normalized)", mixed_pairs_normalized)
        _log_stats("Malicious-Malicious (Align Normalized)", malicious_pairs_normalized)
        _log_stats("Benign-Benign (Cosine Normalized)", benign_cosine_normalized)
        _log_stats("Benign-Malicious (Cosine Normalized)", mixed_cosine_normalized)
        _log_stats("Malicious-Malicious (Cosine Normalized)", malicious_cosine_normalized)

        cluster_method = getattr(self.args, "align_cluster_method", "none").lower()
        selected_indices = list(range(n_clients))

        def _report_clusters(labels, method_name):
            unique_labels = sorted(set(labels))
            stats = []
            for cluster_id in unique_labels:
                member_idx = np.where(labels == cluster_id)[0]
                if len(member_idx) == 0:
                    continue
                member_ids = [client_ids[idx] for idx in member_idx]
                benign_cnt = sum(1 for cid in member_ids if cid >= num_corrupt)
                malicious_cnt = len(member_ids) - benign_cnt
                majority_type = "Benign" if benign_cnt >= malicious_cnt else "Malicious"
                logging.info(
                    f"[AvgAlign][{method_name}][cluster={cluster_id}] size={len(member_ids)} ({majority_type}), "
                    f"benign={benign_cnt}, malicious={malicious_cnt}, members={member_ids}"
                )
                if len(member_idx) >= 2:
                    # 报告符号一致性统计
                    sub_align = align_matrix_normalized[np.ix_(member_idx, member_idx)]
                    tril = np.tril_indices_from(sub_align, k=-1)
                    if len(tril[0]) > 0:
                        align_vals = sub_align[tril]
                        logging.info(
                            f"[AvgAlign][{method_name}][cluster={cluster_id}] pairwise align mean={align_vals.mean():.4f}, "
                            f"std={align_vals.std():.4f}, min={align_vals.min():.4f}, max={align_vals.max():.4f}"
                        )
                    # 报告余弦相似度统计
                    sub_cosine = cosine_matrix_normalized[np.ix_(member_idx, member_idx)]
                    if len(tril[0]) > 0:
                        cosine_vals = sub_cosine[tril]
                        logging.info(
                            f"[AvgAlign][{method_name}][cluster={cluster_id}] pairwise cosine mean={cosine_vals.mean():.4f}, "
                            f"std={cosine_vals.std():.4f}, min={cosine_vals.min():.4f}, max={cosine_vals.max():.4f}"
                        )
                    # 报告二维特征距离统计
                    sub_dist = feature_dist_matrix[np.ix_(member_idx, member_idx)]
                    if len(tril[0]) > 0:
                        dist_vals = sub_dist[tril]
                        logging.info(
                            f"[AvgAlign][{method_name}][cluster={cluster_id}] pairwise feature distance mean={dist_vals.mean():.4f}, "
                            f"std={dist_vals.std():.4f}, min={dist_vals.min():.4f}, max={dist_vals.max():.4f}"
                        )
                stats.append((cluster_id, len(member_idx), majority_type, member_idx))
            return stats

        # 从归一化后的二维特征计算距离矩阵（用于聚类）
        # 距离 = 1 - 相似度，其中相似度基于二维特征的欧氏距离
        # 对于每个客户端对 (i, j)，计算二维特征 [cosine_sim, align_score] 的欧氏距离
        feature_dist_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    feature_dist_matrix[i, j] = 0.0
                else:
                    # 计算归一化后的二维特征的欧氏距离
                    feat_i = feature_matrix_normalized[i, j]  # [cosine_sim, align_score]
                    # 由于特征矩阵是对称的，feat_i 和 feat_j 应该相同
                    # 但为了清晰，我们使用标准的距离计算方式
                    # 实际上，对于对称矩阵，我们可以直接计算 1 - 相似度
                    # 这里我们使用：距离 = sqrt((1-cosine)^2 + (1-align)^2)
                    cosine_sim = feat_i[0]
                    align_score = feat_i[1]
                    # 将相似度转换为距离：距离越大，相似度越小
                    dist = np.sqrt((1.0 - cosine_sim) ** 2 + (1.0 - align_score) ** 2)
                    feature_dist_matrix[i, j] = dist
        
        logging.info("[AvgAlign] 基于二维特征的距离矩阵已计算完成")
        logging.info("[AvgAlign] Feature distance matrix: %s", np.round(feature_dist_matrix, 3).tolist())

        cluster_labels = None
        cluster_stats = []
        if cluster_method != "none":
            if n_clients < 2:
                logging.info("[AvgAlign][Cluster] 客户端数量不足，跳过聚类（n_clients=%d）", n_clients)
            else:
                if cluster_method == "kmeans":
                    if KMeans is None:  # pragma: no cover
                        logging.warning("[AvgAlign][KMeans] sklearn 未安装，无法执行 KMeans 聚类")
                    else:
                        cluster_k = int(getattr(self.args, "align_cluster_k", 2))
                        cluster_k = max(1, min(cluster_k, n_clients))
                        if cluster_k == 1:
                            logging.info("[AvgAlign][KMeans] 聚类数为1，所有客户端视为同一簇")
                            cluster_labels = np.zeros(n_clients, dtype=int)
                        else:
                            # 使用二维特征：每个客户端表示为与其他所有客户端的平均特征向量
                            # 或者直接使用距离矩阵进行 KMeans（需要转换为特征向量）
                            # 这里我们使用每个客户端与其他所有客户端的平均二维特征
                            client_features = np.zeros((n_clients, 2), dtype=np.float64)
                            for i in range(n_clients):
                                # 计算客户端 i 与其他所有客户端的平均二维特征
                                other_features = feature_matrix_normalized[i, :, :]  # [n_clients, 2]
                                # 排除自己（对角线）
                                mask = np.arange(n_clients) != i
                                if mask.any():
                                    client_features[i] = np.mean(other_features[mask], axis=0)
                                else:
                                    client_features[i] = [1.0, 1.0]  # 默认值
                            
                            logging.info("[AvgAlign][KMeans] 使用二维特征进行聚类: [平均余弦相似度, 平均符号一致性]")
                            try:
                                kmeans = KMeans(n_clusters=cluster_k, n_init="auto", random_state=42)
                            except TypeError:
                                kmeans = KMeans(n_clusters=cluster_k, n_init=10, random_state=42)
                            cluster_labels = kmeans.fit_predict(client_features)
                elif cluster_method == "spectral":
                    if SpectralClustering is None:  # pragma: no cover
                        logging.warning("[AvgAlign][Spectral] sklearn 未安装，无法执行谱聚类")
                    else:
                        cluster_k = int(getattr(self.args, "align_spectral_clusters", 2))
                        cluster_k = max(2, min(cluster_k, n_clients))
                        # 将距离矩阵转换为相似度矩阵（用于谱聚类）
                        similarity_matrix = 1.0 / (1.0 + feature_dist_matrix)
                        np.fill_diagonal(similarity_matrix, 1.0)
                        spectral = SpectralClustering(
                            n_clusters=cluster_k, affinity="precomputed", assign_labels="kmeans", random_state=42
                        )
                        cluster_labels = spectral.fit_predict(similarity_matrix)
                elif cluster_method == "dbscan":
                    if DBSCAN is None:  # pragma: no cover
                        logging.warning("[AvgAlign][DBSCAN] sklearn 未安装，无法执行 DBSCAN 聚类")
                    else:
                        eps = float(getattr(self.args, "align_dbscan_eps", 0.3))
                        min_samples = int(getattr(self.args, "align_dbscan_min_samples", 2))
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                        cluster_labels = dbscan.fit_predict(feature_dist_matrix)
                elif cluster_method == "hdbscan":
                    if hdbscan is None:  # pragma: no cover
                        logging.warning("[AvgAlign][HDBSCAN] hdbscan 未安装，无法执行 HDBSCAN 聚类")
                    else:
                        clusterer = hdbscan.HDBSCAN(
                            metric="precomputed",
                            min_cluster_size=max(2, n_clients // 2 + 1),
                            min_samples=1,
                            allow_single_cluster=True,
                        )
                        cluster_labels = clusterer.fit(feature_dist_matrix).labels_
                elif cluster_method == "finch":
                    if finch is None:  # pragma: no cover
                        logging.warning("[AvgAlign][FINCH] finch 未安装，无法执行 FINCH 聚类")
                    else:
                        try:
                            # 使用基于二维特征的距离矩阵
                            np.fill_diagonal(feature_dist_matrix, 0.0)  # 确保对角线为0
                            
                            try:
                                c, num_clust, req_c = finch.FINCH(
                                    feature_dist_matrix,
                                    initial_rank=None,
                                    req_clust=None,
                                    distance='precomputed',
                                    ensure_early_termination=True,
                                    verbose=False
                                )
                            except (TypeError, ValueError):
                                # 如果 precomputed 不工作，尝试使用特征向量
                                client_features = np.zeros((n_clients, 2), dtype=np.float64)
                                for i in range(n_clients):
                                    mask = np.arange(n_clients) != i
                                    if mask.any():
                                        client_features[i] = np.mean(feature_matrix_normalized[i, mask, :], axis=0)
                                    else:
                                        client_features[i] = [1.0, 1.0]
                                c, num_clust, req_c = finch.FINCH(
                                    client_features,
                                    initial_rank=None,
                                    req_clust=None,
                                    distance='euclidean',
                                    ensure_early_termination=True,
                                    verbose=False
                                )
                            
                            if c is not None and c.shape[1] > 0:
                                cluster_labels = c[:, -1].astype(int)
                                logging.info(f"[AvgAlign][FINCH] 检测到 {num_clust} 个聚类")
                            else:
                                logging.warning("[AvgAlign][FINCH] FINCH 未返回有效聚类结果，退化为使用全部客户端")
                                cluster_labels = None
                        except Exception as e:
                            logging.warning(f"[AvgAlign][FINCH] FINCH 聚类执行失败: {e}，退化为使用全部客户端")
                            cluster_labels = None
                else:
                    logging.warning(f"[AvgAlign][Cluster] 未知聚类方法: {cluster_method}")

        if cluster_labels is not None:
            # 先打印各簇的统计信息（仍然保留原有日志）
            cluster_stats = _report_clusters(cluster_labels, cluster_method.upper())

            # 基于“簇质量（Cohesion）”的选择策略：
            # Score_k = Size_k * AvgPairwiseCosine_k
            # - Size_k：簇大小（成员数）
            # - AvgPairwiseCosine_k：簇内两两客户端余弦相似度的平均值（使用归一化后的 cosine_matrix_normalized）
            best_cluster = None  # (cluster_id, size, majority_type, member_idx)
            best_score = None
            best_avg_cos = None

            for stat in cluster_stats:
                cluster_id, size, majority_type, member_idx = stat
                if size == 0:
                    continue

                member_tensor_idx = np.array(member_idx, dtype=int)

                # 计算簇内两两归一化余弦相似度的平均值
                sub_cos = cosine_matrix_normalized[np.ix_(member_tensor_idx, member_tensor_idx)]
                tril = np.tril_indices_from(sub_cos, k=-1)
                if len(tril[0]) > 0:
                    pair_vals = sub_cos[tril]
                    avg_pair_cos = float(pair_vals.mean())
                else:
                    # 只有 1 个成员的簇，没有成对相似度，默认认为内聚度为 1.0
                    avg_pair_cos = 1.0

                score = size * (1 + avg_pair_cos)

                logging.info(
                    f"[AvgAlign][ClusterSelect] cluster={cluster_id}, size={size}, "
                    f"majority={majority_type}, avg_pairwise_cos={avg_pair_cos:.4f}, score={score:.4f}"
                )

                if best_score is None or score > best_score:
                    best_score = score
                    best_avg_cos = avg_pair_cos
                    best_cluster = stat

            if best_cluster is not None:
                chosen_cluster_id, chosen_size, _, chosen_member_idx = best_cluster
                # numpy 索引转换为 Python 列表
                selected_indices = np.array(chosen_member_idx, dtype=int).tolist()
                logging.info(
                    f"[AvgAlign][Cluster] 选用 cluster={chosen_cluster_id} 进行聚合，"
                    f"成员数={chosen_size}, avg_pairwise_cos={best_avg_cos:.4f}, score={best_score:.4f}"
                )
            else:
                logging.info("[AvgAlign][Cluster] 未获得有效聚类结果（所有簇为空），退化为使用全部客户端")

        selected_client_ids = [client_ids[idx] for idx in selected_indices]
        selected_updates = {cid: agent_updates_dict[cid] for cid in selected_client_ids}

        sm_updates, total_data = 0, 0
        for cid, update in selected_updates.items():
            data_sz = self.agent_data_sizes[cid]
            sm_updates += data_sz * update
            total_data += data_sz
        if total_data == 0:
            aggregated_update = torch.mean(torch.stack(list(selected_updates.values()), dim=0), dim=0)
        else:
            aggregated_update = sm_updates / total_data

        return aggregated_update

    def agg_avg_alignment2(self, agent_updates_dict):
        """
        与 agg_avg_alignment 相同的诊断与筛选过程，但不做最终聚合，
        而是返回筛选出的“良性”客户端更新字典，供其他流程复用。
        """
        if len(agent_updates_dict) == 0:
            return {}

        client_ids = list(agent_updates_dict.keys())
        local_updates = [agent_updates_dict[cid] for cid in client_ids]
        stacked = torch.stack(local_updates, dim=0)
        n_clients, dim = stacked.shape

        topk_ratio = float(getattr(self.args, "avg_align_topk_ratio", 0.3))
        topk_dim = max(1, int(dim * topk_ratio))
        abs_updates = torch.abs(stacked)
        topk_indices = torch.topk(abs_updates, k=topk_dim, dim=1).indices
        topk_mask = torch.zeros_like(stacked, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)

        sign_updates = torch.sign(stacked)
        num_corrupt = int(getattr(self.args, "num_corrupt", 0))
        eps = 1e-8

        # 计算对齐矩阵与特征（与 agg_avg_alignment 相同）
        align_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        cosine_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        feature_matrix = np.zeros((n_clients, n_clients, 2), dtype=np.float64)

        benign_pairs = []
        mixed_pairs = []
        malicious_pairs = []
        benign_cosine_pairs = []
        mixed_cosine_pairs = []
        malicious_cosine_pairs = []
        gradient_ratio_list = []

        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j:
                    align_matrix[i, j] = 1.0
                    cosine_matrix[i, j] = 0.0
                    feature_matrix[i, j] = [0.0, 1.0]
                    continue

                common_mask = topk_mask[i] & topk_mask[j]
                intersect_count = int(common_mask.sum().item())
                gradient_ratio = intersect_count / dim if dim > 0 else 0.0
                gradient_ratio_list.append(gradient_ratio)

                if intersect_count == 0:
                    align_score = 0.0
                    cosine_sim = 0.0
                else:
                    same_sign = torch.sum(sign_updates[i][common_mask] == sign_updates[j][common_mask]).item()
                    align_score = float(same_sign / intersect_count)

                    vec_i = stacked[i][common_mask]
                    vec_j = stacked[j][common_mask]
                    norm_i = torch.norm(vec_i).item()
                    norm_j = torch.norm(vec_j).item()
                    if norm_i > eps and norm_j > eps:
                        cosine_sim = float(torch.dot(vec_i, vec_j).item() / (norm_i * norm_j))
                    else:
                        cosine_sim = 0.0

                align_matrix[i, j] = align_matrix[j, i] = align_score
                cosine_matrix[i, j] = cosine_matrix[j, i] = cosine_sim
                feature_matrix[i, j] = feature_matrix[j, i] = [cosine_sim, align_score]

                id_i = client_ids[i]
                id_j = client_ids[j]
                if id_i >= num_corrupt and id_j >= num_corrupt:
                    benign_pairs.append(align_score)
                    benign_cosine_pairs.append(cosine_sim)
                elif id_i < num_corrupt and id_j < num_corrupt:
                    malicious_pairs.append(align_score)
                    malicious_cosine_pairs.append(cosine_sim)
                else:
                    mixed_pairs.append(align_score)
                    mixed_cosine_pairs.append(cosine_sim)

        # 归一化矩阵
        triu_indices = np.triu_indices_from(cosine_matrix, k=1)
        cosine_vals = cosine_matrix[triu_indices]
        align_vals = align_matrix[triu_indices]

        eps = 1e-12
        if len(cosine_vals) > 0:
            cmin, cmax = np.min(cosine_vals), np.max(cosine_vals)
        else:
            cmin, cmax = 0.0, 1.0
        if abs(cmax - cmin) < eps:
            cosine_matrix_normalized = cosine_matrix.copy()
        else:
            cosine_matrix_normalized = (cosine_matrix - cmin) / (cmax - cmin)
            np.fill_diagonal(cosine_matrix_normalized, 0.0)

        if len(align_vals) > 0:
            amin, amax = np.min(align_vals), np.max(align_vals)
        else:
            amin, amax = 0.0, 1.0
        if abs(amax - amin) < eps:
            align_matrix_normalized = align_matrix.copy()
        else:
            align_matrix_normalized = (align_matrix - amin) / (amax - amin)
            np.fill_diagonal(align_matrix_normalized, 1.0)

        feature_matrix_normalized = np.zeros((n_clients, n_clients, 2), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                feature_matrix_normalized[i, j] = [cosine_matrix_normalized[i, j], align_matrix_normalized[i, j]]

        # 聚类与簇选择（沿用原逻辑，基于 Cohesion: Size * AvgPairwiseCosine）
        cluster_method = getattr(self.args, "align_cluster_method", "none").lower()
        selected_indices = list(range(n_clients))

        def _report_clusters(labels, method_name):
            unique_labels = sorted(set(labels))
            stats = []
            for cluster_id in unique_labels:
                member_idx = np.where(labels == cluster_id)[0]
                if len(member_idx) == 0:
                    continue
                member_ids = [client_ids[idx] for idx in member_idx]
                benign_cnt = sum(1 for cid in member_ids if cid >= num_corrupt)
                malicious_cnt = len(member_ids) - benign_cnt
                majority_type = "Benign" if benign_cnt >= malicious_cnt else "Malicious"
                stats.append((cluster_id, len(member_idx), majority_type, member_idx))
            return stats

        # 计算基于二维特征的距离矩阵
        feature_dist_matrix = np.zeros((n_clients, n_clients), dtype=np.float64)
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    feature_dist_matrix[i, j] = 0.0
                else:
                    feat_i = feature_matrix_normalized[i, j]
                    cosine_sim = feat_i[0]
                    align_score = feat_i[1]
                    dist = np.sqrt((1.0 - cosine_sim) ** 2 + (1.0 - align_score) ** 2)
                    feature_dist_matrix[i, j] = dist

        cluster_labels = None
        cluster_stats = []
        if cluster_method != "none":
            if n_clients < 2:
                logging.info("[AvgAlign2][Cluster] 客户端数量不足，跳过聚类（n_clients=%d）", n_clients)
            else:
                if cluster_method == "kmeans":
                    if KMeans is not None:
                        cluster_k = int(getattr(self.args, "align_cluster_k", 2))
                        cluster_k = max(1, min(cluster_k, n_clients))
                        if cluster_k == 1:
                            cluster_labels = np.zeros(n_clients, dtype=int)
                        else:
                            client_features = np.zeros((n_clients, 2), dtype=np.float64)
                            for i in range(n_clients):
                                mask = np.arange(n_clients) != i
                                if mask.any():
                                    client_features[i] = np.mean(feature_matrix_normalized[i, mask, :], axis=0)
                                else:
                                    client_features[i] = [1.0, 1.0]
                            try:
                                kmeans = KMeans(n_clusters=cluster_k, n_init="auto", random_state=42)
                            except TypeError:
                                kmeans = KMeans(n_clusters=cluster_k, n_init=10, random_state=42)
                            cluster_labels = kmeans.fit_predict(client_features)
                    else:
                        logging.warning("[AvgAlign2][KMeans] sklearn 未安装，无法执行 KMeans 聚类")
                elif cluster_method == "spectral":
                    if SpectralClustering is not None:
                        cluster_k = int(getattr(self.args, "align_spectral_clusters", 2))
                        cluster_k = max(2, min(cluster_k, n_clients))
                        similarity_matrix = 1.0 / (1.0 + feature_dist_matrix)
                        np.fill_diagonal(similarity_matrix, 1.0)
                        spectral = SpectralClustering(
                            n_clusters=cluster_k, affinity="precomputed", assign_labels="kmeans", random_state=42
                        )
                        cluster_labels = spectral.fit_predict(similarity_matrix)
                    else:
                        logging.warning("[AvgAlign2][Spectral] sklearn 未安装，无法执行谱聚类")
                elif cluster_method == "dbscan":
                    if DBSCAN is not None:
                        eps_db = float(getattr(self.args, "align_dbscan_eps", 0.3))
                        min_samples = int(getattr(self.args, "align_dbscan_min_samples", 2))
                        dbscan = DBSCAN(eps=eps_db, min_samples=min_samples, metric="precomputed")
                        cluster_labels = dbscan.fit_predict(feature_dist_matrix)
                    else:
                        logging.warning("[AvgAlign2][DBSCAN] sklearn 未安装，无法执行 DBSCAN 聚类")
                elif cluster_method == "hdbscan":
                    if hdbscan is not None:
                        clusterer = hdbscan.HDBSCAN(
                            metric="precomputed",
                            min_cluster_size=max(2, n_clients // 2 + 1),
                            min_samples=1,
                            allow_single_cluster=True,
                        )
                        cluster_labels = clusterer.fit(feature_dist_matrix).labels_
                    else:
                        logging.warning("[AvgAlign2][HDBSCAN] hdbscan 未安装，无法执行 HDBSCAN 聚类")
                elif cluster_method == "finch":
                    if finch is not None:
                        try:
                            np.fill_diagonal(feature_dist_matrix, 0.0)
                            try:
                                c, num_clust, req_c = finch.FINCH(
                                    feature_dist_matrix,
                                    initial_rank=None,
                                    req_clust=None,
                                    distance='precomputed',
                                    ensure_early_termination=True,
                                    verbose=False
                                )
                            except (TypeError, ValueError):
                                client_features = np.zeros((n_clients, 2), dtype=np.float64)
                                for i in range(n_clients):
                                    mask = np.arange(n_clients) != i
                                    if mask.any():
                                        client_features[i] = np.mean(feature_matrix_normalized[i, mask, :], axis=0)
                                    else:
                                        client_features[i] = [1.0, 1.0]
                                c, num_clust, req_c = finch.FINCH(
                                    client_features,
                                    initial_rank=None,
                                    req_clust=None,
                                    distance='euclidean',
                                    ensure_early_termination=True,
                                    verbose=False
                                )
                            if c is not None and c.shape[1] > 0:
                                cluster_labels = c[:, -1].astype(int)
                            else:
                                cluster_labels = None
                        except Exception as e:
                            logging.warning(f"[AvgAlign2][FINCH] FINCH 聚类执行失败: {e}")
                            cluster_labels = None
                    else:
                        logging.warning("[AvgAlign2][FINCH] finch 未安装，无法执行 FINCH 聚类")

        if cluster_labels is not None:
            cluster_stats = _report_clusters(cluster_labels, cluster_method.upper())

            best_cluster = None
            best_score = None
            best_avg_cos = None

            for stat in cluster_stats:
                cluster_id, size, majority_type, member_idx = stat
                if size == 0:
                    continue

                member_tensor_idx = np.array(member_idx, dtype=int)
                sub_cos = cosine_matrix_normalized[np.ix_(member_tensor_idx, member_tensor_idx)]
                tril = np.tril_indices_from(sub_cos, k=-1)
                if len(tril[0]) > 0:
                    pair_vals = sub_cos[tril]
                    avg_pair_cos = float(pair_vals.mean())
                else:
                    avg_pair_cos = 1.0

                score = size * (1.0 + avg_pair_cos)  # 与主版本一致

                if best_score is None or score > best_score:
                    best_score = score
                    best_avg_cos = avg_pair_cos
                    best_cluster = stat

            if best_cluster is not None:
                _, _, _, chosen_member_idx = best_cluster
                selected_indices = np.array(chosen_member_idx, dtype=int).tolist()
            else:
                logging.info("[AvgAlign2][Cluster] 未获得有效聚类结果，退化为使用全部客户端")

        selected_client_ids = [client_ids[idx] for idx in selected_indices]
        selected_updates = {cid: agent_updates_dict[cid] for cid in selected_client_ids}
        return selected_updates
    def agg_median(self, agent_updates_dict):
        """Coordinate-wise median aggregation."""
        if len(agent_updates_dict) == 0:
            raise ValueError("agent_updates_dict must not be empty for median aggregation")
        stacked = torch.stack(list(agent_updates_dict.values()), dim=0)
        aggregated_update = torch.median(stacked, dim=0).values
        return aggregated_update

    
    def agg_mkrum(self, agent_updates_dict):
        krum_param_m = 10
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            # local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]

        print('%d clients are selected' % len(score_index))
        return_updates = [agent_updates_dict[i] for i in score_index]


        return sum(return_updates)/len(return_updates)

    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    def agg_mul_metric(self, agent_updates_dict, global_model, flat_global_model):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        vectorize_nets = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        print(scores)

        p = 0.3
        p_num = p*len(scores)
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        print(topk_ind)
        current_dict = {}

        for idx in topk_ind:
            current_dict[chosen_clients[idx]] = agent_updates_dict[chosen_clients[idx]]

        update = self.agg_avg(current_dict)

        return update

    def agg_mpsa_guard(self, agent_updates_dict, global_model, flat_global_model):
        """
        AlignIns-inspired defense:
        1. Use AlignIns' MPSA pipeline (client ordering, mz-score) to identify candidate benign clients.
        2. Instead of discarding anomalies, flip their signs toward the clean-majority sign.
        3. Apply IQR-based magnitude normalization per neuron.
        4. Aggregate clean clients to form a temporary centroid and only reintroduce corrected suspects
           if their distance to the centroid lies within the benign distance range.
        """
        sparsity = float(getattr(self.args, "sparsity", 0.3))
        lambda_s = float(getattr(self.args, "lambda_s", 1.0))
        eps = getattr(self.args, "eps", 1e-12)
        iqr_mu = float(getattr(self.args, "iqr_mu", 1.5))
        dist_iqr_mu = float(getattr(self.args, "distance_iqr_mu", 0.75))

        malicious_id = []
        benign_id = []
        for _id in agent_updates_dict.keys():
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        chosen_clients = malicious_id + benign_id
        num_chosen_clients = len(chosen_clients)
        if num_chosen_clients == 0:
            return torch.zeros_like(flat_global_model)
        if num_chosen_clients == 1:
            return agent_updates_dict[chosen_clients[0]]

        # AlignIns ordering: stack updates following (malicious + benign) id list
        local_updates = [agent_updates_dict[cid] for cid in chosen_clients]
        inter_model_updates = torch.stack(local_updates, dim=0)

        # --- Stage 1: AlignIns' MPSA pipeline to find candidate benign indices ---
        major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
        mpsa_list = []
        topk_dim = max(1, int(inter_model_updates.shape[1] * sparsity))
        topk_indices_per_client = []
        for i in range(len(inter_model_updates)):
            _, init_indices = torch.topk(torch.abs(inter_model_updates[i]), topk_dim)
            topk_indices_per_client.append(init_indices)
            mpsa = (
                torch.sum(
                    torch.sign(inter_model_updates[i][init_indices]) == major_sign[init_indices]
                )
                / torch.numel(inter_model_updates[i][init_indices])
            ).item()
            mpsa_list.append(mpsa)
        logging.info('MPSA: %s' % [round(i, 4) for i in mpsa_list])

        mpsa_std = np.std(mpsa_list) if len(mpsa_list) > 1 else 1e-12
        mpsa_med = np.median(mpsa_list)
        mzscore_mpsa = [np.abs(m - mpsa_med) / mpsa_std for m in mpsa_list]
        logging.info('MZ-score of MPSA: %s' % [round(i, 4) for i in mzscore_mpsa])

        benign_idx_set = set(range(num_chosen_clients))
        benign_idx_set.intersection_update(
            set(np.argwhere(np.array(mzscore_mpsa) < lambda_s).flatten().astype(int))
        )
        clean_indices = sorted(list(benign_idx_set))
        suspect_indices = [idx for idx in range(num_chosen_clients) if idx not in benign_idx_set]

        def _calc_precision(selected_indices, chosen_ids, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = chosen_ids[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        pre_precision, pre_selected_cnt, pre_clean_cnt = _calc_precision(clean_indices, chosen_clients, benign_id)
        if pre_precision is not None:
            logging.info(f"[MPSAGuard][MPSA阶段] Precision: {pre_precision:.4f} | 选中数: {pre_selected_cnt} 真正干净数: {pre_clean_cnt}")

        if len(clean_indices) == 0:
            logging.info("[MPSAGuard] No clients fall below lambda_s, returning zero update.")
            return torch.zeros_like(local_updates[0])

        # --- Stage 2: Flip suspect signs to match clean-majority orientation ---
        clean_stack = inter_model_updates[clean_indices]
        clean_major_sign = torch.sign(torch.sum(torch.sign(clean_stack), dim=0))

        corrected_updates = []
        for idx, vec in enumerate(local_updates):
            corrected = vec.clone()
            if idx in suspect_indices:
                target_sign = torch.where(clean_major_sign == 0, torch.sign(corrected), clean_major_sign)
                topk_idx = topk_indices_per_client[idx]
                corrected[topk_idx] = corrected[topk_idx].abs() * target_sign[topk_idx]
            corrected_updates.append(corrected)

        corrected_stack = torch.stack(corrected_updates, dim=0)
        amplitudes = corrected_stack.abs()
        q1 = torch.quantile(amplitudes, 0.25, dim=0)
        q3 = torch.quantile(amplitudes, 0.75, dim=0)
        median_amp = torch.quantile(amplitudes, 0.5, dim=0)
        iqr = q3 - q1
        lower_bound = q1 - iqr_mu * iqr
        upper_bound = q3 + iqr_mu * iqr

        normalized_updates = []
        for vec in corrected_updates:
            amp = vec.abs()
            sign = torch.sign(vec)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            adjust_mask = (amp < lower_bound) | (amp > upper_bound)
            amp = torch.where(adjust_mask, median_amp, amp)
            normalized_updates.append(sign * amp)

        # Shared helper: data-size weighted averaging with safe fallback
        def _weighted_average(indices):
            if len(indices) == 0:
                return torch.zeros_like(local_updates[0])
            total = 0.0
            acc = torch.zeros_like(local_updates[0])
            for local_idx in indices:
                cid = chosen_clients[local_idx]
                weight = float(self.agent_data_sizes[cid])
                total += weight
                acc += normalized_updates[local_idx] * weight
            if total == 0.0:
                return torch.mean(torch.stack([normalized_updates[i] for i in indices], dim=0), dim=0)
            return acc / total

        temp_update = _weighted_average(clean_indices)

        # --- Stage 3: distance-based gating using benign centroid statistics ---
        def _distance_stats(indices, centroid):
            if len(indices) == 0:
                return -np.inf, np.inf, []
            centroid_norm = torch.norm(centroid) + eps
            dists = []
            for idx in indices:
                vec = normalized_updates[idx]
                vec_norm = torch.norm(vec) + eps
                cosine_distance = 1.0 - torch.dot(vec, centroid) / (vec_norm * centroid_norm)
                dists.append(cosine_distance.item())
            if len(dists) < 2:
                return -np.inf, np.inf, dists
            dist_tensor = torch.tensor(dists, dtype=torch.float64)
            q1 = torch.quantile(dist_tensor, 0.25).item()
            q3 = torch.quantile(dist_tensor, 0.75).item()
            iqr = q3 - q1
            lower = q1 - dist_iqr_mu * iqr
            upper = q3 + dist_iqr_mu * iqr
            lower = 0
            upper = min(1.0, upper)
            return lower, upper, dists

        lower_dist, upper_dist, benign_dists = _distance_stats(clean_indices, temp_update)
        accepted_suspects = []
        for idx in suspect_indices:
            vec = normalized_updates[idx]
            vec_norm = torch.norm(vec) + eps
            centroid_norm = torch.norm(temp_update) + eps
            dist = 1.0 - torch.dot(vec, temp_update) / (vec_norm * centroid_norm)
            if lower_dist <= dist <= upper_dist:
                accepted_suspects.append(idx)

        selected_indices = sorted(clean_indices + accepted_suspects)
        if len(selected_indices) == 0:
            logging.warning("[MPSAGuard] No clients selected after filtering, returning zero update.")
            return torch.zeros_like(local_updates[0])

        logging.info(f"[MPSAGuard] Clean indices: {clean_indices}, Accepted suspects: {accepted_suspects}")
        logging.info(f"[MPSAGuard] Benign distance stats: n={len(benign_dists)}, range=({lower_dist:.4f}, {upper_dist:.4f})")

        def _calc_precision(selected_indices, chosen_ids, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = chosen_ids[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        final_update = _weighted_average(selected_indices)

        correct = 0
        for idx in selected_indices:
            if chosen_clients[idx] in benign_id:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        FPR = 0.0
        if len(malicious_id) > 0:
            wrong = 0
            for idx in selected_indices:
                if chosen_clients[idx] in malicious_id:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str([chosen_clients[idx] for idx in selected_indices]))
        logging.info('FPR:       %.4f' % FPR)
        logging.info('TPR:       %.4f' % TPR)

        return final_update
   
    def agg_foolsgold(self, agent_updates_dict):
        def foolsgold(updates):
            """
            :param updates:
            :return: compute similatiry and return weightings
            """
            n_clients = updates.shape[0]
            cs = smp.cosine_similarity(updates) - np.eye(n_clients)

            maxcs = np.max(cs, axis=1)
            # pardoning
            for i in range(n_clients):
                for j in range(n_clients):
                    if i == j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
            wv = 1 - (np.max(cs, axis=1))

            wv[wv > 1] = 1
            wv[wv < 0] = 0

            alpha = np.max(cs, axis=1)

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log(wv / (1 - wv)) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # wv is the weight
            return wv, alpha

        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        names = malicious_id + benign_id
        num_chosen_clients = len(malicious_id + benign_id)

        client_updates = [update.detach().cpu().numpy() for update in agent_updates_dict.values()]
        update_len = np.array(client_updates[0].shape).prod()
        # print("client_updates size", client_models[0].parameters())
        # update_len = len(client_updates)
        # if self.memory is None:
        #     self.memory = np.zeros((self.num_clients, update_len))
        if len(names) < len(client_updates):
            names = np.append([-1], names)  # put in adv

        num_clients = num_chosen_clients
        memory = np.zeros((num_clients, update_len))
        updates = np.zeros((num_clients, update_len))

        for i in range(len(client_updates)):
            # updates[i] = np.reshape(client_updates[i][-2].cpu().data.numpy(), (update_len))
            updates[i] = np.reshape(client_updates[i], (update_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]] += updates[i]
            else:
                self.memory_dict[names[i]] = copy.deepcopy(updates[i])
            memory[i] = self.memory_dict[names[i]]
        # self.memory += updates
        use_memory = False

        if use_memory:
            wv, alpha = foolsgold(None)  # Use FG
        else:
            wv, alpha = foolsgold(updates)  # Use FG
        # logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        print(len(client_updates), len(wv))


        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        print(aggregated_model.shape)

        return aggregated_model

    def agg_scope_multimetric(self, agent_updates_dict, global_model, flat_global_model):
        """
        Scope-style multi-metric defense adapted to aggregation setting.
        - Build per-client model vectors as (flat_global_model + update)
        - Compute relative change pre-metric
        - Multi-metric pairwise distances (cosine, L1, L2), z-score standardize and combine
        - MPSA-based prefilter on updates
        - Wave expansion from a seed in allowed set to select clients
        - Weighted average of selected updates by agent_data_sizes
        """
        eps = getattr(self.args, "eps", 1e-12)
        sparsity = getattr(self.args, "sparsity", 0.3)
        lambda_s = getattr(self.args, "lambda_s", 1.0)
        percent_select = float(getattr(self.args, "percent_select", 20.0))
        combine_method = getattr(self.args, "combine_method", "max")
        use_candidate_seed = getattr(self.args, "use_candidate_seed", False)
        use_mpsa_prefilter = getattr(self.args, "use_mpsa_prefilter", False)
        self.candidate_seed_ratio = float(getattr(self.args, "candidate_seed_ratio", 0.25))
        # FedID dynamic weighting regularization coefficient
        self.fedid_reg = float(getattr(self.args, "fedid_reg", 1e-3))

        # Maintain client id order for weighting and build label lists
        client_ids = []
        local_updates = []
        benign_id = []
        malicious_id = []
        for _id, update in agent_updates_dict.items():
            client_ids.append(_id)
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        # Build client model vectors (numpy)
        vectorize_global = flat_global_model.detach().cpu().numpy()
        client_model_vecs = []
        for upd in local_updates:
            client_model_vecs.append((flat_global_model + upd).detach().cpu().numpy())

        # Step: relative change pre-metric
        pre_metric_dis = []
        for g_i in client_model_vecs:
            pre_metric = np.power(np.abs(g_i - vectorize_global) / (np.abs(g_i) + np.abs(vectorize_global) + eps), 2.0) * np.sign(g_i - vectorize_global)
            pre_metric_dis.append(pre_metric)

        n = len(pre_metric_dis)
        if n == 0:
            return torch.zeros_like(flat_global_model)
        if n == 1:
            return local_updates[0]

        # Multi-metric pairwise distances
        cos_mat = np.zeros((n, n), dtype=np.float64)
        l1_mat = np.zeros((n, n), dtype=np.float64)
        l2_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            gi = pre_metric_dis[i]
            ni = np.linalg.norm(gi) + eps
            for j in range(i + 1, n):
                gj = pre_metric_dis[j]
                nj = np.linalg.norm(gj) + eps
                cosine_distance = float(1.0 - (np.dot(gi, gj) / (ni * nj)))
                manhattan_distance = float(np.linalg.norm(gi - gj, ord=1))
                euclidean_distance = float(np.linalg.norm(gi - gj))
                cos_mat[i, j] = cos_mat[j, i] = cosine_distance
                l1_mat[i, j] = l1_mat[j, i] = manhattan_distance
                l2_mat[i, j] = l2_mat[j, i] = euclidean_distance

        # z-score each metric using upper triangle stats
        # std_mats = []
        # for M in (cos_mat, l1_mat, l2_mat):
        #     triu = M[np.triu_indices_from(M, k=1)]
        #     mean = np.mean(triu) if triu.size > 0 else 0.0
        #     std = np.std(triu) if triu.size > 0 else 1.0
        #     std = std if std > eps else 1.0
        #     Z = (M - mean) / std
        #     std_mats.append(Z)
        # cosZ, l1Z, l2Z = std_mats

        std_mats = []
        for M in (cos_mat, l1_mat, l2_mat):
            triu = M[np.triu_indices_from(M, k=1)]  # 取上三角（不含对角线）的距离值（避免对角线0值影响极值计算）
            if triu.size == 0:
                # 无有效距离值（仅1个客户端，实际前面已处理n=1的情况），直接返回全0矩阵
                Z = np.zeros_like(M)
            else:
                min_val = np.min(triu)
                max_val = np.max(triu)
                # 避免分母为0（所有距离相同），添加eps保护
                denominator = max_val - min_val + eps
                # Min-Max 公式：Z = (M - min_val) / denominator，缩放到 [0, 1]
                Z = (M - min_val) / denominator
            std_mats.append(Z)
        cosZ, l1Z, l2Z = std_mats

        # Combine
        combined_D = np.zeros((n, n), dtype=np.float64)
        if combine_method == "euclidean":
            combined_D = np.sqrt(np.maximum(cosZ, 0.0) ** 2 + np.maximum(l1Z, 0.0) ** 2 + np.maximum(l2Z, 0.0) ** 2)
        elif combine_method == "max":
            combined_D = np.maximum.reduce([cosZ, l1Z, l2Z])
        elif combine_method == "scope":
            # Original scope method: use cosine distance with special handling
            # 注意：循环包含 i == j，显式计算对角线（自己到自己的距离为0）
            # for i in range(n):
            #     gi = pre_metric_dis[i]
            #     ni = np.linalg.norm(gi)
            #     ni = ni if ni > eps else eps
            #     for j in range(i, n):  # 包含 i == j，与原始实现一致
            #         gj = pre_metric_dis[j]
            #         nj = np.linalg.norm(gj)
            #         nj = nj if nj > eps else eps
            #         cosine_distance = float(1.0 - (np.dot(gi, gj) / (ni * nj)))
            #         if abs(cosine_distance) < 0.000001:
            #             cosine_distance = 100.0
            #         combined_D[i, j] = combined_D[j, i] = cosine_distance
            combined_D = cosZ
            logging.info("[Scope][max] cosZ: %s" % np.round(cosZ, 3).tolist())
        elif combine_method == "mahalanobis":
            idx_i, idx_j = np.triu_indices(n, k=1)
            feats = np.stack([cosZ[idx_i, idx_j], l1Z[idx_i, idx_j], l2Z[idx_i, idx_j]], axis=1)
            cov = np.cov(feats.T)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                return
            for i in range(n):
                for j in range(i + 1, n):
                    v = np.array([cosZ[i, j], l1Z[i, j], l2Z[i, j]])
                    d = float(v.T @ inv @ v)
                    combined_D[i, j] = combined_D[j, i] = d
        elif combine_method == "mahalanobis_raw":
            idx_i, idx_j = np.triu_indices(n, k=1)
            feats = np.stack([cos_mat[idx_i, idx_j], l1_mat[idx_i, idx_j], l2_mat[idx_i, idx_j]], axis=1)
            cov = np.cov(feats.T)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                return
            for i in range(n):
                for j in range(i + 1, n):
                    v = np.array([cos_mat[i, j], l1_mat[i, j], l2_mat[i, j]])
                    d = float(v.T @ inv @ v)
                    combined_D[i, j] = combined_D[j, i] = d
        elif combine_method == "fedid_dynamic":  # 新增：FedID动态加权策略
            # 步骤1：构建每个客户端的三维距离特征向量（对所有其他客户端的距离取平均）
            client_features = np.zeros((n, 3), dtype=np.float64)  # [n, 3]：n个客户端，3种度量
            for i in range(n):
                mask = np.arange(n) != i  # 排除自身距离（对角线）
                # client_features[i, 0] = np.mean(cosZ[i, mask])  # 余弦距离均值
                # client_features[i, 1] = np.mean(l1Z[i, mask])  # L1距离均值
                # client_features[i, 2] = np.mean(l2Z[i, mask])  # L2距离均值
                client_features[i, 0] = np.sum(cosZ[i, mask])  # 余弦距离总和
                client_features[i, 1] = np.sum(l1Z[i, mask])  # L1距离总和
                client_features[i, 2] = np.sum(l2Z[i, mask])  # L2距离总和

            # 步骤2：计算浓度矩阵（协方差矩阵的逆，加入正则化）
            cov_matrix = np.cov(client_features.T)  # 3x3协方差矩阵（反映度量间相关性）
            reg_cov = cov_matrix + self.fedid_reg * np.eye(3)  # 正则化，避免奇异
            try:
                concentration_matrix = np.linalg.inv(reg_cov)  # 浓度矩阵（动态权重核心）
                logging.info(f"[FedID动态加权] 浓度矩阵计算完成，正则化系数={self.fedid_reg}")
            except np.linalg.LinAlgError:
                # 极端情况：协方差矩阵无法求逆，退化为单位矩阵（等权重）
                concentration_matrix = np.eye(3)
                logging.warning("[FedID动态加权] 协方差矩阵奇异，退化为单位矩阵加权")

            # 步骤3：用浓度矩阵对距离向量进行白化处理（动态加权融合）
            for i in range(n):
                for j in range(i + 1, n):
                    # 客户端i和j的三维距离向量（标准化后）
                    dist_vec = np.array([cosZ[i, j], l1Z[i, j], l2Z[i, j]])
                    # 白化处理：通过浓度矩阵自适应加权（二次型计算）
                    dynamic_dist = dist_vec.T @ concentration_matrix @ dist_vec
                    # 确保距离非负（理论上应为非负，实际加安全保障）
                    combined_D[i, j] = combined_D[j, i] = max(dynamic_dist, 0.0)
        else:
            combined_D = np.sqrt(np.maximum(cosZ, 0.0) ** 2 + np.maximum(l1Z, 0.0) ** 2 + np.maximum(l2Z, 0.0) ** 2)
        np.fill_diagonal(combined_D, 0.0)

        # MPSA prefilter on updates (torch)
        if use_mpsa_prefilter:
            inter_model_updates = torch.stack(local_updates, dim=0)  # updates w.r.t global
            major_sign = torch.sign(torch.sum(torch.sign(inter_model_updates), dim=0))
            topk_dim = max(1, int(inter_model_updates.shape[1] * float(sparsity)))
            mpsa_list = []
            for i in range(inter_model_updates.shape[0]):
                vec_i = inter_model_updates[i]
                abs_i = torch.abs(vec_i)
                _, init_indices = torch.topk(abs_i, topk_dim)
                agree = torch.sum(torch.sign(vec_i[init_indices]) == major_sign[init_indices]).item()
                mpsa = agree / float(init_indices.numel())
                mpsa_list.append(mpsa)
            # MZ-score on MPSA
            mpsa_arr = np.array(mpsa_list, dtype=np.float64)
            mpsa_std = np.std(mpsa_arr)
            mpsa_med = np.median(mpsa_arr)
            mpsa_std = mpsa_std if mpsa_std > eps else 1.0
            mzscore_mpsa = np.abs(mpsa_arr - mpsa_med) / mpsa_std
            allowed_indices = [int(i) for i in np.where(mzscore_mpsa < float(lambda_s))[0]]
            if len(allowed_indices) == 0:
                logging.info("[ScopeMM] MPSA未筛出良性客户端，退化为使用全部客户端")
                allowed_indices = list(range(n))
            allowed_indices = sorted(allowed_indices)
        else:
            # No MPSA prefilter: use all clients as allowed set
            allowed_indices = list(range(n))
            logging.info("[ScopeMM] 已关闭MPSA预筛选，使用全部客户端作为允许集合")

        # MPSA precision logging (chosen = allowed_indices)
        def _calc_precision(selected_indices, id_list, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = id_list[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count
        mpsa_precision, mpsa_selected, mpsa_true_clean = _calc_precision(allowed_indices, client_ids, benign_id)
        if mpsa_precision is not None:
            logging.info(f"[MPSA] 识别的干净客户端准确率(Precision): {mpsa_precision:.4f}  |  选中数: {mpsa_selected}  真正干净数: {mpsa_true_clean}")
        else:
            logging.info(f"[MPSA] 无选中客户端，无法计算干净客户端准确率")

        # Wave expansion on allowed set
        if use_candidate_seed:
            allowed_arr = np.array(allowed_indices, dtype=int)
            if len(allowed_arr) == 0:
                logging.info("[ScopeMultiMetricDefense] 候选种子阶段允许集合为空，退化为使用全部客户端")
                allowed_indices = list(range(n))
                allowed_arr = np.array(allowed_indices, dtype=int)
            sum_dis_allowed = np.zeros(len(allowed_indices))
            for idx, local_i in enumerate(allowed_indices):
                mask = allowed_arr != local_i
                sum_dis_allowed[idx] = np.sum(combined_D[local_i, allowed_arr[mask]])
            num_candidates = max(1, int(len(allowed_indices) * self.candidate_seed_ratio))
            sorted_candidate_indices = np.argsort(sum_dis_allowed)[:num_candidates]
            candidate_seeds = [allowed_indices[i] for i in sorted_candidate_indices]
            logging.info(
                f"[种子选择] 候选种子比例：{self.candidate_seed_ratio}，候选种子：{candidate_seeds}，共{len(candidate_seeds)}个")
            if len(candidate_seeds) == 1:
                seed_local = candidate_seeds[0]
            else:
                candidate_dist_sum = []
                for seed in candidate_seeds:
                    other_candidates = [s for s in candidate_seeds if s != seed]
                    if len(other_candidates) == 0:
                        dist_sum = 0.0
                    else:
                        dist_sum = np.sum(combined_D[seed, other_candidates])
                    candidate_dist_sum.append(dist_sum)
                best_candidate_idx = int(np.argmin(candidate_dist_sum))
                seed_local = candidate_seeds[best_candidate_idx]
                logging.info(f"[种子校验] 候选种子距离和：{candidate_dist_sum}，最终选择种子：{seed_local}")
        else:
            sum_dis_full = np.sum(combined_D, axis=1)
            seed_local = int(allowed_indices[int(np.argmin(sum_dis_full[allowed_indices]))])
        logging.info(f"[ScopeMM] Seed local index: {seed_local}, client ID: {client_ids[seed_local]}")
        cluster = set([seed_local])
        visited = set([seed_local])
        front = set([seed_local])
        cur_percent = float(percent_select)
        allowed_arr = np.array(allowed_indices, dtype=int)
        round_idx = 1
        while cur_percent > 0 and len(front) > 0:
            k = max(1, int(round(len(allowed_indices) * (cur_percent / 100.0))))
            next_front = set()
            for u in front:
                dists = combined_D[u, allowed_arr]
                neigh_pos = np.argsort(dists)[:k]
                neighbors = allowed_arr[neigh_pos]
                for v in neighbors:
                    if int(v) not in visited:
                        next_front.add(int(v))
            if len(next_front) == 0:
                break
            # Branch-1 per-round logging with malicious counts (local IDs)
            added_local_sorted = sorted(list(next_front))
            added_client_ids = [client_ids[i] for i in added_local_sorted]
            malicious_added = [cid for cid in added_client_ids if cid < self.args.num_corrupt]
            logging.info(f"[ScopeMM][Branch1][Round {round_idx}] 新加入客户端ID: {added_client_ids}")
            logging.info(f"[ScopeMM][Branch1][Round {round_idx}] 新加入恶意客户端数: {len(malicious_added)} | 恶意客户端局部ID: {malicious_added}")
            logging.info(f"[ScopeMM][Branch1][Round {round_idx}] 本轮加入客户端个数: {len(next_front)}")
            cluster |= next_front
            visited |= next_front
            front = next_front
            cur_percent /= 2.0
            round_idx += 1
        selected = sorted(list(cluster))

        # Weighted average of selected updates
        if len(selected) == 0:
            return torch.zeros_like(local_updates[0])
        selected_ids = [client_ids[i] for i in selected]
        total = 0.0
        for cid in selected_ids:
            total += float(self.agent_data_sizes[cid])
        if total <= 0:
            weights = [1.0 / len(selected)] * len(selected)
        else:
            weights = [float(self.agent_data_sizes[cid]) / total for cid in selected_ids]
        stacked = torch.stack([local_updates[i] for i in selected], dim=0)
        w_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype).view(-1, 1)
        aggregated = torch.sum(stacked * w_tensor, dim=0)

        # Metrics logging similar to reference
        benign_idx = selected
        correct = 0
        for idx in benign_idx:
            if client_ids[idx] >= self.args.num_corrupt:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0
        if len(malicious_id) == 0:
            FPR = 0.0
        else:
            wrong = 0
            for idx in benign_idx:
                if client_ids[idx] < self.args.num_corrupt:
                    wrong += 1
            FPR = wrong / len(malicious_id)
        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str(benign_idx))
        logging.info('FPR:       %.4f' % FPR)
        logging.info('TPR:       %.4f' % TPR)
        self.tpr_history.append(TPR)
        self.fpr_history.append(FPR)
        return aggregated

    def agg_scope(self, agent_updates_dict, global_model, flat_global_model):
        """
        Scope aggregation method with support for multiple combine methods.
        """
        # Get parameters
        eps = getattr(self.args, "eps", 1e-12)
        sparsity = getattr(self.args, "sparsity", 0.3)
        lambda_s = getattr(self.args, "lambda_s", 1.0)
        combine_method = getattr(self.args, "combine_method", "scope")  # Default to "scope" for backward compatibility
        self.fedid_reg = float(getattr(self.args, "fedid_reg", 1e-3))
        use_mpsa_prefilter = getattr(self.args, "use_mpsa_prefilter", False)
        use_norm_prefilter = getattr(self.args, "use_norm_prefilter", False)
        norm_lower = float(getattr(self.args, "norm_prefilter_lower", 0.4))
        norm_upper = float(getattr(self.args, "norm_prefilter_upper", 3.0))

        client_ids = []
        local_updates = []
        benign_id = []
        malicious_id = []
        for _id, update in agent_updates_dict.items():
            client_ids.append(_id)
            local_updates.append(update)
            if _id < self.args.num_corrupt:
                malicious_id.append(_id)
            else:
                benign_id.append(_id)

        vectorize_global = flat_global_model.detach().cpu().numpy()
        vectorize_nets = []
        for upd in local_updates:
            vectorize_nets.append((flat_global_model + upd).detach().cpu().numpy())

        n = len(vectorize_nets)
        if n == 0:
            return torch.zeros_like(flat_global_model)
        if n == 1:
            return local_updates[0]

        inter_model_updates = torch.stack(local_updates, dim=0)

        # Compute relative change pre-metric
        pre_metric_dis = []
        for g_i in vectorize_nets:
            pre_metric = np.power(
                np.abs(g_i - vectorize_global) / (np.abs(g_i) + np.abs(vectorize_global) + eps),
                2.0,
            ) * np.sign(g_i - vectorize_global)
            pre_metric_dis.append(pre_metric)

        # Multi-metric pairwise distances
        cos_mat = np.zeros((n, n), dtype=np.float64)
        l1_mat = np.zeros((n, n), dtype=np.float64)
        l2_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            gi = pre_metric_dis[i]
            ni = np.linalg.norm(gi) + eps
            for j in range(i + 1, n):
                gj = pre_metric_dis[j]
                nj = np.linalg.norm(gj) + eps
                cosine_distance = float(1.0 - (np.dot(gi, gj) / (ni * nj)))
                manhattan_distance = float(np.linalg.norm(gi - gj, ord=1))
                euclidean_distance = float(np.linalg.norm(gi - gj))
                cos_mat[i, j] = cos_mat[j, i] = cosine_distance
                l1_mat[i, j] = l1_mat[j, i] = manhattan_distance
                l2_mat[i, j] = l2_mat[j, i] = euclidean_distance

        # z-score each metric using upper triangle stats   zscore的方式有负数

        # for M in (cos_mat, l1_mat, l2_mat):
        #     triu = M[np.triu_indices_from(M, k=1)]
        #     mean = np.mean(triu) if triu.size > 0 else 0.0
        #     std = np.std(triu) if triu.size > 0 else 1.0
        #     std = std if std > eps else 1.0
        #     Z = (M - mean) / std
        #     std_mats.append(Z)

        std_mats = []
        for M in (cos_mat, l1_mat, l2_mat):
            triu = M[np.triu_indices_from(M, k=1)]  # 取上三角（不含对角线）的距离值（避免对角线0值影响极值计算）
            if triu.size == 0:
                # 无有效距离值（仅1个客户端，实际前面已处理n=1的情况），直接返回全0矩阵
                Z = np.zeros_like(M)
            else:
                min_val = np.min(triu)
                max_val = np.max(triu)
                # 避免分母为0（所有距离相同），添加eps保护
                denominator = max_val - min_val + eps
                # Min-Max 公式：Z = (M - min_val) / denominator，缩放到 [0, 1]
                Z = (M - min_val) / denominator
            std_mats.append(Z)

        cosZ, l1Z, l2Z = std_mats

        # Combine metrics based on combine_method
        combined_D = np.zeros((n, n), dtype=np.float64)
        if combine_method == "euclidean":
            combined_D = np.sqrt(np.maximum(cosZ, 0.0) ** 2 + np.maximum(l1Z, 0.0) ** 2 + np.maximum(l2Z, 0.0) ** 2)
        elif combine_method == "max":
            combined_D = np.maximum.reduce([cosZ, l1Z, l2Z])
            logging.info("[Scope][max] raw cos_mat: %s" % np.round(cos_mat, 3).tolist())
            logging.info("[Scope][max] cosZ: %s" % np.round(cosZ, 3).tolist())
            logging.info("[Scope][max] raw l1_mat: %s" % np.round(l1_mat, 3).tolist())
            logging.info("[Scope][max] l1Z: %s" % np.round(l1Z, 3).tolist())
            logging.info("[Scope][max] raw l2_mat: %s" % np.round(l2_mat, 3).tolist())
            logging.info("[Scope][max] l2Z: %s" % np.round(l2Z, 3).tolist())
            logging.info("[Scope][max] combined_D: %s" % np.round(combined_D, 3).tolist())
        elif combine_method == "scope":
            # Original scope method: use cosine distance with special handling
            # 注意：循环包含 i == j，显式计算对角线（自己到自己的距离为0）
            # for i in range(n):
            #     gi = pre_metric_dis[i]
            #     ni = np.linalg.norm(gi)
            #     ni = ni if ni > eps else eps
            #     for j in range(i, n):  # 包含 i == j，与原始实现一致
            #         gj = pre_metric_dis[j]
            #         nj = np.linalg.norm(gj)
            #         nj = nj if nj > eps else eps
            #         cosine_distance = float(1.0 - (np.dot(gi, gj) / (ni * nj)))
            #         if abs(cosine_distance) < 0.000001:
            #             cosine_distance = 100.0
            #         combined_D[i, j] = combined_D[j, i] = cosine_distance
            combined_D = cosZ
            logging.info("[Scope][max] cosZ: %s" % np.round(cosZ, 3).tolist())
        elif combine_method == "mahalanobis":
            idx_i, idx_j = np.triu_indices(n, k=1)
            feats = np.stack([cosZ[idx_i, idx_j], l1Z[idx_i, idx_j], l2Z[idx_i, idx_j]], axis=1)
            cov = np.cov(feats.T)
            try:
                inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                inv = np.eye(3)
            for i in range(n):
                for j in range(i + 1, n):
                    v = np.array([cosZ[i, j], l1Z[i, j], l2Z[i, j]])
                    d = float(v.T @ inv @ v)
                    combined_D[i, j] = combined_D[j, i] = d
        elif combine_method == "fedid_dynamic":
            # FedID dynamic weighting strategy
            client_features = np.zeros((n, 3), dtype=np.float64)
            for i in range(n):
                mask = np.arange(n) != i
                client_features[i, 0] = np.sum(cosZ[i, mask])
                client_features[i, 1] = np.sum(l1Z[i, mask])
                client_features[i, 2] = np.sum(l2Z[i, mask])

            cov_matrix = np.cov(client_features.T)
            reg_cov = cov_matrix + self.fedid_reg * np.eye(3)
            try:
                concentration_matrix = np.linalg.inv(reg_cov)
                logging.info(f"[Scope][FedID动态加权] 浓度矩阵计算完成，正则化系数={self.fedid_reg}")
            except np.linalg.LinAlgError:
                concentration_matrix = np.eye(3)
                logging.warning("[Scope][FedID动态加权] 协方差矩阵奇异，退化为单位矩阵加权")

            for i in range(n):
                for j in range(i + 1, n):
                    dist_vec = np.array([cosZ[i, j], l1Z[i, j], l2Z[i, j]])
                    dynamic_dist = dist_vec.T @ concentration_matrix @ dist_vec
                    combined_D[i, j] = combined_D[j, i] = max(dynamic_dist, 0.0)
        else:
            # Default: euclidean combination
            combined_D = np.sqrt(np.maximum(cosZ, 0.0) ** 2 + np.maximum(l1Z, 0.0) ** 2 + np.maximum(l2Z, 0.0) ** 2)

        np.fill_diagonal(combined_D, np.inf)

        allowed_set = set(range(n))
        if use_norm_prefilter:
            grad_l2 = torch.norm(inter_model_updates, dim=1).cpu().numpy()
            grad_l1 = torch.norm(inter_model_updates, p=1, dim=1).cpu().numpy()

            def _apply_range_filter(values, lower_scale, upper_scale, tag):
                nonlocal allowed_set, client_ids
                if values.size == 0:
                    return
                median_val = np.median(values)
                base = max(median_val, eps)
                lower = lower_scale * base
                upper = upper_scale * base
                
                # 创建 (范数值, 客户端ID) 配对并排序
                norm_client_pairs = [(values[i], client_ids[i]) for i in range(len(values))]
                norm_client_pairs_sorted = sorted(norm_client_pairs, key=lambda x: x[0])
                sorted_norms = [f"{val:.4e}" for val, _ in norm_client_pairs_sorted]
                sorted_client_ids = [cid for _, cid in norm_client_pairs_sorted]
                logging.info(
                    f"[Scope][NormFilter][{tag}] 范数值排序(从小到大): {sorted_norms}"
                )
                logging.info(
                    f"[Scope][NormFilter][{tag}] 对应客户端ID: {sorted_client_ids}"
                )
                
                filtered_idx = set(np.argwhere((values > lower) & (values < upper)).flatten().astype(int))
                removed_idx = sorted(list(allowed_set - filtered_idx))
                allowed_set = allowed_set.intersection(filtered_idx)
                logging.info(
                    f"[Scope][NormFilter][{tag}] median={median_val:.4e}, range=({lower:.4e}, {upper:.4e}), keep={len(allowed_set)} / {n}, removed={removed_idx}"
                )

            _apply_range_filter(grad_l2, norm_lower, norm_upper, "L2")
            _apply_range_filter(grad_l1, norm_lower, norm_upper, "L1")

            # 余弦相似性过滤
            cosine_similarities = []
            global_norm = np.linalg.norm(vectorize_global) + eps
            for i in range(n):
                client_model = vectorize_nets[i]
                client_norm = np.linalg.norm(client_model) + eps
                cosine_sim = np.dot(vectorize_global, client_model) / (global_norm * client_norm)
                cosine_similarities.append(cosine_sim)
            cosine_similarities = np.array(cosine_similarities)
            _apply_range_filter(cosine_similarities, norm_lower, norm_upper, "Cosine")

            if len(allowed_set) == 0:
                logging.info("[Scope][NormFilter] 无满足范数阈值的客户端，退化为使用全部客户端")
                allowed_set = set(range(n))
        else:
            logging.info("[Scope] 已关闭范数预筛选，默认使用全部客户端")

        if use_mpsa_prefilter:
            allowed_sorted = sorted(list(allowed_set))
            subset_updates = inter_model_updates[allowed_sorted]
            major_sign = torch.sign(torch.sum(torch.sign(subset_updates), dim=0))
            topk_dim = max(1, int(subset_updates.shape[1] * float(sparsity)))
            mpsa_list = []
            for i in range(subset_updates.shape[0]):
                vec_i = subset_updates[i]
                abs_i = torch.abs(vec_i)
                _, init_indices = torch.topk(abs_i, topk_dim)
                agree = torch.sum(torch.sign(vec_i[init_indices]) == major_sign[init_indices]).item()
                mpsa = agree / float(init_indices.numel())
                mpsa_list.append(mpsa)
            mpsa_arr = np.array(mpsa_list, dtype=np.float64)
            mpsa_std = np.std(mpsa_arr)
            mpsa_med = np.median(mpsa_arr)
            mpsa_std = mpsa_std if mpsa_std > eps else 1.0
            mzscore_mpsa = np.abs(mpsa_arr - mpsa_med) / mpsa_std
            allowed_indices_local = np.where(mzscore_mpsa < float(lambda_s))[0]
            mpsa_allowed = set(allowed_sorted[int(i)] for i in allowed_indices_local)
            if len(mpsa_allowed) == 0:
                logging.info("[Scope] MPSA 未筛出良性客户端，退化为使用全部客户端")
                allowed_set = set(range(n))
            else:
                allowed_set = allowed_set.intersection(mpsa_allowed)
                if len(allowed_set) == 0:
                    logging.info("[Scope] 范数+MPSA 交集为空，退化为使用 MPSA 允许集合")
                    allowed_set = mpsa_allowed
        else:
            logging.info("[Scope] 已关闭 MPSA 预筛选")

        allowed_indices = sorted(list(allowed_set))
        if len(allowed_indices) == 0:
            return torch.zeros_like(local_updates[0])

        def _calc_precision(selected_indices, id_list, actual_benign_ids):
            selected_count = len(selected_indices)
            if selected_count == 0:
                return None, selected_count, 0
            true_clean_count = 0
            for idx in selected_indices:
                actual_id = id_list[idx]
                if actual_id in actual_benign_ids:
                    true_clean_count += 1
            precision = true_clean_count / selected_count
            return precision, selected_count, true_clean_count

        filter_precision, filter_selected, filter_true_clean = _calc_precision(allowed_indices, client_ids, benign_id)
        if filter_precision is not None:
            logging.info(
                f"[Scope][Filter] 识别的干净客户端准确率(Precision): {filter_precision:.4f}  |  选中数: {filter_selected}  真正干净数: {filter_true_clean}"
            )
        else:
            logging.info(f"[Scope][Filter] 无选中客户端，无法计算干净客户端准确率")

        allowed_arr = np.array(allowed_indices, dtype=int)
        allowed_sum_dis = np.zeros(len(allowed_arr))
        for idx, local_idx in enumerate(allowed_arr):
            mask = allowed_arr != local_idx
            if mask.any():
                allowed_sum_dis[idx] = np.sum(combined_D[local_idx, allowed_arr[mask]])
            else:
                allowed_sum_dis[idx] = 0.0

        # Candidate seed selection logic (similar to agg_scope_multimetric)
        use_candidate_seed = getattr(self.args, "use_candidate_seed", False)
        candidate_seed_ratio = float(getattr(self.args, "candidate_seed_ratio", 0.5))

        logging.info(f"[Scope] Using combine_method: {combine_method}")

        if use_candidate_seed:
            num_candidates = max(1, int(len(allowed_arr) * candidate_seed_ratio))
            sorted_candidate_positions = np.argsort(allowed_sum_dis)[:num_candidates]
            candidate_seeds = allowed_arr[sorted_candidate_positions].tolist()
            logging.info(
                f"[Scope][种子选择] 候选种子比例：{candidate_seed_ratio}，候选种子局部索引：{candidate_seeds}，候选种子客户端ID：{[client_ids[i] for i in candidate_seeds]}，共{len(candidate_seeds)}个")

            if len(candidate_seeds) == 1:
                seed_idx = candidate_seeds[0]
            else:
                candidate_dist_sum = []
                for seed in candidate_seeds:
                    other_candidates = [s for s in candidate_seeds if s != seed]
                    if len(other_candidates) == 0:
                        dist_sum = 0.0
                    else:
                        dist_sum = np.sum(combined_D[seed, other_candidates])
                    candidate_dist_sum.append(dist_sum)
                best_candidate_idx = int(np.argmin(candidate_dist_sum))
                seed_idx = candidate_seeds[best_candidate_idx]
                logging.info(
                    f"[Scope][种子校验] 候选种子距离和：{candidate_dist_sum}，最终选择种子局部索引：{seed_idx}，客户端ID：{client_ids[seed_idx]}")
        else:
            seed_idx = allowed_arr[int(np.argmin(allowed_sum_dis))]

        choice = seed_idx
        logging.info(f"[Scope] Seed local index: {seed_idx}, client ID: {client_ids[seed_idx]}")
        cluster = [choice]
        round_idx = 1
        for _ in range(len(allowed_arr)):
            dist_row = combined_D[choice, allowed_arr]
            tmp = allowed_arr[int(np.argmin(dist_row))]
            if tmp not in cluster:
                cluster.append(tmp)
                logging.info(f"[Scope][Round {round_idx}] Added client ID: {client_ids[tmp]} (local idx: {tmp})")
                round_idx += 1
            else:
                break
            choice = tmp

        logging.info(f"[Scope] Selected cluster indices: {cluster}")
        logging.info(f"[Scope] Selected client IDs: {[client_ids[i] for i in cluster]}")

        selected_ids = [client_ids[i] for i in cluster]
        total = 0.0
        for cid in selected_ids:
            total += float(self.agent_data_sizes[cid])

        if total <= 0:
            weights = [1.0 / len(cluster)] * len(cluster)
        else:
            weights = [float(self.agent_data_sizes[cid]) / total for cid in selected_ids]

        stacked = torch.stack([local_updates[i] for i in cluster], dim=0)
        w_tensor = torch.tensor(weights, device=stacked.device, dtype=stacked.dtype).view(-1, 1)
        aggregated = torch.sum(stacked * w_tensor, dim=0)

        correct = 0
        for idx in cluster:
            if client_ids[idx] >= self.args.num_corrupt:
                correct += 1
        TPR = correct / len(benign_id) if len(benign_id) > 0 else 0.0

        if len(malicious_id) == 0:
            FPR = 0.0
        else:
            wrong = 0
            for idx in cluster:
                if client_ids[idx] < self.args.num_corrupt:
                    wrong += 1
            FPR = wrong / len(malicious_id)

        logging.info('benign update index:   %s' % str(benign_id))
        logging.info('selected update index: %s' % str([client_ids[i] for i in cluster]))
        logging.info('FPR:       %.4f' % FPR)
        logging.info('TPR:       %.4f' % TPR)

        return aggregated
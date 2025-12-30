# 距离矩阵计算流程说明

## 命令分析
```bash
python federated.py \
--poison_frac 0.5 --num_corrupt 4 \
--non_iid --beta 0.5 \
--aggr median_guard_align3 --data cifar10 --attack soda \
--align_cluster_method spectral
```

## 整体流程

### 阶段1: MedianGuard 筛选 (`_median_guard_select`)

在进入距离矩阵计算之前，先通过 MedianGuard 进行初步筛选：

1. **计算中位数锚点**：
   ```python
   tmp = torch.median(stacked, dim=0).values  # [D] 所有客户端更新的中位数
   ```

2. **计算 MPSA-like 和 TDA-like 分数**：
   - **MPSA-like**: 计算每个客户端更新与中位数锚点 `tmp` 的符号一致性
     - 选择 top-k 重要坐标（按绝对值，k = dim * sparsity，默认 sparsity=0.3）
     - 计算这些坐标的符号与 `tmp` 主符号的一致性比例
   - **TDA-like**: 计算每个客户端更新与 `tmp` 的余弦相似度

3. **MZ-score 过滤**：
   - 对 MPSA 和 TDA 分数分别计算 MZ-score
   - 过滤掉 MZ-score 超过阈值（lambda_s 或 lambda_c）的客户端
   - 返回筛选后的客户端更新字典

### 阶段2: AvgAlign3 距离矩阵计算 (`agg_avg_alignment3`)

对 MedianGuard 筛选后的客户端，计算距离矩阵：

#### 步骤1: 计算原始相似度矩阵

**1.1 符号对齐矩阵 (align_matrix)**：
```python
# 对于每对客户端 (i, j):
# 1. 找到两个客户端的 top-k 重要坐标的交集
common_mask = topk_mask[i] & topk_mask[j]  # topk_ratio 默认 0.3

# 2. 计算交集坐标中符号相同的比例
same_sign = torch.sum(sign_updates[i][common_mask] == sign_updates[j][common_mask])
align_score = same_sign / intersect_count

# 结果：align_matrix[i, j] = align_score (范围 [0, 1])
```

**1.2 余弦相似度矩阵 (cosine_matrix)**：
```python
# 对于每对客户端 (i, j):
# 在共同的重要坐标上计算余弦相似度
vec_i = stacked[i][common_mask]
vec_j = stacked[j][common_mask]
cosine_sim = dot(vec_i, vec_j) / (norm(vec_i) * norm(vec_j))

# 结果：cosine_matrix[i, j] = cosine_sim (范围 [-1, 1])
```

#### 步骤2: 归一化矩阵

**2.1 归一化余弦相似度矩阵**：
```python
# 将余弦相似度从 [-1, 1] 归一化到 [0, 1]
cmin, cmax = np.min(cosine_vals), np.max(cosine_vals)
cosine_matrix_normalized = (cosine_matrix - cmin) / (cmax - cmin)
# 对角线设为 0.0（自己与自己的距离为0）
np.fill_diagonal(cosine_matrix_normalized, 0.0)
```

**2.2 归一化符号对齐矩阵**：
```python
# 将符号对齐从 [0, 1] 归一化到 [0, 1]（通常已经是 [0, 1]）
amin, amax = np.min(align_vals), np.max(align_vals)
align_matrix_normalized = (align_matrix - amin) / (amax - amin)
# 对角线设为 1.0（自己与自己的对齐度为1）
np.fill_diagonal(align_matrix_normalized, 1.0)
```

#### 步骤3: 构建特征距离矩阵

**3.1 组合特征矩阵**：
```python
# 对于每对客户端 (i, j)，构建特征向量
feature_matrix_normalized[i, j] = [
    cosine_matrix_normalized[i, j],  # 如果 use_cosine=True
    align_matrix_normalized[i, j]    # 如果 use_align=True
]
# 形状: [n_clients, n_clients, n_features]
# n_features = 2 (如果 use_cosine=True 且 use_align=True)
```

**3.2 计算特征距离矩阵**：
```python
# 对于每对客户端 (i, j):
feat = feature_matrix_normalized[i, j]  # [n_features]
# 将相似度转换为距离：距离 = sqrt(sum((1 - similarity)^2))
dist = sqrt(sum((1.0 - feat)^2))
feature_dist_matrix[i, j] = dist

# 结果：feature_dist_matrix[i, j] 表示客户端 i 和 j 之间的距离
# 距离越小，表示两个客户端越相似
```

#### 步骤4: 谱聚类（如果使用 spectral）

**4.1 将距离矩阵转换为相似度矩阵**：
```python
# 将距离转换为相似度：similarity = 1 / (1 + distance)
similarity_matrix = 1.0 / (1.0 + feature_dist_matrix)
np.fill_diagonal(similarity_matrix, 1.0)  # 自己与自己的相似度为1
```

**4.2 执行谱聚类**：
```python
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=cluster_k,           # 默认 2
    affinity="precomputed",         # 使用预计算的相似度矩阵
    assign_labels="kmeans",         # 使用 kmeans 分配标签
    random_state=42
)
cluster_labels = spectral.fit_predict(similarity_matrix)
```

## 关键参数

- **sparsity**: 0.3 (默认) - 用于选择 top-k 重要坐标的比例
- **topk_ratio**: 0.3 (默认) - 用于符号对齐的 top-k 比例
- **use_align**: True (默认) - 是否使用符号对齐度量
- **use_cosine**: True (默认) - 是否使用余弦相似度度量
- **align_cluster_method**: "spectral" - 聚类方法
- **align_spectral_clusters**: 2 (默认) - 谱聚类的簇数

## 距离矩阵的含义

1. **align_matrix**: 符号对齐矩阵，值越大表示两个客户端的更新方向越一致
2. **cosine_matrix**: 余弦相似度矩阵，值越大表示两个客户端的更新方向越相似
3. **feature_dist_matrix**: 特征距离矩阵，值越小表示两个客户端越相似
4. **similarity_matrix**: 相似度矩阵（用于谱聚类），值越大表示两个客户端越相似

## 输出信息

代码会输出以下日志信息：
- `[AvgAlign3] Pairwise sign-alignment matrix`: 原始符号对齐矩阵
- `[AvgAlign3] Pairwise cosine similarity matrix`: 原始余弦相似度矩阵
- `[AvgAlign3] Normalized pairwise sign-alignment matrix`: 归一化后的符号对齐矩阵
- `[AvgAlign3] Normalized pairwise cosine similarity matrix`: 归一化后的余弦相似度矩阵
- `[AvgAlign3] Feature distance matrix`: 特征距离矩阵
- `[AvgAlign3][Spectral]` 聚类结果和统计信息





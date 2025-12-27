# SoDa 攻击实现核对报告

## 已核对的部分 ✅

### 1. **Self-reference Training 逻辑** ✅
- ✅ 使用干净数据集训练临时模型
- ✅ 提取参考参数 `fixed_params`
- ✅ 学习率使用固定值（与 SoDa-BNGuard 一致）

### 2. **正则化损失** ✅
- ✅ L2 损失权重：0.1
- ✅ 余弦相似度损失权重：100
- ✅ 公式：`minibatch_loss + 0.1 * l2_loss + 100 * (1 - cos_loss)`

### 3. **clean_backup_dataset 和 clean_train_loader** ✅
- ✅ 在毒化之前备份干净数据集
- ✅ 使用已分割的数据集备份（与 SoDa-BNGuard 一致）

### 4. **参数检查** ✅
- ✅ 使用 `hasattr` 和 `is not None` 进行安全检查

## 发现的差异（需要评估）

### 1. **poison_idxs 的生成方式** ⚠️

**SoDa-BNGuard**:
```python
poison_idxs = random.sample(data_idxs, math.floor(poison_frac * len(data_idxs)))
utils.poison_dataset(train_dataset, args, poison_idxs, ...)
```
- 直接从 `data_idxs` 中随机选择
- 不排除任何类别

**AlignIns**:
```python
utils.poison_dataset(train_dataset, args, data_idxs, ...)
# 在 poison_dataset 内部：
all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
all_idxs = list(set(all_idxs).intersection(data_idxs))
poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
```
- 先排除 `target_class` 的样本
- 然后从剩余样本中随机选择

**影响评估**：
- 这个差异是 AlignIns 的设计选择（通常不会毒化已经是目标类别的样本）
- 对于 SoDa 攻击的核心机制（self-reference training + 正则化）**没有影响**
- 只影响数据毒化的样本选择，不影响攻击的核心逻辑

**建议**：保持当前实现，这是合理的防御性编程

### 2. **学习率衰减** ⚠️

**SoDa-BNGuard**:
- Self-reference training: 固定学习率 `lr=self.args.client_lr`
- 正常训练: 固定学习率 `lr=self.args.client_lr`

**AlignIns**:
- Self-reference training: 固定学习率 `lr=self.args.client_lr` ✅
- 正常训练: 衰减学习率 `lr=self.args.client_lr * (self.args.lr_decay) ** round`

**影响评估**：
- Self-reference training 的学习率已统一为固定值 ✅
- 正常训练的学习率衰减是 AlignIns 的设计选择
- 对 SoDa 攻击的核心机制**影响较小**（主要依赖 self-reference training）

**建议**：保持当前实现，或提供选项让用户选择是否使用学习率衰减

### 3. **self-reference training 的条件检查** ✅

**SoDa-BNGuard**:
```python
if self.is_malicious and self.args.attack == 'soda':
```

**AlignIns**:
```python
if self.is_malicious and self.args.attack == 'soda' and hasattr(self, 'clean_train_loader'):
```

**评估**：
- AlignIns 的额外检查更安全，防止属性不存在时的错误
- 如果 `attack == 'soda'`，`clean_train_loader` 应该已经创建
- 这个检查是**防御性编程**，没有问题

### 4. **fixed_params 的作用域** ✅

**SoDa-BNGuard**:
```python
if self.is_malicious and self.args.attack == 'soda':
    fixed_params = self.get_model_parameters(temp_model)
# 使用时：
if self.is_malicious and self.args.attack == 'soda':
    # 使用 fixed_params
```

**AlignIns**:
```python
if self.is_malicious and self.args.attack == 'soda' and hasattr(self, 'clean_train_loader'):
    fixed_params = self.get_model_parameters(temp_model)
else:
    fixed_params = None
# 使用时：
if self.is_malicious and self.args.attack == 'soda' and fixed_params is not None:
    # 使用 fixed_params
```

**评估**：
- AlignIns 的处理更安全，确保 `fixed_params` 总是有值
- 额外的 `is not None` 检查是**防御性编程**，没有问题

## 总结

### 核心功能 ✅
所有 SoDa 攻击的核心功能都已正确实现：
- ✅ Self-reference training
- ✅ 正则化损失（L2 + 余弦相似度）
- ✅ 干净数据集的备份和使用

### 设计差异（不影响核心功能）
1. **poison_idxs 生成**：AlignIns 排除 target_class，这是合理的设计选择
2. **学习率衰减**：正常训练使用衰减，self-reference training 使用固定值（正确）
3. **安全检查**：AlignIns 有更多的防御性检查，这是好的实践

### 建议
当前实现是**正确的**，可以正常使用。差异主要是设计选择，不影响 SoDa 攻击的核心机制。


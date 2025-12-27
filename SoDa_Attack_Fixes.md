# SoDa 攻击实现修复说明

## 发现的问题及修复

### 1. ✅ **clean_backup_dataset 的创建时机和内容**（已修复）

**问题**：
- 原实现：在毒化之前备份原始数据集 `train_dataset`，然后在创建 `clean_train_loader` 时重新分割
- SoDa-BNGuard：在毒化之前备份已分割的干净数据集 `self.train_dataset`

**修复**：
- 对于 SoDa 攻击，在毒化之前备份已分割的干净数据集 `self.train_dataset`
- 直接使用备份的数据集创建 `clean_train_loader`，无需重新分割
- 对于其他攻击，保持原有逻辑（备份原始数据集）

**代码位置**：`agent.py` lines 23-40

### 2. ✅ **学习率衰减处理**（已修复）

**问题**：
- 原实现：self-reference training 使用衰减学习率 `lr=self.args.client_lr * (self.args.lr_decay) ** round`
- SoDa-BNGuard：self-reference training 使用固定学习率 `lr=self.args.client_lr`

**修复**：
- 统一为固定学习率，与 SoDa-BNGuard 保持一致
- 正常训练阶段仍使用学习率衰减（如果 AlignIns 需要）

**代码位置**：`agent.py` line 82

### 3. ✅ **clean_train_loader 的创建时机**（已修复）

**问题**：
- 原实现：在 `__init__` 方法的最后创建 `clean_train_loader`
- SoDa-BNGuard：在毒化之前创建 `clean_train_loader`

**修复**：
- 在毒化之前创建 `clean_train_loader`，确保使用的是干净数据
- 移除了重复的创建逻辑

**代码位置**：`agent.py` lines 25-35

## 已验证正确的部分

### 1. ✅ **Self-reference training 逻辑**
- 使用干净数据集训练临时模型
- 提取参考参数 `fixed_params`
- 实现正确

### 2. ✅ **正则化损失**
- L2 损失：`0.1 * l2_loss`
- 余弦相似度损失：`100 * (1 - cos_loss)`
- 权重与 SoDa-BNGuard 一致

### 3. ✅ **参数检查**
- 使用 `hasattr(self, 'clean_train_loader')` 检查是否存在
- 使用 `fixed_params is not None` 检查是否已定义
- 逻辑正确

## 当前实现状态

### 核心功能 ✅
- [x] Self-reference training 阶段
- [x] 正则化损失（L2 + 余弦相似度）
- [x] 干净数据集的备份和加载
- [x] 学习率处理

### 与 SoDa-BNGuard 的差异

1. **数据毒化方式**：
   - SoDa-BNGuard：使用 OOD 数据（如 MNIST）作为后门触发器
   - AlignIns：使用传统后门模式（如 plus 模式）
   - **影响**：不影响 SoDa 攻击的核心机制（self-reference training + 正则化）

2. **学习率衰减**：
   - SoDa-BNGuard：正常训练也使用固定学习率
   - AlignIns：正常训练使用学习率衰减
   - **影响**：不影响 SoDa 攻击的核心机制

## 测试建议

1. **验证 clean_train_loader 使用干净数据**：
   - 检查 self-reference training 阶段使用的数据是否未被毒化
   - 可以通过打印数据标签或可视化数据来验证

2. **验证正则化损失生效**：
   - 检查训练日志中是否有 "Self-reference training" 和 "Self-reference finished" 消息
   - 检查损失值是否包含正则化项

3. **对比攻击效果**：
   - 与 SoDa-BNGuard 的实现对比攻击成功率（ASR）
   - 检查是否能绕过防御机制

## 注意事项

1. **数据集备份**：
   - SoDa 攻击需要干净数据集用于 self-reference training
   - 确保 `clean_backup_dataset` 在毒化之前创建

2. **内存使用**：
   - 备份数据集会增加内存使用
   - 如果内存不足，可以考虑只备份必要的数据索引

3. **兼容性**：
   - 当前实现与 AlignIns 的其他攻击方式兼容
   - SoDa 攻击不会影响其他攻击的实现


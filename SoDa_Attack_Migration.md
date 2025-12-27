# SoDa 攻击迁移说明

## 概述
本文档说明了如何将 SoDa-BNGuard 目录下的 SoDa 攻击迁移到 AlignIns 目录下。

## 迁移内容

### 1. 核心机制
SoDa 攻击的核心思想是：
- **Self-reference training**: 使用干净数据集训练一个临时模型，得到参考参数 `fixed_params`
- **正则化损失**: 在正常训练过程中，添加额外的损失项来保持模型参数与参考模型接近：
  - L2 距离损失：`0.1 * l2_loss`
  - 余弦相似度损失：`100 * (1 - cos_loss)`

### 2. 代码修改

#### 2.1 `AlignIns/src/agent.py`

**添加的方法：**
- `get_model_parameters(model)`: 提取模型所有参数并展平为向量

**修改的 `__init__` 方法：**
- 为 SoDa 攻击创建 `clean_train_loader`，用于 self-reference training
- 条件：`self.id < args.num_corrupt and self.args.attack == 'soda' and hasattr(self, 'clean_backup_dataset')`

**修改的 `local_train` 方法：**
- 添加 SoDa 攻击的 self-reference training 阶段
- 在正常训练循环中添加正则化损失项

#### 2.2 `AlignIns/src/federated.py`

**修改的参数：**
- 在 `--attack` 参数的 `choices` 中添加 `"soda"` 选项

## 使用方法

运行 SoDa 攻击的示例命令：

```bash
python federated.py --attack soda --aggr alignins --data cifar10 --num_corrupt 2 --poison_frac 0.3
```

## 注意事项

1. **数据集差异**：
   - SoDa-BNGuard 使用 OOD 数据集（如 MNIST）作为后门触发器
   - AlignIns 使用传统的后门模式（如 plus 模式）
   - SoDa 攻击的核心机制（self-reference training + 正则化）仍然可以工作

2. **依赖项**：
   - 确保 `clean_backup_dataset` 在恶意客户端初始化时被创建
   - 当前实现中，当 `self.id < args.num_corrupt and self.args.attack != 'non'` 时会创建 `clean_backup_dataset`

3. **超参数**：
   - L2 损失权重：`0.1`
   - 余弦相似度损失权重：`100`
   - 这些权重与 SoDa-BNGuard 中的实现保持一致

## 验证

迁移完成后，可以通过以下方式验证：
1. 检查代码是否有语法错误（已通过 linter 检查）
2. 运行一个小规模测试，确认 SoDa 攻击能够正常执行
3. 检查日志输出，确认 "Self-reference training" 和 "Self-reference finished" 消息出现

## 参考

- SoDa-BNGuard 原始实现：`SoDa-BNGuard/src/agent.py` (lines 45-99)
- SoDa 攻击论文：Self-Distillation for Backdoor Attack (SoDa)


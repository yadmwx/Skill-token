# SE-FM 训练问题分析

## 问题1: Offline Loss 不在 [-1, 1] 范围内

### 原因分析

1. **RLDS数据集中的actions已经归一化**：
   - 在数据加载时（`prismatic/vla/datasets/rlds/dataset.py`），通过`normalize_action_and_proprio`函数
   - Actions被归一化到`[-1, 1]`范围（使用`BOUNDS_Q99`时）
   - 公式：`2 * (x - q01) / (q99 - q01) - 1`

2. **predict_action()返回的动作是未归一化的**：
   - `DiffusionActionHead.predict_action()`返回的动作在**原始空间**（未归一化）
   - 这些动作需要经过反归一化才能用于环境执行

3. **之前的错误**：
   - 我们对`target_actions`（已经归一化）和`pred_actions`（未归一化）都进行了归一化
   - 这导致`target_actions`被归一化了两次，导致范围错误

### 修复方案

- ✅ **只对`pred_actions`归一化**，然后和已经归一化的`target_actions`比较
- ✅ 添加了调试信息，检查归一化后的范围

---

## 问题2: Flow-Matching vs Diffusion 训练方式

### 当前实现

当前使用的是 **DDPM (Denoising Diffusion Probabilistic Models)**，不是真正的 Flow Matching：

1. **推理时**（`predict_action`）：
   - 从随机噪声开始
   - 通过50步去噪过程生成动作
   - 返回最终去噪后的动作（原始空间）

2. **训练时**（`compute_offline_loss`）：
   - 直接用最终去噪后的动作和target做L1 loss
   - **这不是正确的diffusion训练方式！**

### 正确的Diffusion训练方式

Diffusion模型的正确训练应该是：

```python
# 训练时：
1. 对target_actions加噪声：noisy_actions = add_noise(target_actions, timestep)
2. 预测噪声：noise_pred = predict_noise(noisy_actions, timestep, conditions)
3. Loss = MSE(noise_pred, actual_noise)

# 推理时：
1. 从随机噪声开始
2. 迭代去噪：noisy_actions = denoise(noisy_actions, noise_pred, timestep)
3. 返回最终去噪后的动作
```

### 当前问题

- ❌ 我们直接用最终去噪后的动作和target做loss
- ❌ 这相当于在训练一个"一步到位"的回归模型，而不是训练去噪过程
- ❌ 这可能导致模型无法学习到正确的去噪路径

### 建议的修复

如果要正确训练Diffusion模型，应该：

1. **在训练时**：
   - 随机采样timestep `t`
   - 对target_actions加噪声得到`noisy_actions`
   - 预测噪声：`noise_pred = action_head.predict_noise(noisy_actions, t, ...)`
   - Loss = MSE/L1(noise_pred, actual_noise)

2. **或者**，如果只想做简单的BC训练：
   - 使用`L1RegressionActionHead`而不是`DiffusionActionHead`
   - 直接回归到target actions

---

## 问题3: VLM (Qwen 0.5B) 的作用

### VLM在训练中的角色

1. **生成条件信息（actions_hidden_states）**：
   ```
   VLM (Qwen 0.5B) 
   → 输入：图像 + 语言指令
   → 输出：actions_hidden_states (batch, num_layers, num_tokens, hidden_dim)
   ```

2. **作为动作头的条件输入**：
   - `actions_hidden_states`包含：
     - **task_hidden_states**: 视觉+语言特征（前512个tokens）
     - **action_hidden_states**: 动作相关的特征（后NUM_TOKENS个tokens）
   - 这些特征通过交叉注意力融合到去噪网络中

3. **在SE-FM训练中**：
   - VLM是**冻结的**（`model.eval()`, `p.requires_grad = False`）
   - 只训练`DiffusionActionHead`
   - VLM提供稳定的条件特征，指导动作生成

### 为什么VLM是必要的

- **多模态理解**：VLM理解图像和语言指令，生成语义丰富的特征
- **条件生成**：动作头需要这些特征来生成与任务相关的动作
- **迁移学习**：预训练的VLM已经学会了视觉-语言-动作的关联

### 训练流程

```
1. 离线数据加载：
   - 图像 + 语言指令 → VLM → actions_hidden_states
   - target_actions (已归一化)

2. 动作预测：
   - actions_hidden_states → DiffusionActionHead → pred_actions (未归一化)

3. Loss计算：
   - 归一化pred_actions → 与target_actions (已归一化) 比较 → L1 Loss

4. 反向传播：
   - 只更新DiffusionActionHead的参数
   - VLM参数保持不变
```

---

## 总结和建议

### 已修复的问题

1. ✅ **归一化问题**：只对pred_actions归一化，target_actions已经是归一化的
2. ✅ **Loss函数**：从MSE改为L1 Loss（匹配VLA-Adapter）
3. ✅ **添加调试信息**：检查归一化范围和梯度

### 需要进一步考虑的问题

1. **Diffusion训练方式**：
   - 当前直接用最终动作做loss，不是正确的diffusion训练
   - 建议：要么改用正确的diffusion训练（预测噪声），要么改用`L1RegressionActionHead`

2. **学习率**：
   - 当前：`5e-5`
   - 原始VLA-Adapter：`2e-4`
   - 建议：尝试`1e-4`或`2e-4`

3. **梯度检查**：
   - 已添加梯度范数监控
   - 如果梯度太小，需要增大学习率或检查模型初始化


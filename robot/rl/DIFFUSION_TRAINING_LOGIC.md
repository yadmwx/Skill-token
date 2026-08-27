# DiffusionActionHead 训练逻辑梳理

## 📋 训练流程概览

### 1. 数据准备阶段

```python
# 从 RLDS 数据集加载
target_actions: (batch, NUM_ACTIONS_CHUNK, ACTION_DIM)  # 已归一化到 [-1, 1]
actions_hidden_states: (batch, num_layers, num_tokens, hidden_dim)  # VLM 输出
proprio: (batch, PROPRIO_DIM)  # 本体感知
```

**关键点**：
- `target_actions` 已经在 RLDS 数据集中归一化到 `[-1, 1]` 范围
- 使用 `normalize_action_and_proprio()` 函数，基于 `q01/q99` 或 `min/max` 统计量

### 2. 噪声添加阶段

```python
# 随机采样 timesteps
timesteps = torch.randint(0, T, (batch_size,))  # T = 1000 (num_train_timesteps)

# 生成标准高斯噪声
noise = torch.randn_like(target_actions)  # N(0, 1)

# 使用 scheduler.add_noise() 添加噪声
noisy_actions_norm = scheduler.add_noise(target_actions, noise, timesteps)
```

**DDPMScheduler.add_noise() 的实现**：
```python
# 伪代码（来自 diffusers 库）
def add_noise(self, original_samples, noise, timesteps):
    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    
    noisy_samples = (
        sqrt_alpha_prod * original_samples + 
        sqrt_one_minus_alpha_prod * noise
    )
    return noisy_samples
```

**关键点**：
- `alphas_cumprod` 是预计算的累积乘积，范围从 `[1, 0]`（t=0 到 t=1000）
- 当 `t=0` 时：`sqrt_alpha_prod ≈ 1`，`sqrt_one_minus_alpha_prod ≈ 0` → 几乎无噪声
- 当 `t=1000` 时：`sqrt_alpha_prod ≈ 0`，`sqrt_one_minus_alpha_prod ≈ 1` → 几乎全是噪声
- **预测目标**：模型需要预测原始的 `noise`（不是缩放后的噪声）

### 3. 噪声预测阶段

```python
pred_noise = action_head.predict_noise(
    noisy_action=noisy_actions_norm,  # (batch, num_chunks, action_dim)
    timestep=timesteps,                # (batch,)
    hidden=actions_hidden_states,      # (batch, num_layers, num_tokens, hidden_dim)
    proprio=proprio,                   # (batch, proprio_dim)
    proprio_projector=proprio_projector
)
```

**predict_noise() 内部流程**：
1. 将 `noisy_actions` 转换为模型 dtype（bfloat16）
2. 使用 `time_encoder` 编码 timestep → `timestep_emb` (batch, hidden_dim)
3. 使用 `proprio_projector` 投影 proprio → `proprio_features` (batch, 1, hidden_dim)
4. 提取 `task_hidden_states` 和 `action_hidden_states`
5. 通过 `denoising_network` 预测噪声

**DenoisingNetwork 结构**：
```
noisy_actions (batch, num_chunks, action_dim)
    ↓ [action_proj]
hidden_features (batch, num_chunks, hidden_dim)
    ↓ [+ timestep_emb + proprio_emb]
    ↓ [12/24 DenoisingBlocks with cross-attention]
    ↓ [output_proj]
pred_noise (batch, num_chunks, action_dim)
```

### 4. Loss 计算阶段

```python
# 标准 DDPM Loss: MSE(predicted_noise, actual_noise)
bc_loss = F.mse_loss(pred_noise, noise)
```

**关键点**：
- Loss 是预测噪声和实际噪声之间的 MSE
- `noise` 是原始的 `N(0, 1)` 噪声（不是缩放后的）
- 对于归一化到 `[-1, 1]` 的动作，预期 loss 范围：
  - 随机初始化：`[0, ~10]` 或更高
  - 训练良好：`[0, ~1]`

---

## ⚠️ 潜在问题分析

### 问题 1: 数据归一化不一致

**现象**：
- Loss 在 `0.5-2.5` 之间波动，不收敛
- 紫色曲线（L1Regression）从 `1,500,000` 快速下降到 `0`

**可能原因**：
1. `target_actions` 可能没有正确归一化到 `[-1, 1]`
2. `action_stats` 可能为 `None`，导致无法正确归一化
3. 不同实验使用了不同的归一化方法

**检查点**：
```python
# 在 compute_offline_loss 中检查
print(f"target_actions range: [{target_actions.min():.3f}, {target_actions.max():.3f}]")
print(f"action_stats: {action_stats}")
```

### 问题 2: 噪声尺度不匹配

**现象**：
- Loss 不收敛，但梯度正常

**可能原因**：
1. `scheduler.add_noise()` 返回的 `noisy_actions_norm` 的尺度可能不对
2. 预测的噪声和实际噪声的尺度不匹配

**检查点**：
```python
# 检查噪声统计
print(f"noise: mean={noise.mean():.4f}, std={noise.std():.4f}")
print(f"pred_noise: mean={pred_noise.mean():.4f}, std={pred_noise.std():.4f}")
print(f"noisy_actions_norm: mean={noisy_actions_norm.mean():.4f}, std={noisy_actions_norm.std():.4f}")
```

### 问题 3: 梯度裁剪过小

**现象**：
- Loss 波动但不下降
- 梯度范数被裁剪得太小

**可能原因**：
1. `grad_clip=0.5` 可能太小，导致梯度被过度裁剪
2. 梯度范数本身很小，但被裁剪后更小

**检查点**：
```python
# 在训练循环中检查
print(f"grad_norm_before: {grad_norm_before:.6f}")
print(f"grad_norm_after: {grad_norm_after:.6f}")
print(f"grad_clip: {cfg.grad_clip}")
```

### 问题 4: 学习率过小

**现象**：
- Loss 几乎不变
- 梯度正常但更新很小

**可能原因**：
1. `learning_rate=1e-4` 对于随机初始化的模型可能太小
2. 需要更高的学习率（`2e-4` 到 `5e-4`）

**检查点**：
```python
# 检查参数更新
for name, param in action_head.named_parameters():
    if param.grad is not None:
        param_update = param.grad * cfg.learning_rate
        print(f"{name}: update_norm={param_update.norm():.6f}")
```

### 问题 5: Batch Size 太小

**现象**：
- 训练不稳定
- Loss 波动大

**可能原因**：
1. `offline_batch_size=1` 导致梯度估计不准确
2. 需要更大的 batch size（但受 GPU 内存限制）

---

## 🔧 修复建议

### 1. 验证数据归一化

```python
# 在 compute_offline_loss 开始处添加
assert target_actions.min() >= -1.1 and target_actions.max() <= 1.1, \
    f"target_actions not normalized: range=[{target_actions.min():.3f}, {target_actions.max():.3f}]"
```

### 2. 检查噪声预测

```python
# 在 loss 计算前添加
noise_scale = noise.std()
pred_scale = pred_noise.std()
if abs(noise_scale - pred_scale) > 0.5:
    print(f"[WARN] Noise scale mismatch: noise_std={noise_scale:.4f}, pred_std={pred_scale:.4f}")
```

### 3. 调整超参数

```python
# 建议的超参数
learning_rate = 2e-4  # 或 5e-4
grad_clip = 1.0       # 或 2.0
offline_batch_size = 2  # 如果 GPU 内存允许
```

### 4. 添加更多诊断信息

```python
# 在训练循环中添加
if iteration % 10 == 0:
    print(f"[DEBUG] Loss breakdown:")
    print(f"  - bc_loss: {bc_loss.item():.6f}")
    print(f"  - noise_std: {noise.std().item():.4f}")
    print(f"  - pred_noise_std: {pred_noise.std().item():.4f}")
    print(f"  - target_actions_range: [{target_actions.min().item():.3f}, {target_actions.max().item():.3f}]")
    print(f"  - grad_norm: {grad_norm_before:.6f} -> {grad_norm_after:.6f}")
```

---

## 📊 预期行为

### 正常训练曲线

1. **初始阶段**（前 10-50 步）：
   - Loss: `~2-5`（随机初始化）
   - 梯度范数: `~0.1-1.0`
   - Loss 应该快速下降

2. **中期阶段**（50-200 步）：
   - Loss: `~0.5-1.5`
   - 梯度范数: `~0.01-0.1`
   - Loss 应该稳定下降

3. **后期阶段**（200+ 步）：
   - Loss: `~0.1-0.5`
   - 梯度范数: `~0.001-0.01`
   - Loss 应该缓慢下降或收敛

### 异常情况

1. **Loss 不下降**：
   - 检查学习率和梯度裁剪
   - 检查数据归一化
   - 检查噪声预测是否正确

2. **Loss 波动大**：
   - 增加 batch size
   - 降低学习率
   - 检查梯度是否正常

3. **Loss 为 NaN/Inf**：
   - 检查数据是否有异常值
   - 检查梯度是否爆炸
   - 检查模型参数是否正常

---

## 🎯 下一步行动

1. **添加详细的诊断信息**到 `compute_offline_loss()`
2. **验证数据归一化**是否正确
3. **检查噪声预测**的尺度是否匹配
4. **调整超参数**（学习率、梯度裁剪）
5. **对比实验**：使用 L1RegressionActionHead 作为 baseline



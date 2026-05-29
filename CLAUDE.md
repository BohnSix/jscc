# EHB-NTSCC: Expert Hierarchical Bottleneck with Cross-Attention JSCC

## 概述

EHB-NTSCC 在 NTSCC 的 Swin Transformer 非线性变换基础上：
1. 利用 AnalysisTransform (g_a) 四个层级的中间特征作为四个专家的输入
2. 无条件融合（concat + Conv1x1），不依赖 SNR/eta
3. RatePredictor 从融合特征预测连续码率值
4. Cross-attention JSCC encoder/decoder：rate 和 SNR embedding 作为 KV 调制图片 token (Q)
5. RateAdaptionEncoder 根据预测码率选择传输特征

与原始 NTSCC 完全向后兼容：`config.ehb_mode = False` 时行为不变。

---

## 架构数据流

```
输入图像 x (B, 3, 256, 256)
    │
    ▼
┌─ AnalysisTransform (g_a) ── forward_hierarchical() ──┐
│  Layer 0: → F1 (B, 256, 128, 128)                    │
│  Layer 1: → F2 (B, 256, 64, 64)                      │
│  Layer 2: → F3 (B, 256, 32, 32)                      │
│  Layer 3: → F4 (B, 256, 16, 16)                      │
└───────────────────────────────────────────────────────┘
    │ F1, F2, F3, F4
    ▼
┌─ EHBExperts (strided conv 下采样到 16×16) ────────────┐
│  Expert1: F1 (128×128) → U1 (B, 256, 16, 16)         │
│  Expert2: F2 (64×64)  → U2 (B, 256, 16, 16)          │
│  Expert3: F3 (32×32)  → U3 (B, 256, 16, 16)          │
│  Expert4: F4 (16×16)  → U4 (B, 256, 16, 16)          │
└───────────────────────────────────────────────────────┘
    │ U1, U2, U3, U4
    ▼
┌─ ExpertFusion (无条件) ───────────────────────────────┐
│  Z = Conv1×1([U1, U2, U3, U4])  (B, 256, 16, 16)    │
└───────────────────────────────────────────────────────┘
    │ Z
    ▼
┌─ RatePredictor ───────────────────────────────────────┐
│  GlobalAvgPool(Z) → MLP(256→64→1) → Sigmoid × 256    │
│  输出: rate_continuous (B,) → searchsorted → rate_idx │
└───────────────────────────────────────────────────────┘
    │ rate_idx
    ▼
┌─ Conditioning ────────────────────────────────────────┐
│  rate_embed = Embedding(rate_idx)         (B, 256)    │
│  snr_embed = MLP(SNR/20)                  (B, 256)    │
│  cond = stack([rate_embed, snr_embed])    (B, 2, 256) │
└───────────────────────────────────────────────────────┘
    │ cond
    ▼
┌─ JSCCEncoderCross (Cross-Attention × 4) ─────────────┐
│  Z tokens + pos_embed → (B, 256, 256)                 │
│  × 4 layers: Q=tokens, KV=cond → CrossAttn + FFN     │
│  → norm → RateAdaptionEncoder(indexes)                │
│  输出: s_masked, mask_BCHW                            │
└───────────────────────────────────────────────────────┘
    │
    ▼  Channel (功率归一化 + AWGN)
    │
    ▼
┌─ JSCCDecoderCross (Cross-Attention × 4) ─────────────┐
│  RateAdaptionDecoder → tokens + pos_embed             │
│  × 4 layers: Q=tokens, KV=cond → CrossAttn + FFN     │
│  → norm → y_hat (B, 256, 16, 16)                     │
└───────────────────────────────────────────────────────┘
    │
    ▼  SynthesisTransform (g_s) → x_hat (B, 3, 256, 256)
```

---

## 模块详解

### ExpertHead (`layer/ehb_modules.py`)

每个专家通过 strided conv 下采样到 16×16：
- Expert1 (128→16): 3 层 stride-2 conv + PReLU + Conv1×1
- Expert2 (64→16): 2 层 stride-2 conv + PReLU + Conv1×1
- Expert3 (32→16): 1 层 stride-2 conv + PReLU + Conv1×1
- Expert4 (16→16): 仅 Conv1×1

### ExpertFusion (`layer/ehb_modules.py`)

无条件融合：`Conv1x1(cat([U1, U2, U3, U4]))` → Z (B, 256, 16, 16)

不使用 GateNet/alpha/SNR/eta 条件。

### RatePredictor (`layer/ehb_modules.py`)

从融合特征 Z 预测 per-image 连续码率：
```
GlobalAvgPool(Z) → Linear(256, 64) → ReLU → Linear(64, 1) → Sigmoid × max_rate
```
输出 ∈ [0, 256]，通过 searchsorted 映射到离散 rate_choice 索引。

### CrossAttentionBlock (`layer/layers.py`)

```python
Q = norm(image_tokens)      # (B, 256, 256)
KV = cond_tokens            # (B, 2, 256) — rate + SNR embeddings
x = x + MultiheadCrossAttn(Q, K=cond, V=cond)
x = x + FFN(norm(x))
```

Rate/SNR 信息通过 cross-attention 调制每个图片 token，比 additive token 注入更强。

### JSCCEncoderCross (`layer/jscc_encoder_cross.py`)

- 输入: Z (B, 256, 16, 16), cond (B, 2, 256), indexes
- flatten + pos_embed → CrossAttentionBlock × 4 → norm → RateAdaptionEncoder

### JSCCDecoderCross (`layer/jscc_decoder_cross.py`)

- 输入: s_hat (B, max_rate, 16, 16), indexes, cond
- RateAdaptionDecoder → pos_embed → CrossAttentionBlock × 4 → norm → output

---

## 损失函数

```
L = mse_loss + λ_red · redundancy_loss + λ_cbr · rate_loss
```

### redundancy_cosine_loss (`loss/ehb_loss.py`)

惩罚专家输出间的余弦相似度：
```
L_red = (1/6) Σ_{i<j} (cos_sim(flatten(Ui), flatten(Uj)))²
```

### rate_loss (main.py)

驱动 RatePredictor 学习目标码率：
```
target_rate = ehb_target_cbr × 16 × 16 × 3 × 2
L_rate = mean((rate_continuous - target_rate)²)
```

注：searchsorted 不可导，rate_loss 直接作用于 rate_continuous 提供梯度。

---

## CBR 计算

```python
channel_input = torch.masked_select(s_masked, mask_BCHW)
channel_output, channel_usage = self.channel.forward(channel_input)
cbr_y = channel_usage / (num_pixels * B)
```

- channel 将每 2 个实数配对为 1 个复数符号
- `channel_usage` = 复数符号总数
- CBR = 每像素复数符号数

---

## 配置参数 (`config.py`)

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `ehb_mode` | `True` | 启用 EHB-NTSCC |
| `ehb_num_experts` | 4 | 专家数量 |
| `ehb_use_red_loss` | `True` | 启用冗余损失 |
| `ehb_lambda_red` | 1e-4 | 冗余损失权重 |
| `ehb_rate_hidden_dim` | 64 | RatePredictor 隐藏层维度 |
| `ehb_target_cbr` | 0.08 | 目标 CBR |
| `ehb_lambda_cbr` | 10.0 | CBR 损失权重 |

其他关键参数：
- `channel`: AWGN, SNR=10 dB
- `multiple_rate`: [16, 32, ..., 256]，步长 16
- `image_dims`: (3, 256, 256)
- `batch_size`: 4, `lr`: 1e-4

---

## 与 NTSCC 的核心区别

### NTSCC 流程

```
x → g_a → y → Hyperprior → likelihoods
    y + likelihoods × eta → JSCCEncoder(per-patch rate) → Channel → JSCCDecoder → g_s → x̂
```

### EHB-NTSCC 流程

```
x → g_a → F1~F4 → Experts → U1~U4 → Fusion → Z
    Z → RatePredictor → rate_idx
    rate_embed + snr_embed → cond
    Z + cond → CrossAttn Encoder → Channel → CrossAttn Decoder → g_s → x̂
```

### 关键差异

| 维度 | NTSCC | EHB-NTSCC |
|------|-------|-----------|
| 特征提取 | 仅最终 y | 四层中间特征 F1~F4 |
| 复杂度/码率估计 | Hyperprior (大量参数) | RatePredictor (轻量 MLP) |
| 码率分配 | per-patch (likelihoods × eta) | per-image (RatePredictor) |
| 信道适配方式 | additive rate_token + self-attention | cross-attention (rate/SNR as KV) |
| SNR 注入 | 无（训练时固定） | SNR embedding 作为 KV |
| Side info | 需要传输 z 的 bpp | 不需要 |
| 解耦约束 | 无 | 冗余损失 |

---

## 文件结构

```
layer/analysis_transform.py    forward_hierarchical() 输出四层特征
layer/ehb_modules.py           EHBExperts, ExpertFusion, RatePredictor
layer/layers.py                CrossAttentionBlock, ViTBlock
layer/jscc_encoder_cross.py    JSCCEncoderCross (cross-attention 编码器)
layer/jscc_decoder_cross.py    JSCCDecoderCross (cross-attention 解码器)
loss/ehb_loss.py               redundancy_cosine_loss

net/NTSCC_Hyperior.py
├── NTC_Hyperprior             (原有)
├── NTSCC_Hyperprior           (原有，ehb_mode=False)
└── EHB_NTSCC                  (ehb_mode=True)

config.py                      EHB 参数配置
main.py                        训练/测试入口
```

---

## 向后兼容性

- `ehb_mode=False`：使用 `NTSCC_Hyperprior`，行为与原始完全一致
- `ehb_mode=True`：使用 `EHB_NTSCC`
- 原始 NTSCC checkpoint 在 `ehb_mode=False` 下正常加载

---

## 使用方法

```bash
# 训练 EHB-NTSCC
python main.py -p train --gpu-id 0

# 测试
python main.py -p test --gpu-id 0 --checkpoint path/to/model.pth

# 切回原始 NTSCC: config.py 中 ehb_mode=False
python main.py -p train --gpu-id 0 --checkpoint checkpoints/ntscc_pretrained.pth
```

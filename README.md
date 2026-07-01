# MVTec Layout-Guided Defect Generation

基于 **IC-DiT (In-Context Diffusion Transformer)** 思想的 MVTec AD 缺陷生成研究代码框架。

> ⚠️ **重要说明**: 本项目参考的是 layout-guided + multimodal in-context diffusion 的核心思想，**并非原论文官方复现**。原论文针对的是病理图像生成（TCGA 数据集），本项目将其适配到 MVTec AD 工业缺陷生成场景。

---

## 环境配置

```bash
# 创建并激活 conda 环境
conda create -n omg python=3.10
conda activate omg

# 安装依赖
pip install -r requirements.txt
```

## 核心思想

给定产品类别、缺陷类型、缺陷布局 mask 和可选的正常参考图，模型生成：

1. **符合输入 mask 区域** 的缺陷位置
2. **符合 defect_type 语义** 的缺陷外观
3. **保持产品整体外观** 合理的缺陷图像

### 架构概览

```
Text Prompt  ──→ Frozen Text Encoder (CLIP/T5) ──→ Text Projector ──→ text tokens
Defect Mask  ──→ Frozen Layout Encoder (VAE)   ──→ Layout Projector ──→ layout tokens
Target Image ──→ Frozen VAE Encoder            ──→ (latent) + noise
Reference    ──→ Frozen Visual Encoder (DINOv2)──→ Visual Projector ──→ visual tokens
                                                                         │
                                                            ┌────────────┘
                                                            ▼
                                              MM-Attention Fusion Blocks
                                              + Latent DiT Denoiser
                                                            │
                                                            ▼
                                              Frozen VAE Decoder
                                                            │
                                                            ▼
                                               Generated Defect Image
```

## 模块冻结/训练状态

### Frozen（冻结，不更新梯度）

| 模块 | 实现 | 理由 |
|------|------|------|
| Text Encoder | CLIP-ViT-L/14 或 T5 | 保留预训练语义知识 |
| VAE Encoder | SD VAE (ft-mse) | 稳定的 latent space |
| VAE Decoder | SD VAE (ft-mse) | 稳定解码 |
| Layout Encoder | VAE 编码 binary mask | 复用预训练编码能力 |
| Visual Encoder | DINOv2 ViT-B/14 | 保留自监督视觉先验 |

### Trainable（可训练）

| 模块 | 说明 |
|------|------|
| Text Projector | text_dim → hidden_dim (768) |
| Layout Projector | layout_dim → hidden_dim (768) |
| Visual Projector | visual_dim → hidden_dim (768) |
| MM-Attention Blocks | image↔text, image↔layout, image↔visual 交叉注意力 |
| Latent DiT Backbone | patch embedding, positional emb, transformer layers, unpatchify |
| Timestep Embedding | 扩散步数条件编码 |

---

## 数据准备

MVTec AD 数据集需按以下结构组织：

```text
mvtec/
  bottle/
    train/
      good/
        xxx.png
    test/
      good/
      broken_large/
      broken_small/
      contamination/
      ...
    ground_truth/
      broken_large/
        xxx_mask.png
      broken_small/
        xxx_mask.png
      contamination/
        xxx_mask.png
  capsule/
    ...
```

下载地址: [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 训练

```bash
conda activate omg

# 修改 configs/mvtec_icdit.yaml 中的 data.root 为你的 MVTec 路径

# 单卡训练
accelerate launch train.py --config configs/mvtec_icdit.yaml

# 多卡训练 (自动检测)
accelerate launch --multi_gpu train.py --config configs/mvtec_icdit.yaml
```

### 训练输出
```
outputs/
  checkpoints/
    best.pt                    # 最佳验证损失模型
    step_0001000.pt           # 定期保存
  samples/
    samples_step_001000.png   # 定期生成的 sample 图像
  logs/
    events.out.tfevents.*     # TensorBoard 日志
  config.yaml                  # 配置备份
```

### 查看日志
```bash
tensorboard --logdir outputs/logs
```

---

## 采样/推理

```bash
conda activate omg

# 使用已有 mask 生成缺陷图像
python sample.py \
  --ckpt outputs/checkpoints/best.pt \
  --category bottle \
  --defect_type broken_large \
  --mask_path demo/mask.png \
  --reference_image demo/normal.png \
  --output_dir ./outputs

# 生成无缺陷图像（all-zero mask）
python sample.py \
  --ckpt outputs/checkpoints/best.pt \
  --category bottle \
  --defect_type good \
  --reference_image demo/normal.png \
  --output_dir ./outputs
```

---

## 检查 Frozen/Trainable 参数

训练启动时会自动打印：

```
============================================================
Parameter Statistics [ICDiTDefectGenerator]
============================================================

--- Frozen Parameters (X groups, N1 params) ---
  [FROZEN]  text_encoder.encoder.embeddings.weight
  [FROZEN]  vae.vae.encoder.conv_in.weight
  ...

--- Trainable Parameters (Y groups, N2 params) ---
  [TRAINABLE] text_projector.0.weight
  [TRAINABLE] latent_dit.patch_embed.weight
  ...

--- Summary ---
  Trainable: N2 (xx.xx%)
  Frozen:    N1 (xx.xx%)
  Total:     N_total
============================================================
```

同时在构建 optimizer 后会运行 `assert_no_frozen_params_in_optimizer` 确保冻结参数未被误加入。

---

## 配置文件说明

主要配置项（`configs/mvtec_icdit.yaml`）：

```yaml
model:
  training_mode: "full_generator"       # "full_generator" | "adapter_lora"
  visual_embedding_source: "reference_normal"  # "reference_normal" | "target"
  hidden_dim: 768                       # 统一隐空间维度
  num_layers: 12                        # MM-Attention 层数
  freeze_*: true                        # 各种冻结开关

diffusion:
  num_train_timesteps: 1000            # 扩散总步数
  num_inference_steps: 50              # 采样步数
  beta_schedule: "linear"              # "linear" | "cosine"

train:
  lr: 1e-4
  batch_size: 8
  gradient_accumulation_steps: 1
  mixed_precision: "fp16"
```

---

## 项目结构

```text
mvtec_layout_guided_defect_generation/
  configs/
    mvtec_icdit.yaml                     # 主配置文件
  datasets/
    __init__.py
    mvtec.py                             # MVTec AD Dataset
  models/
    __init__.py
    frozen_encoders.py                   # FrozenTextEncoder, FrozenVAE, FrozenVisualEncoder
    mask_encoder.py                      # FrozenLayoutEncoder
    mm_attention.py                      # MMAttentionBlock, MMAttentionStack
    latent_dit.py                        # LatentDiT backbone
    icdit_defect_generator.py            # ICDiTDefectGenerator (top-level)
  diffusion/
    __init__.py
    scheduler.py                         # DDPM training + DDIM sampling scheduler
  utils/
    __init__.py
    freeze.py                            # freeze/unfreeze utilities
    image_utils.py                       # Image I/O and transforms
    train_utils.py                       # Seed, config, AverageMeter
    logging_utils.py                     # TensorBoard + console logger
  train.py                               # Training entry point
  sample.py                              # Inference/sampling entry point
  requirements.txt                       # Python dependencies
  README.md                              # This file
```

---

## 后续迭代方向

- [ ] 适配自定义数据集格式
- [ ] 实现 `adapter_lora` 训练模式
- [ ] 添加更多评估指标（FID, LPIPS, Mask IoU）
- [ ] 支持更高分辨率生成
- [ ] CFG 条件组合实验

---

## 引用

本项目参考以下工作：

- **IC-DiT**: Layout-Guided Controllable Pathology Image Generation with In-Context Diffusion Transformers (Shou et al., 2026)
- **DiT**: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
- **MVTec AD**: The MVTec Anomaly Detection Dataset (Bergmann et al., 2019)

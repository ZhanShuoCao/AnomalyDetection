# VLM 异常检测 Prompt 设计笔记

## 核心发现

### 1. 工业异常检测领域 Prompt 设计范式

| 方法 | 思路 | 来源 |
|------|------|------|
| WinCLIP | 状态词 × 模板句式的组合集成 (CPE) | CVPR 2023 |
| AnomalyCLIP | 可学习连续 token 替代手写文本 | ICLR 2024 |
| FAPrompt | 复合异常 prompt + 数据依赖异常先验，实例级自适应 | ICCV 2025 |
| MALM-CLIP | LLM 多智能体自动生成 prompt，解除人工设计 | 2025 |
| IAD-GPT | APG 生成类别专属异常 prompt + TGE 动态增强 | IEEE 2025 |
| SSVP | CLIP(语义) + DINOv3(结构) 层次化语义-视觉协同 | arXiv 2025 |

### 2. 医疗影像领域 Prompt 设计范式

| 方法 | 思路 | 来源 |
|------|------|------|
| 结构化报告 | Findings → Impression 分层生成 | npj Digital Medicine 2025 |
| K2Sight | 临床概念拆解为视觉属性原语(形状/密度/位置) | WACV 2026 |
| 分层区域描述 | 按器官/区域逐一 query，标记正常/异常 | CTPA 2025 |
| CoPS | Context + State + Class 三组件可学习 prompt | arXiv 2025 |

### 3. 关键认知：T5 vs CLIP 编码器

- **CLIP Text Encoder**: 77 token 上限，适合短标签式文本
- **T5 Encoder**: 512+ token，能消化丰富、结构化的长文本描述
- WinCLIP/AnomalyCLIP 等文献实际多用 T5 编码器处理长 prompt
- 因此 VLM 生成的描述可以更丰富详细，不必受 77 token 约束

### 4. MVTec 数据集处理决策

- 只处理异常样本（跳过 `good/` 和 `ground_truth/`）
- 输出格式：`{category, defect_type, image_id, description, tokens}`
- 调用方式：DashScope API (OpenAI 兼容模式)
- 模型：Qwen3-VL 系列

### 参考文献

- WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation (CVPR 2023)
- AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models (AAAI 2024 Oral)
- FAPrompt: Fine-grained Abnormality Prompt Learning (ICCV 2025)
- K2Sight: Knowledge to Sight via Visual Attribute Decomposition (WACV 2026)
- SSVP: Synergistic Semantic-Visual Prompting (arXiv 2025)
- One-for-All Few-Shot AD via Instance-Induced Prompt Learning (ICLR 2025)

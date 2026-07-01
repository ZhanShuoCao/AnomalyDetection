#!/usr/bin/env python3
"""
使用 Qwen3-VL (DashScope API) 对 MVTec 数据集的异常样本生成 VLM 描述。
聚焦：背景、缺陷形态、相对位置与类型。

用法：
  export DASHSCOPE_API_KEY="your-key"
  python3 mvtec_describe.py \
    --data-dir /home/czs/桌面/MVTec \
    --output mvtec_descriptions.json \
    --model qwen3-vl-8b-instruct
"""

import os
import sys
import json
import re
import time
import base64
import argparse
from pathlib import Path
from openai import OpenAI

# ---------- Prompt (T5 encoder — rich structured descriptions) ----------

PROMPT = """You are an industrial defect inspector. Analyze the defect region in this product image.

Describe ONLY the defect's visual appearance using these 5 attributes. Each attribute should be 2-4 sentences with rich visual detail — include texture, color gradients, spatial relationships, and material appearance. Do NOT describe the background or undamaged parts of the product.

Attributes:
- shape: geometric form of the defect
- texture: surface quality within the defect area
- color: color characteristics of the defect
- boundary: edge characteristics between defect and normal area
- extent: size and spatial distribution

Output strictly as JSON with exactly these 5 keys. No extra text outside the JSON.

Example output for a scratched metal surface:
{"shape": "A thin linear groove with parallel edges running diagonally across the surface. The scratch has a slight curvature near its midpoint, creating a gentle arc rather than a perfectly straight line.", "texture": "Bright metallic exposed substrate is visible within the groove. The scratch interior shows fine parallel striations running along the scratch direction, indicating abrasive contact.", "color": "Silver-white metallic scratch against a darker oxidized gray-brown surface. A slight shadow is visible at the deeper sections of the groove where the scratch penetrates more deeply.", "boundary": "Sharp, clean transitions along both edges of the scratch. No chipping, flaking, or material displacement is visible at the boundaries.", "extent": "Extends diagonally across approximately 30% of the visible surface area. The scratch width is consistent at roughly 0.5mm throughout its entire length."}

Example output for a fabric stain:
{"shape": "An irregular diffuse blotch with softly feathered edges. The shape has no clear geometric boundary and exhibits an organic spreading pattern.", "texture": "No surface disruption or roughness is present. The fabric weave remains fully intact and visible through the discoloration with no texture change.", "color": "Dark brown discoloration on light gray fabric. The color intensity gradually fades from the center outward, with the deepest concentration of pigment at the core and lighter haloing at the periphery.", "boundary": "Gradual fade from stain center to clean fabric. No sharp edge definition exists; the transition zone spans approximately 2-3mm of gradient fading.", "extent": "Roughly circular patch covering about 10% of the visible surface area. Centered in the middle-left quadrant of the image."}

Now describe the defect in this image:"""

# ---------- Helpers ----------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# DashScope 配额/限流错误的特征关键词
QUOTA_KEYWORDS = [
    "quota", "insufficient", "exhausted", "limit", "throttl",
    "rate limit", "too many requests", "capacity", "overloaded",
    "resource", "429", "402", "balance", "余额", "额度",
]


def gather_images(data_dir: str) -> list[Path]:
    """只收集 MVTec 异常样本（跳过 good/ 和 ground_truth/ 目录）。"""
    root = Path(data_dir)
    if not root.is_dir():
        sys.exit(f"错误：目录不存在 -- {data_dir}")
    images = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
        and "/good/" not in str(p)
        and "/ground_truth/" not in str(p)
    )
    return images


def parse_mvtc_path(path: Path, data_dir: str) -> dict:
    """从 MVTec 路径提取: category, defect_type, image_id。"""
    root = Path(data_dir)
    rel = path.relative_to(root)
    parts = rel.parts
    # 结构: category/test/defect_type/xxx.png 或 category/test/defect_type/sub/xxx.png
    info = {"category": "", "defect_type": "", "image_id": ""}
    if len(parts) >= 4:
        info["category"] = parts[0]
        info["defect_type"] = parts[2]
        info["image_id"] = str(Path(*parts[3:]))
    elif len(parts) >= 3:
        info["category"] = parts[0]
        info["defect_type"] = parts[2] if len(parts) == 3 else parts[1]
        info["image_id"] = parts[-1]
    return info


def encode_image(image_path: Path) -> str:
    """将本地图像转为 base64 data URI。"""
    ext = image_path.suffix[1:].lower()
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


def _build_concatenated_text(attributes: dict) -> str:
    """将 5 个属性 dict 拼接为适合 T5 编码的自然语言描述。"""
    field_labels = {
        "shape": "The defect shape is",
        "texture": "The surface texture shows",
        "color": "The defect color is",
        "boundary": "The boundary between defect and normal area is",
        "extent": "The extent of the defect is",
    }
    parts = []
    for key, label in field_labels.items():
        value = attributes.get(key, "")
        if value:
            parts.append(f"{label} {value}")
    return " ".join(parts)


def _is_quota_error(error: Exception) -> bool:
    """判断是否为配额/限流错误（应触发模型降级而非中断）。"""
    msg = str(error).lower()
    return any(kw in msg for kw in QUOTA_KEYWORDS)


def call_qwen(client: OpenAI, models: str | list, image_path: Path, data_dir: str) -> dict:
    """调用 Qwen3-VL，返回结构化 5 属性描述 + T5 拼接文本。

    支持 model 降级：传列表时按顺序尝试，配额耗尽自动切到下一个模型。
    """
    mvtc = parse_mvtc_path(image_path, data_dir)
    data_uri = encode_image(image_path)

    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = list(models)

    last_error = None
    for model in model_list:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT},
                    ]
                }],
                temperature=0.3,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()

            # 尝试解析 JSON
            try:
                attributes = json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r'\{[^{}]*"shape"[^{}]*\}', raw, re.DOTALL)
                if match:
                    try:
                        attributes = json.loads(match.group())
                    except json.JSONDecodeError:
                        attributes = {}
                else:
                    attributes = {}

            for k in ("shape", "texture", "color", "boundary", "extent"):
                attributes.setdefault(k, "")

            concatenated = _build_concatenated_text(attributes)

            return {
                "category": mvtc["category"],
                "defect_type": mvtc["defect_type"],
                "image_id": mvtc["image_id"],
                "attributes": attributes,
                "description": concatenated,
                "description_raw": raw,
                "model_used": model,
                "tokens": resp.usage.total_tokens if resp.usage else 0,
            }

        except Exception as e:
            last_error = e
            if _is_quota_error(e) and model != model_list[-1]:
                print(f"\n    ⚠ 模型 {model} 配额/限流，降级到 {model_list[model_list.index(model)+1]} ... ",
                      end="", flush=True)
                time.sleep(2)  # 短暂等待后重试
                continue
            # 非配额错误或已是最后一个模型，直接返回失败
            break

    return {
        "category": mvtc["category"],
        "defect_type": mvtc["defect_type"],
        "image_id": mvtc["image_id"],
        "attributes": {},
        "description": "",
        "description_raw": "",
        "model_used": model_list[0],
        "error": str(last_error) if last_error else "unknown",
        "error_type": "quota" if (last_error and _is_quota_error(last_error)) else "other",
        "tokens": 0,
    }


def load_cache(output_path: str) -> dict:
    """加载已有输出，用 category/defect_type/image_id 做 key。"""
    cache = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for item in existing:
                # 跳过失败的条目，下次运行重试
                if item.get("error") or not item.get("description", "").strip():
                    continue
                key = f"{item['category']}/{item['defect_type']}/{item['image_id']}"
                cache[key] = item
        except (json.JSONDecodeError, KeyError):
            pass
    return cache


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="MVTec 异常样本 VLM 描述")
    parser.add_argument("--data-dir", required=True, help="MVTec 数据集根目录")
    parser.add_argument("--output", default="mvtec_descriptions.json", help="输出 JSON")
    parser.add_argument("--model", default="qwen3-vl-8b-instruct", help="主 Qwen3-VL 模型 ID")
    parser.add_argument("--fallback-model", default="qwen3-vl-8b-thinking",
                        help="主模型配额耗尽后降级的备用模型")
    parser.add_argument("--delay", type=float, default=0.5, help="API 调用间隔(秒)")
    parser.add_argument("--base-url",
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="DashScope API 地址")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张(0=全部)")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续传")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        sys.exit("错误：请先设置环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # 构建模型降级链
    models = [args.model]
    if args.fallback_model and args.fallback_model != args.model:
        models.append(args.fallback_model)
    print(f"模型链: {' → '.join(models)}（配额耗尽自动降级）")

    # 收集异常图像
    images = gather_images(args.data_dir)
    print(f"找到 {len(images)} 张异常图像")
    if args.limit > 0:
        images = images[:args.limit]
        print(f"限制处理前 {args.limit} 张")

    # 断点续传
    results = load_cache(args.output) if args.resume else {}
    todo = []
    for p in images:
        m = parse_mvtc_path(p, args.data_dir)
        key = f"{m['category']}/{m['defect_type']}/{m['image_id']}"
        if key not in results:
            todo.append(p)
    print(f"已缓存 {len(results)} 条，待处理 {len(todo)} 张")

    total_tokens = 0
    for i, img in enumerate(todo, 1):
        m = parse_mvtc_path(img, args.data_dir)
        key = f"{m['category']}/{m['defect_type']}/{m['image_id']}"
        print(f"[{i}/{len(todo)}] {key} ... ", end="", flush=True)

        entry = call_qwen(client, models, img, args.data_dir)
        results[key] = entry
        total_tokens += entry.get("tokens", 0)

        if "error" in entry:
            model_tag = entry.get("model_used", "?")
            err_type = entry.get("error_type", "other")
            if err_type == "quota":
                print(f"⛔ 全部模型配额耗尽 ({model_tag}): {entry['error'][:60]}")
            else:
                print(f"失败 ({model_tag}): {entry['error'][:60]}")
        else:
            model_tag = entry.get("model_used", "?")
            preview = entry["description"][:60].replace("\n", " ")
            print(f"ok [{model_tag}] ({preview}...)")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(list(results.values()), f, ensure_ascii=False, indent=2)

        if args.delay > 0 and i < len(todo):
            time.sleep(args.delay)

    print(f"\n完成！共 {len(results)} 条描述")
    print(f"总 token 数: {total_tokens}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()

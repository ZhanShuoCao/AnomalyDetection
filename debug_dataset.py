#!/usr/bin/env python3
"""
诊断脚本：验证 VLM 描述与 MVTec 图像匹配是否正确，并检查 text encoder 输出。
用法：
    python debug_dataset.py --data-dir /home/czs/Desktop/MVTec \
                            --descriptions ./mvtec_descriptions.json
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np


def check_json(args):
    """检查 VLM 描述 JSON 的完整性"""
    print("=" * 60)
    print("1. VLM 描述 JSON 完整性检查")
    print("=" * 60)

    with open(args.descriptions, "r") as f:
        data = json.load(f)

    total = len(data)
    ok = [d for d in data if d.get("description") and not d.get("error")]
    err = [d for d in data if d.get("error")]
    empty = [d for d in data if not d.get("description") and not d.get("error")]

    print(f"  总条目: {total}")
    print(f"  成功:   {len(ok)}")
    print(f"  失败:   {len(err)}")
    print(f"  空描述: {len(empty)}")

    models = Counter(d.get("model_used", "?") for d in data)
    print(f"  模型分布: {dict(models)}")

    if err:
        print(f"\n  失败条目 ({len(err)} 条):")
        for d in err[:5]:
            print(f"    {d['category']}/{d['defect_type']}/{d['image_id']}: {d.get('error', '?')[:80]}")
        if len(err) > 5:
            print(f"    ... 还有 {len(err)-5} 条")

    # 检查 description 平均长度
    lengths = [len(d["description"].split()) for d in ok]
    print(f"\n  描述长度: min={min(lengths)} words, max={max(lengths)} words, avg={np.mean(lengths):.0f} words")

    # 检查属性完整性
    attr_complete = sum(1 for d in ok if all(d.get("attributes", {}).get(k) for k in ("shape", "texture", "color", "boundary", "extent")))
    print(f"  5属性完整: {attr_complete}/{len(ok)}")

    return data, ok


def check_dataset_matching(args):
    """检查 Dataset 是否能正确匹配 VLM 描述"""
    print("\n" + "=" * 60)
    print("2. Dataset VLM 描述匹配检查")
    print("=" * 60)

    from datasets.mvtec import MVTecDefectDataset

    ds = MVTecDefectDataset(
        root=args.data_dir,
        split="train",
        descriptions_path=args.descriptions,
    )

    total = len(ds)
    vlm_hit = 0
    fallback = 0
    good_samples = 0

    print(f"  Dataset 样本数: {total}")
    print(f"  VLM lookup 条目数: {len(ds._vlm_lookup)}")

    # 按缺陷类型统计匹配率
    type_stats = {}  # {(cat, defect): {"vlm": N, "fallback": N}}

    for i in range(total):
        s = ds[i]
        prompt = s["prompt"]
        cat, defect = s["category"], s["defect_type"]
        key = (cat, defect)

        if key not in type_stats:
            type_stats[key] = {"vlm": 0, "fallback": 0, "total": 0}

        is_vlm = prompt.startswith("The defect shape is")
        if is_vlm:
            vlm_hit += 1
            type_stats[key]["vlm"] += 1
        else:
            if defect == "good":
                good_samples += 1
            else:
                fallback += 1
            type_stats[key]["fallback"] += 1
        type_stats[key]["total"] += 1

    print(f"\n  VLM 命中: {vlm_hit}")
    print(f"  Fallback(手写描述): {fallback}")
    print(f"  Good 样本(无 VLM 描述): {good_samples}")
    print(f"  覆盖率: {vlm_hit}/{total - good_samples} = {vlm_hit/(total-good_samples)*100:.1f}% (排除 good)")

    # 显示 fallback 的类型
    fallback_types = {k: v for k, v in type_stats.items() if v["fallback"] > 0}
    if fallback_types:
        print(f"\n  Fallback 类型 ({len(fallback_types)} 个):")
        for (cat, defect), stats in sorted(fallback_types.items()):
            print(f"    {cat}/{defect}: VLM={stats['vlm']}, fallback={stats['fallback']}")

    # 随机抽 10 条展示
    print(f"\n  随机 10 条样本:")
    indices = random.sample(range(total), min(10, total))
    for i in indices:
        s = ds[i]
        is_vlm = s["prompt"].startswith("The defect shape is")
        tag = "VLM" if is_vlm else "FALLBACK"
        preview = s["prompt"][:130].replace("\n", " ")
        print(f"  [{tag}] {s['category']}/{s['defect_type']}")
        print(f"        {preview}...")
        print()

    return ds


def check_text_encoder(args):
    """检查 T5 text encoder 的 forward 输出是否正常"""
    print("=" * 60)
    print("3. T5 Text Encoder 输出检查")
    print("=" * 60)

    from models.frozen_encoders import FrozenTextEncoder

    encoder = FrozenTextEncoder(
        model_name="google/t5-v1_1-base",
        max_length=512,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    print(f"  设备: {device}")

    # 测试 1: 短文本
    short_texts = ["a photo of a glass bottle with a crack"]
    with torch.no_grad():
        out = encoder(short_texts)
    print(f"  短文本: shape={out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}, "
          f"mean={out.mean().item():.4f}, has_nan={torch.isnan(out).any().item()}")

    # 测试 2: VLM 描述（典型长度）
    vlm_texts = [
        "The defect shape is An irregular fracture with jagged edges and multiple branching cracks. "
        "The surface texture shows Freshly broken glass surface with conchoidal fracture patterns. "
        "The defect color is Bright white primary fracture surfaces with faint rainbow-like interference colors at thin edges. "
        "The boundary between defect and normal area is Abrupt transition from intact smooth glass to fractured void with sharp edge definition. "
        "The extent of the defect is Covers approximately 15-20% of the bottle rim circumference, concentrated at the top-right quadrant."
    ]
    with torch.no_grad():
        out = encoder(vlm_texts)
    print(f"  VLM描述: shape={out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}, "
          f"mean={out.mean().item():.4f}, has_nan={torch.isnan(out).any().item()}")

    # 测试 3: batch 多条
    batch_texts = vlm_texts * 4
    with torch.no_grad():
        out = encoder(batch_texts)
    print(f"  Batch×4: shape={out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}, "
          f"mean={out.mean().item():.4f}, has_nan={torch.isnan(out).any().item()}")

    # 测试 4: 空文本 (CFG)
    with torch.no_grad():
        out = encoder([""])
    print(f"  空文本(CFG): shape={out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}, "
          f"has_nan={torch.isnan(out).any().item()}")

    print(f"\n  ✅ Text encoder 输出正常，无 NaN")

    # 检查数值范围是否适合 fp16
    print(f"\n  fp16 兼容性检查:")
    fp16_max = 65504.0
    abs_max = out.abs().max().item()
    print(f"    输出绝对值最大值: {abs_max:.2f}")
    if abs_max > fp16_max:
        print(f"    ⚠️ 超出 fp16 可表示范围！会导致 NaN")
    elif abs_max > fp16_max * 0.1:
        print(f"    ⚠️ 接近 fp16 上限，self-attention 放大后可能溢出")
    else:
        print(f"    ✅ fp16 安全范围")

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="VLM 描述 + 训练数据诊断")
    parser.add_argument("--data-dir", default="/home/czs/Desktop/MVTec", help="MVTec 根目录")
    parser.add_argument("--descriptions", default="./mvtec_descriptions.json", help="VLM 描述 JSON")
    parser.add_argument("--skip-encoder", action="store_true", help="跳过 T5 encoder 测试")
    args = parser.parse_args()

    if not os.path.exists(args.descriptions):
        sys.exit(f"错误: 描述文件不存在: {args.descriptions}")

    # Step 1: JSON 检查
    data, ok = check_json(args)

    if not ok:
        sys.exit("❌ 没有成功条目，请先跑完 VLM 生成")

    # Step 2: Dataset 匹配
    ds = check_dataset_matching(args)

    # Step 3: Text encoder
    if not args.skip_encoder:
        check_text_encoder(args)

    print("\n" + "=" * 60)
    print("诊断完成")


if __name__ == "__main__":
    main()

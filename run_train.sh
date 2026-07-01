#!/bin/bash
# Single-GPU training launcher for IC-DiT

export CUDA_VISIBLE_DEVICES=0

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    train.py \
    --config configs/mvtec_icdit.yaml

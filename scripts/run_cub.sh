#!/bin/bash

set -e
set -x

cuda_id=0
TES_exp_name="TES_CUB"
V2T_GCD_exp_name="V2T-GCD_CUB_mw_2_lc_1"

exp_id="${TES_exp_name}_${V2T_GCD_exp_name}_$(date +%Y%m%d_%H%M%S)"


python s1_TES.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --cuda_dev $cuda_id \
    --exp_name $TES_exp_name \
    --exp_id $exp_id 


python s2_V2T_GCD.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 2 \
    --cuda_dev $cuda_id \
    --exp_name $V2T_GCD_exp_name\
    --exp_id $exp_id \




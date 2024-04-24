#!/bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

INPUT_SR_IN_K=24
OUTPUT_SR_IN_K=48
INPUT_SR=$(($INPUT_SR_IN_K * 1000))
OUTPUT_SR=$(($OUTPUT_SR_IN_K * 1000))

python train.py \
    --name mdctGAN_${INPUT_SR_IN_K}_to_${OUTPUT_SR_IN_K} \
    --dataroot ./data/train.csv --evalroot ./data/test.csv \
    --lr_sampling_rate $INPUT_SR --sr_sampling_rate $OUTPUT_SR --hr_sampling_rate $OUTPUT_SR \
    --batchSize 20 \
    --gpu_id 0 --fp16 --nThreads 16 --lr 1.5e-4 \
    --arcsinh_transform --abs_spectro --arcsinh_gain 1000 --center \
    --norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
    --netG local --ngf 56 \
    --n_downsample_global 3 --n_blocks_global 4 \
    --n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
    --n_blocks_attn_l 0 --n_blocks_local 3 \
    --fit_residual --upsample_type interpolate --downsample_type resconv \
    --niter 60 --niter_decay 60 --num_D 3 \
    --eval_freq 32000 --save_latest_freq 16000 --save_epoch_freq 10 --display_freq 16000 --tf_log
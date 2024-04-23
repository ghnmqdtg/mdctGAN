#!/bin/bash

# vctk_fintune_G4A3L3_56ngf_2x: 24kHz -> 48kHz
# vctk_fintune_G4A3L3_56ngf_3x: 16kHz -> 48kHz
# vctk_fintune_G4A3L3_56ngf_4x: 12kHz -> 48kHz
# vctk_fintune_G4A3L3_56ngf_6x: 8kHz -> 48kHz

python generate_audio.py \
    --name output_folder_name \
    --load_pretrain ./checkpoints/mdct_2_to_16 \
    --lr_sampling_rate 2000 --sr_sampling_rate 16000 --hr_sampling_rate 16000 \
    --dataroot ./data/wav48/p376/p376_001.wav --batchSize 16 \
    --gpu_id 0 --fp16 --nThreads 1 \
    --arcsinh_transform --abs_spectro --arcsinh_gain 1000 --center \
    --norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
    --netG local --ngf 56 --niter 40 \
    --n_downsample_global 3 --n_blocks_global 4 \
    --n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
    --n_blocks_attn_l 0 --n_blocks_local 3 --gen_overlap 0 \
    --fit_residual --upsample_type interpolate --downsample_type resconv --phase test
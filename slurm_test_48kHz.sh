#!/bin/bash

#Batch Job Paremeters
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=MST112230

OUTPUT_SR_IN_K=48
OUTPUT_SR=$(($OUTPUT_SR_IN_K * 1000))

SAMPLE_RATES=(8 12 16 24)
# Loop over sample rates and run the Python script
for INPUT_SR_IN_K in "${SAMPLE_RATES[@]}"
do
    INPUT_SR=$(($INPUT_SR_IN_K * 1000))

    if [ $INPUT_SR -eq 8000 ]; then
        CHECKPOINTS=vctk_fintune_G4A3L3_56ngf_6x
    elif [ $INPUT_SR -eq 12000 ]; then
        CHECKPOINTS=vctk_fintune_G4A3L3_56ngf_4x
    elif [ $INPUT_SR -eq 16000 ]; then
        CHECKPOINTS=vctk_fintune_G4A3L3_56ngf_3x
    elif [ $INPUT_SR -eq 24000 ]; then
        CHECKPOINTS=vctk_fintune_G4A3L3_56ngf_2x
    fi
    
    python test.py \
        --name mdctGAN/${CHECKPOINTS} \
        --load_pretrain mdctGAN/${CHECKPOINTS} \
        --dataroot ./data/test_org_filtered.csv --batchSize 1 \
        --lr_sampling_rate $INPUT_SR --sr_sampling_rate $OUTPUT_SR --hr_sampling_rate $OUTPUT_SR \
        --gpu_id 0 --fp16 --nThreads 1 \
        --arcsinh_transform --abs_spectro --arcsinh_gain 1000 --center \
        --norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
        --netG local --ngf 56 --niter 40 \
        --n_downsample_global 3 --n_blocks_global 4 \
        --n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
        --n_blocks_attn_l 0 --n_blocks_local 3 --gen_overlap 0 \
        --fit_residual --upsample_type interpolate --downsample_type resconv --phase test
done
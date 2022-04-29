cd ../evit/

# Train from scratch as RR using the random mask
# Keep Rate 0.95

data_path=${1:-"/home/ImageNet"}
save_path="../checkpoints/exp-deit-base-dim576-keeprate0.95-RR"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_base_patch16_shrink_base_dim576 \
        --random \
        --base_keep_rate 0.95 \
        --input-size 224 \
        --batch-size 128 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
> ../scripts/exp-deit-base-dim576-keeprate0.95-RR.txt 2>&1 &

cd ../deit/

# Train from scratch as RR using the random mask
# Keep Rate 0.75

data_path=${1:-"/mnt/dataset/imagenet"}
save_path="../checkpoints/exp-deit-tiny-keeprate0.75-RR"
log_path="../logs"
mkdir -p $save_path
mkdir -p $log_path


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_tiny_patch16_shrink_base \
        --random \
        --base_keep_rate 0.75 \
        --input-size 224 \
        --batch-size 128 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
2>&1 | tee ../logs/exp-deit-tiny-keeprate0.75-RR.txt

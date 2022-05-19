cd ../deit/

# Train from scratch as LTH using the DeiT Tiny as "teacher"
# Keep Rate 0.85

data_path="/home/ImageNet"
save_path="../checkpoints/exp-deit-small-keeprate0.85-LTH-tiny-teacher"
mkdir -p $save_path


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_small_patch16_shrink_base \
        --lottery ../checkpoints/exp-deit-tiny-keeprate0.85-LTH/best_checkpoint.pth \
        --lottery-model-type deit_tiny_patch16_shrink_base \
        --base_keep_rate 0.85 \
        --input-size 224 \
        --batch-size 128 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
> ../scripts/exp-deit-small-keeprate0.85-LTH-tiny-teacher.txt 2>&1 &

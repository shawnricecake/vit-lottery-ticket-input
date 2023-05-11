cd ../deit/

save_path="../checkpoints/exp-deit-small-cifar10-keeprate0.95-RR"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_small_patch16_shrink_base \
        --random \
        --base_keep_rate 0.95 \
        --input-size 224 \
        --batch-size 32 \
        --shrink_start_epoch 0 \
        --epochs 300 \
        --dist-eval \
        --data-set "CIFAR10" \
        --data-path "../datasets/" \
        --output_dir ${save_path} \
> ../scripts/exp-deit-small-cifar10-keeprate0.95-RR.log 2>&1 &

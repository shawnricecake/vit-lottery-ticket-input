cd ../deit/

save_path="../checkpoints/exp-deit-small-cifar10-keeprate0.9-LTH"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_small_patch16_shrink_base \
        --lottery ../checkpoints/exp-deit-small-cifar10-pretrain/best_checkpoint.pth \
        --base_keep_rate 0.9 \
        --input-size 224 \
        --batch-size 32 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-set "CIFAR10" \
        --data-path "../datasets/" \
        --output_dir ${save_path} \
> ../scripts/exp-deit-small-cifar10-keeprate0.9-LTH.log 2>&1 &

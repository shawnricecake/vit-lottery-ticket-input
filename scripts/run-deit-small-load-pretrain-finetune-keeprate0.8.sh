cd ../evit/

# Train as EViT, load pretrain model and finetune
# Keep Rate 0.8

data_path="/home/ImageNet"
save_path="../checkpoints/exp-deit-small-load-pretrain-finetune-keeprate0.8"
mkdir -p $save_path


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_small_patch16_shrink_base \
        --fuse_token \
        --base_keep_rate 0.8 \
        --input-size 224 \
        --sched cosine \
        --lr 2e-5 \
        --min-lr 2e-6 \
        --weight-decay 1e-6 \
        --batch-size 256 \
        --shrink_start_epoch 0 \
        --warmup-epochs 0 \
        --shrink_epochs 0 \
        --epochs 30 \
        --dist-eval \
        --finetune ../checkpoints/exp-deit-small-pretrain/checkpoint.pth \
        --data-path ${data_path} \
        --output_dir ${save_path} \
> ../scripts/exp-deit-small-load-pretrain-finetune-keeprate0.8.txt 2>&1 &

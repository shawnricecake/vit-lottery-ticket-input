cd ../deit/

# Train as EViT, load pretrain model and finetune
# Keep Rate 0.9

data_path=${1:-"/mnt/dataset/imagenet"}
save_path="../checkpoints/exp-deit-base-keeprate0.9-load-pretrain-finetune"
log_path="../logs"
mkdir -p $save_path
mkdir -p $log_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_base_patch16_shrink_base \
        --base_keep_rate 0.9 \
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
        --finetune https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth \
        --data-path ${data_path} \
        --output_dir ${save_path} \
2>&1 | tee ../logs/exp-deit-base-keeprate0.9-load-pretrain-finetune.txt

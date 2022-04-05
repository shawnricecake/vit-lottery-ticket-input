cd ../evit/

# Train from scratch as LTH using the pretrained EViT model as "teacher"
# Keep Rate 0.9

data_path="/home/ImageNet"
save_path="../checkpoints/exp-deit-small-keeprate0.9-LTH"
mkdir -p $save_path


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_small_patch16_shrink_base \
        --fuse_token \
        --adjust-keep-rate \
        --lottery ../checkpoints/exp-deit-small-keeprate0.9-load-pretrain-finetune/best_checkpoint.pth \
        --base_keep_rate 0.9 \
        --input-size 224 \
        --batch-size 128 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
        --seed 1 \
> ../scripts/exp-deit-small-keeprate0.9-LTH.txt 2>&1 &

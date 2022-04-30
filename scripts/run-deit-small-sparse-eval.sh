cd ../evit/

data_path="/home/ImageNet"
save_path="../checkpoints/exp-deit-sparse-eval-temp"
mkdir -p $save_path

# LTH Test
# --base_keep_rate ...
# --lottery ...
# --resume ...

# Random Test
# --base_keep_rate ...
# --random
# --resume ...

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12345 --use_env main.py \
        --model deit_tiny_patch16_shrink_base \
        --base_keep_rate 0.85 \
        --lottery ... \
        --resume ... \
        --eval \
        --sparse-eval \
        --input-size 224 \
        --batch-size 128 \
        --warmup-epochs 5 \
        --shrink_start_epoch 10 \
        --shrink_epochs 100 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
> ../scripts/exp-deit-sparse-eval-temp.txt 2>&1 &

cd ../deit/

data_path=${1:-"/home/ImageNet"}
save_path="../checkpoints/exp-deit-base-dim576-pretrain"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        --use_env main.py \
        --model deit_base_patch16_shrink_base_dim576 \
        --base_keep_rate 1 \
        --input-size 224 \
        --batch-size 128 \
        --epochs 300 \
        --dist-eval \
        --data-path ${data_path} \
        --output_dir ${save_path} \
> ../scripts/exp-deit-base-dim576-pretrain.txt 2>&1 &

cd ../lvvit/

data_path="/home/ImageNet"
save_path="../checkpoints/exp-lvvit-small-keeprate0.85-RR"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --model lvvit_small_sparse \
        --batch-size 128 \
        --epochs 300 \
        --apex-amp \
        --img-size 224 \
        --drop-path 0.1 \
        --model-ema \
        --data_dir ${data_path} \
        --output ${save_path} \
> ../scripts/exp-lvvit-small-keeprate0.85-RR.txt 2>&1 &

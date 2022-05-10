cd ../lvvit/

label_data="path/to/label_data"
data_path="/home/ImageNet"
save_path="../checkpoints/exp-lvvit-small-sparse-eval"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --model lvvit_small_sparse \
        --base_keep_rate 0.85 \
        --lottery ../checkponts/exp-lvvit-small-keeprate0.85-load-pretrain-finetune/checkpoint.pth \
        --lottery-model-type lvvit_small_sparse \
        --resume ../checkponts/exp-lvvit-small-keeprate0.85-LTH/best_checkpoint.pth \
        --sparse-eval \
        --batch-size 128 \
        --epochs 300 \
        --apex-amp \
        --img-size 224 \
        --drop-path 0.1 \
        --token-label \
        --token-label-data ${label_data} \
        --token-label-size 14 \
        --model-ema \
        --data_dir ${data_path} \
        --output ${save_path} \
> ../scripts/exp-lvvit-small-sparse-eval.txt 2>&1 &

cd ../lvvit/

label_data="path/to/label_data"
data_path="/home/ImageNet"
save_path="../checkpoints/exp-lvvit-small-keeprate0.85-load-pretrain-finetune"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --model lvvit_small_sparse \
        --base_keep_rate 0.85 \
        --batch-size 256 \
        --epochs 30 \
        --finetune https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar \
        --lr 5e-6 \
        --min-lr 5e-6 \
        --weight-decay 1e-8 \
        --apex-amp \
        --img-size 224 \
        --drop-path 0.1 \
        --token-label \
        --token-label-data ${label_data} \
        --token-label-size 14 \
        --model-ema \
        --data_dir ${data_path} \
        --output ${save_path} \
> ../scripts/exp-lvvit-keeprate0.85-load-pretrain-finetune.txt 2>&1 &

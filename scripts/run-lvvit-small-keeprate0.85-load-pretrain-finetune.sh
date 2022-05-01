cd ../lvvit/

data_path="/home/ImageNet"
save_path="../checkpoints/exp-lvvit-small-keeprate0.85-load-pretrain-finetune"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --model lvvit_small_sparse \
        --batch-size 256 \
        --epochs 30 \
        --finetune https://github.com/zihangJiang/TokenLabeling/releases/download/1.0/lvvit_s-26M-224-83.3.pth.tar \
        --lr 6.4e-5 \
        --min-lr 2e-6 \
        --weight-decay 1e-6 \
        --apex-amp \
        --img-size 224 \
        --drop-path 0.1 \
        --model-ema \
        --data_dir ${data_path} \
        --output ${save_path} \
> ../scripts/exp-deit-lvvit-keeprate0.85-load-pretrain-finetune.txt 2>&1 &

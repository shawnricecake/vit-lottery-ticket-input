cd ../swin/

data_path=${1:-"/home/ImageNet"}
save_path="../checkpoints/exp-swin-tiny-sparse-eval"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --cfg configs/swin_tiny_patch4_window7_224.yaml \
        --sparse-eval-with-zero \
        --resume ../checkpoints/exp-swin-tiny-keeprate0.9-LTH-zero-deit-small-keeprate0.9-teacher/swin_tiny_patch4_window7_224/default/best_checkpoint.pth \
        --eval \
        --batch-size 128 \
        --epochs 300 \
        --amp-opt-level O1 \
        --output ${save_path} \
        --data-path ${data_path} \
> ../scripts/exp-swin-tiny-sparse-eval.txt 2>&1 &

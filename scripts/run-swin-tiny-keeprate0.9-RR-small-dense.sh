cd ../swin/

# you should change the "small-dense-input-size" that in computed according to the keep rate
# and the "small-dense-patch-num-one-side" is computed by "small-dense-input-size" / patch size

# the config file should be changed according to the resulting "small-dense-input-size"
# because the window size should be suitable for the "small-dense-input-size" i.e., can be divided

data_path=${1:-"/home/ImageNet"}
save_path="../checkpoints/exp-swin-tiny-keeprate0.9-RR-small-dense"
mkdir -p $save_path

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 \
        main.py \
        --cfg configs/swin_tiny_patch4_window6_224.yaml \
        --random \
        --lottery-model-type deit_small_patch16_shrink_base \
        --base_keep_rate 0.9 \
        --small-dense-input \
        --small-dense-input-size 192 \
        --small-dense-patch-num-one-side 12 \
        --batch-size 128 \
        --epochs 300 \
        --amp-opt-level O1 \
        --output ${save_path} \
        --data-path ${data_path} \
> ../scripts/exp-swin-tiny-keeprate0.9-RR-small-dense.txt 2>&1 &

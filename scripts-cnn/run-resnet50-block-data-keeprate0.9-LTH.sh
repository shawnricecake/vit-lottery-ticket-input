#!/bin/bash

cd ../cnn/

seed=914
dataset="imagenet"
epochs=90
batch_size_per_gpu=128
lr=0.2048
keep_rate=0.9

name=resnet50-${dataset}
save_dir=../checkpoints/${name}-${epochs}epochs-bs${batch_size_per_gpu}-lr${lr}/keeprate${keep_rate}-block-data-LTH/
mkdir -p ${save_dir}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
    train_image_classification_imagenet_timm.py \
    --model resnet50 \
    --seed ${seed} \
    --lr ${lr} \
    --epochs ${epochs} \
    --batch-size ${batch_size_per_gpu} \
    --output ${save_dir} \
    --data_dir /home/ImageNet \
    --dataset ImageNet \
    --amp \
    --resume ../checkpoints-pretrain/${name}-${epochs}epochs-bs${batch_size_per_gpu}/pretrain/initial-model.pth \
    --lottery path_to_lottery_model \
    --base_keep_rate ${keep_rate} \
> ../scripts-cnn/output-${name}-${epochs}epochs-lr${lr}-bs${batch_size_per_gpu}-keeprate${keep_rate}-block-data-LTH.log 2>&1 &


# comment deit/vision_transformer.py 790th row
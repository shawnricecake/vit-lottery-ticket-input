#!/bin/bash

cd ../cnn/

seed=914
dataset="imagenet"
epochs=90
batch_size_per_gpu=128
lr=0.2048


name=resnet50-${dataset}
save_dir=../checkpoints/${name}-${epochs}epochs-bs${batch_size_per_gpu}-lr${lr}/pretrain/
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
> ../scripts-cnn/output-${name}-${epochs}epochs-bs${batch_size_per_gpu}-pretrain.log 2>&1 &










#CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
#python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
#    main.py \
#    --model resnet50 \
#    --seed 914 \
#    --lr 0.2048 \
#    --epochs 90 \
#    --batch-size 128 \
#    --output checkpoints \
#    --data_dir /home/ImageNet \
#    --dataset ImageNet \
#    --amp
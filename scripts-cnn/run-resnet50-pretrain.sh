#!/bin/bash

cd ../cnn/

seed=1
dataset="imagenet"
num_classes=1000
input_resolution=224
epochs=120
batch_size_per_gpu=96

name=resnet50-${dataset}
save_dir=../checkpoints/${name}-${epochs}epochs-bs${batch_size_per_gpu}/pretrain/
mkdir -p ${save_dir}

CUDA_VISIBLE_DEVICES="0" \
nohup \
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env \
  main_only_resnet50_pretrain.py \
  --dataset ${dataset} --num_classes ${num_classes} \
  --dist_mode distribute --workers_per_gpu 6 \
  --input_image_size ${input_resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --batch_size_per_gpu ${batch_size_per_gpu} \
  --seed ${seed} \
  --save_dir ${save_dir} \
> ../scripts-cnn/output-${name}-${epochs}epochs-bs${batch_size_per_gpu}-pretrain.log 2>&1 &


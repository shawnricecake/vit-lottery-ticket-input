# Input Level Lottery Ticket



## Preparation
### DeiT
```
torch==1.9.0
torchvision==0.10.0
timm==0.4.12
tensorboardX==2.4
torchprofile==0.0.4
lmdb==1.2.1
pyarrow==5.0.0
```

### Swin
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

opencv-python==4.4.0.46 
termcolor==1.1.0 
yacs==0.1.8
```

## Training
Replace the **data_path** in shell file with the **ImageNet** dataset path

Download the pretrained model from official github of [**DeiT**](https://github.com/facebookresearch/deit) or [**Swin**](https://github.com/microsoft/Swin-Transformer)

### DeiT
Normal Training Sequence:
```
bash run-deit-small-keeprate0.xx-load-pretrain-fineture.sh
bash run-deit-small-keeprate0.xx-LTH.sh
bash run-deit-small-keeprate0.xx-RR.sh
```

You can revise the keep rate as you want:
```
1. revise the shell file name: "-keeprate0.xx-"
2. revise the save_path in shell file: "exp-deit-small-keeprate0.xx"
3. revise the --lottery at "-keeprate0.xx-"
4. revise the --base_keep_rate 0.xx
5. revise the output txt file name at the last row: "-keeprate0.xx-"
(6.) revise the epoch when training the "RR": --epochs xxx
```

### DeiT as teacher to train Swin
```
bash run-swin-tiny-keeprate0.xxx-LTH.sh
```


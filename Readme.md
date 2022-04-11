# Input Level Lottery Ticket



## Preparation
```
torch==1.9.0
torchvision==0.10.0
timm==0.4.12
tensorboardX==2.4
torchprofile==0.0.4
lmdb==1.2.1
pyarrow==5.0.0
```

## Training
Replace the **data_path** in shell file with the **ImageNet** dataset path

Download the pretrained model from me

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
```


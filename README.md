# RHNet

This repo hosts the implementation for our paper RHNet.


### Installation
- This implementation is based on [MMRotate](https://github.com/open-mmlab/mmrotate). Therefore the installation is the same as original MMRotate.

- Please check [get_started.md](https://github.com/open-mmlab/mmrotate/blob/main/docs/en/get_started.md) for installation.

## Training

The following command line will train `rhnet_r50_dota_train_val_3x` on 4 GPUs:

```
bash dist_train.sh configs/RHNet/rhnet_r50_dota_train_val_3x.py 4
```
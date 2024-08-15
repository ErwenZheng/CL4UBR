# Cross-View and Spatial-Temporal Contrastive Learning for User Behavior Representation in Online Food Ordering Services
This code is the official implementation of "Cross-View and Spatial-Temporal Contrastive Learning for User Behavior Representation in Online Food Ordering Services".

## Requirements
- python >= 3.6
- PyTorch >=1.8

We give three datasets (Sanya, Taiyuan and Yinchuan ).

## Dataset link
OneDrive Disk:
Please download the dataset to the directory ./data

## Training(CTR)
* You can train our CLUBR by run:
```bash
python train.py --config config/model_sanya_ctr.conf
```

## Training(TP)
* You can train our CLUBR by run:
```bash
python train.py --config config/model_sanya_tp.conf
```

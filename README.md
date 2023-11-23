# FAS-DualStream
Implementation of paper [Bandpass Filter Based Dual-stream Network for Face Anti-spoofing (3rd FAS CVPR2023)](https://openaccess.thecvf.com/content/CVPR2023W/FAS/papers/Lu_Bandpass_Filter_Based_Dual-Stream_Network_for_Face_Anti-Spoofing_CVPRW_2023_paper.pdf)
## Dataset
- Related to [FAS Wild CVPRW23](https://github.com/deepinsight/insightface/tree/master/challenges/cvpr23-fas-wild)
## Training
- Use R18 as backbone
```
python train.py --train_txt fas_train.txt --val_txt fas_valid.txt --batch-size 16 --save r18-imagenet --lr 0.0001 --epochs 10 --fp16
```
## Result
| Backbone                | ACER      | APCER     | BPCER     |
|-------------------------|-----------|-----------|-----------|
| R18 (Imagenet)          | 7,319     | **3,641** | 10,997    |
| R18-DualBand (Imagenet) | **5,885** | 4,840     | **6,833** |

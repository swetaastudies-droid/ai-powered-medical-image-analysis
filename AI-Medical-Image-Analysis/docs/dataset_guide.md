# Dataset guide
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Required layout: `data/raw/chest_xray/train/{NORMAL,PNEUMONIA}/` and `data/raw/chest_xray/test/{NORMAL,PNEUMONIA}/`. Original `val` is not used because it is small. The code creates a stratified image-level 20% validation split from `train`. Patient-level separation cannot be confirmed from the folder layout. Keep test images separate until final evaluation. Review source terms and attribution.

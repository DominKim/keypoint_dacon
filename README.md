# keypoint_dacon
- [Dacon의 모션 키포인트 검출 AI 경진대회](https://dacon.io/competitions/official/235701/overview/description/)
- 최종 : 22 / 512

## Dataset:

- 파일 및 폴더 구조
``` python3
├── data
│   ├── Keypoint
│       ├── train_imgs
│       ├── test_imgs
│──── trainer.py
│   ├── train.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── model_loader.py
```

## Model & Data:


### data_loader
- keypoint rcnn model을 수행하기 위해 이미지를 224, 244로 리사이즈를 함

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
│   ├── trainer.py
│   ├── train.py
│   ├── data_loader.py
│   ├── utils.py
│   ├── model_loader.py


## Model & Data:
### 1_Cralwer
- Selenium 패키지를 활용하여 youtube 썸네일 이미지를 저장하고, pickle 형식으로 이미지외 데이터를 저장

### data_loader
- resnet model을 수행하기 위해 이미지를 224, 244로 리사이즈를 하였고 y(종속변수)를 파이토치의 텐서 형식으로 바꾸고 파이토치 Dataloader를 이용해 train, valid, test set의로 나눔

### Image_similrity
- resnet 모델의 미리 학습된 avgpool layer를 가져와 이미지를 layer에 input하고 zero행렬에 embedding을 통해 이미지의 feature를 추출하고 코사인 유사도를 통해 가장 유사도가 높은 이미지를 추천

### model_loader
- 전이학습(transfer_learning)을 통해 회귀분석을 하기 위해 미리 학습된 resnet34(use_traine = True) 모델을 불러오고 학습된 파라미터들(weights)은 freeze 시키고 모델의 fc layer를 nn.Linear(n_features, config.n_classes)과 같이 변경하고 n_classes는 1로 설정(회귀분석)함.

## Why?
### Transfer Learing
- 전이학습을 활용한 이유는 여러 논문과 연구에서 나왔듯이 대량의 이미지를 미리 학습한 모델은 이미지의 특징(feature)을 추출하는 적절한 파라미터를 가지고 있기때문에 전이학습을 활용

- 또한, 위와같은 이유로 이미지 유사도를 추출하기 위해서 renset의 미리 학습된 layer를 활용

## Results 

| Hyperparameter| Choosen Value |
| -------------   | -------------      |
| Loss Function | Mean Sqaured Error	|
| Batch Size | 50	|
| n_Epoch | 10	|

- The corresponding results in our best model is given below, 

| Loss Type       | Mean squared Error |
| -------------   | -------------      |
| Validation Loss | 91.8	       |
| Train Loss 	  | 14.19	       |

- 컴퓨터 사양의 문제로 적은 데이터셋과 Epoch으로 학습을 수행

## Reference

- [Effortlessly Recommending Similar Images](https://towardsdatascience.com/effortlessly-recommending-similar-images-b65aff6aabfb)
- [TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

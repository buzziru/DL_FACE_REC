# DL_FACE_REC
## 1. 프로젝트 주제
  - 얼굴 이미지로 나이, 성별, 감정 추정하기
## 2. DATA
#### 1) 안면 인식 에이징 이미지 데이터
  - 활용 모델: 나이 / 성별 예측 모델
  - 용량: 76.84GB
  - 출처: [AI허브](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71415)
  - 특징: 한 인물의 유아기부터 현재까지의 연령대별 안면 이미지와 해당 이미지 내의 안면 위치, 랜드마크 등을 라벨링하여 동일 인물의 시간의 경과에 따른 노화로 발생하는 안면 변화 정보를 담고 있는 안면 인식용 인공지능 데이터
  - train 이미지: 40,150장
  - test 이미지: 5,050장
#### 2) FER-2013
  - 활용 모델: 감정 예측 모델
  - 용량: 56MB
  - 출처: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data?select=test)
  - 클래스(7개): angry, disgust, fear, happy, neutral, sad, surprise
  - train 이미지: 28,709장
  - test 이미지: 7,178장
## 3. 작업흐름도
  ![image](https://github.com/buzziru/DL_FACE_REC/assets/152848901/85286003-3953-4e6d-a86d-6cea704497de)
## 4. 모델 소개 및 성능 평가
#### 1) 얼굴 검출 모델
  - RetinaFace: 단일 통합 네트워크를 사용하며, 다양한 크기의 얼굴을 찾아낼 수 있고, 기울어진 얼굴의 각도를 수직으로 수정할 수 있습니다.
#### 2) 나이 예측 모델
  - Inception ResNet v1 모델을 사용하였습니다.
  - Total Params : 23,483,137  
![resnet](https://github.com/buzziru/DL_FACE_REC/assets/152848901/1c297dea-2d67-4bf5-86f6-06686186745c)
  > 성능 평가
  - 손실 함수: MAE  
  ![loss1](https://github.com/buzziru/DL_FACE_REC/assets/152848901/89a9e0f0-8c40-4511-b17b-5961f377ae10)
#### 3) 성별 예측 모델
  - 성별 모델은 기존 CNN 모델을 사용하였습니다. 기본적인 CNN 모델에서 각 층의 출력을 정규화해주는 배치정규화를 추가하여 학습을 안정화하고 속도를 높여주었습니다.
  모델 학습 과정에서 훈련 epoch를 30까지 진행해보았을 때 validation 데이터의 정확도와 로스가 epoch 19 이후로 좋아지지 않는 형태를 보여서 이전에 가장 좋았던 epoch 19의 결과로 모델을 선정하였습니다.
  - Total Params : 5,763,905
  > 성능 평가
  - Training and Validation Loss and Accuracy over epochs  
![성별성능](https://github.com/buzziru/DL_FACE_REC/assets/152848901/cabd21fb-2fb7-4d1e-a129-6b45b20a1a40)
#### 4) 감정 예측 모델
  - 감정 모델도 CNN 모델을 사용했습니다. 다층 CNN을 사용하여 이미지 분류를 수행했으며, 모델의 일반화 성능을 향상시키기 위해 배치 정규화와 드롭아웃을 사용했습니다.
  - Total Params : 4,496,390
  - 또한, 데이터셋의 불균형 데이터 존재로 "소수 클래스(disgust) 삭제" 후 모델 생성하였습니다.  
![불균형](https://github.com/buzziru/DL_FACE_REC/assets/152848901/6888ff97-4aca-4c44-bb19-b9bbca2a989d)
  > 성능 평가
  - Training and Validation Loss and Accuracy over epochs
  <img width="235" alt="감정1" src="https://github.com/buzziru/DL_FACE_REC/assets/152848901/21609cc5-5fa3-47b3-a532-4c0d6a56cb10">
  <img width="237" alt="감정2" src="https://github.com/buzziru/DL_FACE_REC/assets/152848901/d5da3ed0-eb9d-4b1f-b01f-058096476b74">  

## 5. 결과 예측
#### 1) 이미지  
![1](https://github.com/buzziru/DL_FACE_REC/assets/152848901/b3b86569-113d-4830-8e7b-b9f9dd3a897c)  
![2](https://github.com/buzziru/DL_FACE_REC/assets/152848901/2c635285-4a29-4b2d-8b38-38cc156279d6)  
![3](https://github.com/buzziru/DL_FACE_REC/assets/152848901/3b9c2eea-037a-4b91-8790-f354be7cb5bf)  
![4](https://github.com/buzziru/DL_FACE_REC/assets/152848901/5af51417-0161-4a8f-9092-dca538cd41db)  
![5](https://github.com/buzziru/DL_FACE_REC/assets/152848901/5911cdc0-436f-47f9-8555-14f7f1e88fd0)  
![6](https://github.com/buzziru/DL_FACE_REC/assets/152848901/2ddff2f8-0167-4045-ae06-04883024a58f)  
![7](https://github.com/buzziru/DL_FACE_REC/assets/152848901/6f86377c-f2c5-415b-bfb9-934da029e813)  
![8](https://github.com/buzziru/DL_FACE_REC/assets/152848901/453e8c4c-9238-4124-a28b-b2dcbb22fcb6)  
![9](https://github.com/buzziru/DL_FACE_REC/assets/152848901/b7af4526-c79c-46a1-b15f-765d028386db)  
#### 2) 영상
  - [링크 바로가기](http://www.youtube.com/watch?v=fwqFS8A_Kf8)
#### 3) STREAMLIT 링크
  - [링크 바로가기](https://dl-face-rec.streamlit.app/)
#### 4) Tkinter  
![image](https://github.com/buzziru/DL_FACE_REC/assets/152848901/2aa38017-e19b-4536-969b-fecb0c1a2436)












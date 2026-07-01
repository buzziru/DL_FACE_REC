---
title: Face Age Gender Emotion API
emoji: 🧑‍🔬
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Face Age · Gender · Emotion — compute API

RetinaFace 얼굴 검출(uniface ONNX) 후, 검출된 얼굴마다 나이(회귀)·성별·감정(6클래스)을
동시 추정하는 compute space. 세 분류기는 원본 프로젝트([DL_FACE_REC](https://github.com/buzziru/DL_FACE_REC))의
PyTorch/Keras 모델을 ONNX로 변환해 onnxruntime(CPU)로 추론합니다.

정적 데모 [ingyoun/face-rec-demo](https://huggingface.co/spaces/ingyoun/face-rec-demo)가
업로드 이미지에 대해 이 space를 `@gradio/client`로 호출합니다.

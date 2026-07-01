---
title: Face Age Gender Emotion Demo
emoji: 🧑
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
---

# 얼굴 나이·성별·감정 추정 — 데모

[DL_FACE_REC](https://github.com/buzziru/DL_FACE_REC) 프로젝트의 정적 인터랙티브 데모.

- **샘플**: 6개 이미지에 대한 추론 결과를 사전 계산해 내장(서버 호출 없음).
- **업로드**: compute space [ingyoun/face-rec-api](https://huggingface.co/spaces/ingyoun/face-rec-api)를
  `@gradio/client`로 호출해 실시간 추론(RetinaFace ONNX 검출 + 나이·성별·감정 ONNX 분류).

포트폴리오 상세페이지에 iframe으로 임베드되며 `face-demo-height` postMessage로 높이를 동기화합니다.

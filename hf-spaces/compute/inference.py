"""Face age/gender/emotion pipeline over ONNX models.

Detection: RetinaFace (uniface, ONNX). Classifiers: the project's own models
converted to ONNX (age regression, gender CNN, emotion CNN). Preprocessing
mirrors the original PyTorch/Keras pipeline (output/tkinter_result.py) so the
predictions match the trained models.
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
from uniface import RetinaFace

# age/gender share the same crop + normalization (torchvision Normalize)
_MEAN = np.array([0.6284, 0.4901, 0.4325], np.float32)
_STD = np.array([0.1869, 0.1712, 0.1561], np.float32)

# emotion model trained on FER-2013 minus the sparse 'disgust' class -> 6 classes
EMOTION_LABELS = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

_HERE = os.path.dirname(os.path.abspath(__file__))


class FacePipeline:
    def __init__(self, models_dir=None):
        d = models_dir or os.path.join(_HERE, "models")
        prov = ["CPUExecutionProvider"]
        self.det = RetinaFace(confidence_threshold=0.6, providers=prov)
        self.age = ort.InferenceSession(os.path.join(d, "age.onnx"), providers=prov)
        self.gender = ort.InferenceSession(os.path.join(d, "gender.onnx"), providers=prov)
        self.emotion = ort.InferenceSession(os.path.join(d, "emotion.onnx"), providers=prov)
        self._age_in = self.age.get_inputs()[0].name
        self._gender_in = self.gender.get_inputs()[0].name
        self._emo_in = self.emotion.get_inputs()[0].name

    def _prep_rgb(self, rgb, box):
        x1, y1, x2, y2 = box
        crop = rgb[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
        crop = (crop - _MEAN) / _STD
        return np.transpose(crop, (2, 0, 1))[None].astype(np.float32)

    def _prep_emotion(self, rgb, box):
        x1, y1, x2, y2 = box
        crop = rgb[y1:y2, x1:x2]
        # original applied COLOR_BGR2GRAY to an RGB array — replicate for parity
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
        return gray[None, :, :, None]

    def predict(self, rgb):
        """rgb: HxWx3 uint8 RGB array. Returns list of dicts per face."""
        faces = self.det.detect(rgb)
        h, w = rgb.shape[:2]
        out = []
        for f in faces:
            x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            box = (x1, y1, x2, y2)
            age = float(self.age.run(None, {self._age_in: self._prep_rgb(rgb, box)})[0].ravel()[0])
            logit = float(self.gender.run(None, {self._gender_in: self._prep_rgb(rgb, box)})[0].ravel()[0])
            female_prob = 1.0 / (1.0 + np.exp(-logit))
            gender = "Male" if female_prob < 0.5 else "Female"
            emo = self.emotion.run(None, {self._emo_in: self._prep_emotion(rgb, box)})[0].ravel()
            emotion = EMOTION_LABELS[int(np.argmax(emo))]
            out.append({
                "box": box,
                "age": round(age),
                "gender": gender,
                "female_prob": round(female_prob, 3),
                "emotion": emotion,
            })
        return out


# blue for male, red/pink for female (BGR for cv2 drawing)
_MALE_BGR = (255, 120, 40)
_FEMALE_BGR = (90, 90, 240)


def draw(rgb, results):
    """Draw boxes + labels on a copy of rgb (returns RGB uint8)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for r in results:
        x1, y1, x2, y2 = r["box"]
        col = _MALE_BGR if r["gender"] == "Male" else _FEMALE_BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f'{r["age"]} {r["gender"]} {r["emotion"]}'
        fs = max(0.4, min(0.8, (x2 - x1) / 200))
        th = max(1, int(fs * 2))
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ytop = max(0, y1 - tht - 6)
        cv2.rectangle(img, (x1, ytop), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

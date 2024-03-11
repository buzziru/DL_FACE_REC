import tkinter as tk
from tkinter import font
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2 
from facenet_pytorch import InceptionResnetV1
from retinaface import RetinaFace
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from keras import regularizers
from keras.models import load_model
from retinaface import RetinaFace

class Gender_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        x = x.squeeze()
        return x

def construct_age_model():
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=1)
    in_dim = model.logits.in_features
    model.logits = nn.Linear(in_dim, 1)
    return model

def load_model(model, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def make_model():
    age = construct_age_model()
    age_path = 'D:/lsh/DL_Team_Project/model/age_model_checkpoint_epoch_10_loss_4.57.pth'
    
    gender = Gender_Net()
    gender_path = 'D:/lsh/DL_Team_Project/model/gender_model_v3_checkpoint_epoch_19.pth'
    
    age_model = load_model(age, age_path)
    gender_model = load_model(gender, gender_path) 

    return age_model, gender_model

def process_input(img, box):
    img_ar = img[box[1]:box[3], box[0]:box[2]]
    img_ar = cv2.resize(img_ar, (224, 224))
    image = Image.fromarray(img_ar)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6284, 0.4901, 0.4325], std=[0.1869, 0.1712, 0.1561]),
    ])
    img_tensor = preprocess(image).unsqueeze(0)
    return img_tensor

def emotion_process_input(img, box):
  img_ar = img[box[1]:box[3], box[0]:box[2]]

  img_gray = cv2.cvtColor(img_ar, cv2.COLOR_BGR2GRAY)
  img_gray = cv2.resize(img_gray, (48, 48))
  image = Image.fromarray(img_gray)
  image = np.expand_dims(image, axis=0)
  img_tensor = image.astype(np.float32) / 255.0
  return img_tensor

def predict_age(image, model):
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(image)

    return prediction.item()

def predict_gender(image, model):
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        logit = model(image)
        pred = torch.sigmoid(logit)
        output = 'Male' if pred.item() < 0.5 else 'Female'

    return output

def predict_emotion(image, model):
  label_dict = {0:'Angry', 1:'Fear', 2:'Happy', 3:'Neutral', 4:'Sad', 5:'Surprise'}

  result = model.predict(image, verbose=0)
  result = list(result[0])

  img_index = result.index(max(result))
  return label_dict[img_index]

def visualize_predictions(img_ar, age_model, gender_model, emotion_model):

    resp = RetinaFace.detect_faces(img_ar)

    pic = img_ar.copy()

    for face in resp.values():
        box = face['facial_area']

        img_tensor = process_input(img_ar, box)
        emotion_img = emotion_process_input(img_ar, box)

        age = int(predict_age(img_tensor, age_model))
        gender = predict_gender(img_tensor, gender_model)
        emotion = predict_emotion(emotion_img, emotion_model)

        if gender == 'Male':
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(pic, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
        if len(resp) > 5:
          cv2.putText(pic, f"Age: {age}", (box[0], box[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
          cv2.putText(pic, f"Gender: {gender}", (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
          cv2.putText(pic, f"Emotion: {emotion}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        elif len(resp) <= 5 & len(resp) >= 2:
          cv2.putText(pic, f"Age: {age}", (box[0], box[1]-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          cv2.putText(pic, f"Gender: {gender}", (box[0], box[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
          cv2.putText(pic, f"Emotion: {emotion}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
          cv2.putText(pic, f"Age: {age}", (box[0], box[1]-55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
          cv2.putText(pic, f"Gender: {gender}", (box[0], box[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
          cv2.putText(pic, f"Emotion: {emotion}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return pic

def predict_image():
    # Open file dialog to select image
    image_path = filedialog.askopenfilename()
    original_img = Image.open(image_path)
    img_ar = np.array(original_img.convert('RGB'))
    
    age_gender_emotion_img = visualize_predictions(img_ar, age_model, gender_model, emotion_model)
    
    # Display original and predicted images
    original_img = Image.open(image_path)
    original_img.thumbnail((500, 500))
    predicted_img = Image.fromarray(age_gender_emotion_img)
    predicted_img.thumbnail((500, 500))

    original_img_tk = ImageTk.PhotoImage(original_img)
    predicted_img_tk = ImageTk.PhotoImage(predicted_img)
    
    original_label.image = original_img_tk
    predicted_label.image = predicted_img_tk

    original_label.config(image=original_img_tk)
    predicted_label.config(image=predicted_img_tk)


emotion_model= tf.keras.models.Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
emotion_model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPool2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(256,activation = 'relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(512,activation = 'relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(6, activation='softmax'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
age_model, gender_model = make_model()
emotion_model_path = 'D:/lsh/DL_Team_Project/model/emotion_model_checkpoint_epoch_82_acc_0.68_v1.h5'
emotion_model.load_weights(emotion_model_path)

age_model = age_model.to(device)
gender_model = gender_model.to(device)


    
# Tkinter 창 생성
root = tk.Tk()
root.title("이미지로 나이, 성별, 감정 예측하는 프로그램")
root.geometry("1000x800")

font = tk.font.Font(size=15, weight = 'bold')

text_label = tk.Label(root, text='아래 버튼을 눌러 예측할 이미지를 선택하세요!', font=font)
text_label.pack(pady=10)
select_button = tk.Button(root, text="이미지 선택", font=font,command=predict_image)
select_button.pack(pady=10, ipadx=20, ipady=10)

info_label = tk.Label(root, text="원본 이미지",  foreground='blue', font=font)
info_label.place(x=150, y=120)
info_label2 = tk.Label(root, text="예측 이미지", foreground='blue', font=font)
info_label2.place(x=750, y=120)

original_label = tk.Label(root)
original_label.pack(side="left", padx=10)
predicted_label = tk.Label(root)
predicted_label.pack(side="right", padx=10)

root.mainloop()
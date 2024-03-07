import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2 
from facenet_pytorch import InceptionResnetV1
from retinaface import RetinaFace

st.set_page_config(layout="wide", page_title="Age and Gender Estimator")

st.write("## Estimate Age and Gender from an Face Image")
st.write(
    ":dog: Try uploading an image to estimate age and gender. :grin:"
)
st.sidebar.write("## Upload an image :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# Download the fixed image
def save_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


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
    age_path = 'model/age_model_checkpoint_epoch_10_loss_4.57.pth'
    
    gender = Gender_Net()
    gender_path = 'model/gender_model_v3_checkpoint_epoch_19.pth'
        
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
    pred_prob = torch.sigmoid(logit)
    output = 'Male' if pred_prob.item() < 0.5 else 'Female'

    return pred_prob.item(), output


def show_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.write('\n')
    col1.image(image)
    
    img_ar = np.array(image.convert('RGB'))
    resp = RetinaFace.detect_faces(img_ar)
    pic = img_ar.copy()    
    
    results = []
    if col2.button("Estimate :wrench:"):
        for idx, face in enumerate(resp.values()):
            box = face['facial_area']
            
            img_tensor = process_input(img_ar, box)
            age = predict_age(img_tensor, age_model)
            gender_prob, gender = predict_gender(img_tensor, gender_model)
            
            if gender == 'Male':
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(pic, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
            cv2.putText(pic, f"Face {idx+1}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            results.append(f"Face {idx+1} => Age: {round(age)} // Gender: {gender}, Female Probability: {gender_prob:.2f}")

        img_pil = Image.fromarray(pic)
        col2.image(img_pil)
        for result in results:
            sex = "Male" if "Male" in result else "Female"
            background_color = "#29a3a3" if sex == 'Male' else '#ff9999'
            text_color = "white"
            col2.markdown(f'<p style="background-color: {background_color}; color: {text_color}; padding: 10px; border-radius: 10px;">{result}</p>', unsafe_allow_html=True)




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    age_model, gender_model = make_model()
    age_model = age_model.to(device)
    gender_model = gender_model.to(device)
    
    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            show_image(upload=my_upload)
    else:
        show_image('./test_images/itzy.png')
import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("🚨 No GOOGLE_API_KEY found in .env file")
else:
    genai.configure(api_key=api_key)


# --------------------------
# Load Model & Class Names
# --------------------------
@st.cache_resource
def load_model(model_path="model/leaf_model.pth", class_file="model/class_names.json"):
    with open(class_file, "r") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, class_names


# --------------------------
# Image Transform
# --------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# --------------------------
# Prediction
# --------------------------
def predict_image(model, class_names, image):
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob, idx = torch.max(probs, 1)

    plant_name = class_names[idx.item()]
    confidence = round(prob.item() * 100, 2)

    return plant_name, confidence


# --------------------------
# Gemini AI Fetch
# --------------------------
def fetch_plant_info(plant_name):
    try:
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Provide detailed medicinal and botanical information about {plant_name}."
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

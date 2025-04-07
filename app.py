import os
import gdown
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from model import EnsembleModel
from grad_cam import generate_gradcam, generate_gradcam_plus_plus, overlay_heatmap

# -------------------------
# ✅ Download model if not exists
# -------------------------
MODEL_PATH = "ensemble_model.pth"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1sfr9LnGfDmKeKhM8aOQEk8IpDbKEy0Kz",
            MODEL_PATH,
            quiet=False,
        )
        st.success("Model downloaded successfully!")

# -------------------------
# ✅ Set device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# ✅ Load model with caching
# -------------------------
@st.cache_resource
def load_model():
    model = EnsembleModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# Select target layer (EfficientNet last conv layer)
target_layer = model.efficientnet._conv_head

# -------------------------
# ✅ Class labels
# -------------------------
class_names = [
    'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
    'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions'
]

# -------------------------
# ✅ Preprocess uploaded image
# -------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(np.transpose(img_np, (2, 0, 1))).unsqueeze(0).to(device)
    return img_tensor, np.array(img)

# -------------------------
# ✅ Streamlit UI
# -------------------------
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered")
st.title("🔬 Skin Lesion Classification with Grad-CAM Visualizations")
st.markdown("Upload a dermoscopic image to classify and visualize important regions using Grad-CAM or Grad-CAM++")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

cam_type = st.radio("Select CAM Type", ["Grad-CAM", "Grad-CAM++"])

if uploaded_file is not None:
    # Process image
    img_tensor, orig_img = preprocess_image(uploaded_file)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()

    # Generate CAM
    if cam_type == "Grad-CAM":
        heatmap = generate_gradcam(model, img_tensor, target_layer)
    else:
        heatmap = generate_gradcam_plus_plus(model, img_tensor, target_layer)

    overlay = overlay_heatmap(orig_img, heatmap)

    # Display results
    st.image(orig_img, caption="Original Image", use_column_width=True)
    st.image(overlay, caption=f"{cam_type} Overlay", use_column_width=True)
    st.success(f"**Prediction:** {pred_class} ({confidence*100:.2f}%)")

import streamlit as st
from PIL import Image
from utils import load_model, predict_image, fetch_plant_info

# --------------------------
# Load Model
# --------------------------
model, class_names = load_model()

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("🌿 Plant Recognition App")
st.sidebar.info(
    """
    Upload a leaf image and the app will:
    1. Predict the plant species.
    2. Fetch botanical & medicinal information using AI.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("App Info")
st.sidebar.write("Version: 1.0")
st.sidebar.write("Model: ResNet18")
st.sidebar.write("Developers: Suraj Singh, Sparsh Kumar, Shivam Singh")
st.sidebar.markdown("---")
st.sidebar.subheader("Usage Tips")
st.sidebar.write(
    """
    - Upload clear leaf images for accurate predictions.
    - Supported formats: JPG, JPEG, PNG.
    """
)

# --------------------------
# Main UI
# --------------------------
st.title("🌱 Plant Recognition Using Machine Learning")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # --------------------------
    # Prediction
    # --------------------------
    with st.spinner("🔍 Identifying the plant..."):
        plant_name, confidence = predict_image(model, class_names, img)

    # --------------------------
    # Fetch Plant Info
    # --------------------------
    with st.spinner("💡 Fetching botanical and medicinal info..."):
        plant_info = fetch_plant_info(plant_name)

    # --------------------------
    # Layout with Columns
    # --------------------------
        st.image(img, caption="Uploaded Leaf", width=250)
        st.success(f"✅ Predicted Plant: **{plant_name}** ({confidence}%)")
        st.subheader("📜 Plant Information")
        st.write(plant_info)

    # --------------------------
    # Download Button
    # --------------------------
    st.download_button(
        label="📥 Download Plant Info",
        data=plant_info,
        file_name=f"{plant_name}_info.txt",
        mime="text/plain"
    )

else:
    st.info("📌 Upload a leaf image to get started!")

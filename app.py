import streamlit as st
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Face Parsing with Segformer",
    layout="centered",
    page_icon="🧠"
    )

# ---------------------- MODEL UTILITIES ----------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load Segformer model and processor with appropriate device."""
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def process_image(image: Image.Image, processor, model, device):
    """Run inference and return segmentation labels."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return labels

def visualize_segmentation(labels: np.ndarray):
    """Visualize segmentation mask using matplotlib with custom colormap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(labels, cmap="jet", alpha=0.8)
    ax.axis("off")
    st.pyplot(fig)

def sidebar_profile():
    # Sidebar info with custom profile section
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <style>
            .custom-sidebar {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                width: 650px;
                padding: 10px;
            }
            .profile-container {
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                width: 100%;
            }
            .profile-image {
                width: 200px;
                height: auto;
                border-radius: 15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                margin-right: 15px;
            }
            .profile-details {
                font-size: 14px;
                width: 100%;
            }
            .profile-details h3 {
                margin: 0 0 10px;
                font-size: 18px;
                color: #333;
            }
            .profile-details p {
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            .profile-details a {
                text-decoration: none;
                color: #1a73e8;
            }
            .profile-details a:hover {
                text-decoration: underline;
            }
            .icon-img {
                width: 18px;
                height: 18px;
                margin-right: 6px;
            }
        </style>

        <div class="custom-sidebar">
            <div class="profile-container">
                <img class="profile-image" src="https://res.cloudinary.com/dwhfxqolu/image/upload/v1744014185/pnhnaejyt3udwalrmnhz.jpg" alt="Profile Image">
                <div class="profile-details">
                    <h3>👨‍💻 Developed by:<br> Tahir Abbas Shaikh</h3>
                    <p>
                        <img class="icon-img" src="https://upload.wikimedia.org/wikipedia/commons/4/4e/Gmail_Icon.png" alt="Gmail">
                        <strong>Email:</strong> <a href="mailto:tahirabbasshaikh555@gmail.com">tahirabbasshaikh555@gmail.com</a>
                    </p>
                    <p>📍 <strong>Location:</strong> Sukkur, Sindh, Pakistan</p>
                    <p>
                        <img class="icon-img" src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub">
                        <strong>GitHub:</strong> <a href="https://github.com/Tahir-Abbas-555" target="_blank">Tahir-Abbas-555</a>
                    </p>
                    <p>
                        <img class="icon-img" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace">
                        <strong>HuggingFace:</strong> <a href="https://huggingface.co/Tahir5" target="_blank">Tahir5</a>
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------- MAIN UI ----------------------

def main():
    # Set page config FIRST
    
    st.title("🎯 Face Parsing using Segformer")
    st.markdown(
        "Upload an image, and this app will perform **semantic segmentation** on faces using the [Segformer](https://huggingface.co/jonathandinu/face-parsing) model."
    )

    # Load the model only once
    processor, model, device = load_model()

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("🧠 Processing with Segformer..."):
            labels = process_image(image, processor, model, device)

        with col2:
            st.markdown("#### 🖼️ Segmentation Output")
            visualize_segmentation(labels)

        st.success("✅ Segmentation completed successfully!")

    else:
        st.info("Please upload an image to start face parsing.")

# ---------------------- LAUNCH APP ----------------------

if __name__ == "__main__":
    sidebar_profile()
    main()
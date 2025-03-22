import streamlit as st
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    """Load the Segformer model and processor."""
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def process_image(image: Image.Image, processor, model, device):
    """Run inference on the image and return the segmentation mask."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return labels

def visualize_segmentation(labels: np.ndarray):
    """Visualize segmentation mask using Matplotlib."""
    fig, ax = plt.subplots()
    ax.imshow(labels, cmap="jet", alpha=0.7)
    ax.axis("off")
    st.pyplot(fig)

# Streamlit UI
st.title("Face Parsing using Segformer")
st.write("Upload an image to perform semantic segmentation on faces.")

# Load model
processor, model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process image
    with st.spinner("Processing..."):
        labels = process_image(image, processor, model, device)
        
        # Display result
        visualize_segmentation(labels)

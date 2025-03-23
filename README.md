# FaceParse AI

🔗 **Live Demo:** [FaceParse AI](https://huggingface.co/spaces/Tahir5/FaceParse-AI)

## Overview
FaceParse AI is a semantic segmentation application that uses the Segformer model to perform face parsing. This tool allows users to upload an image and segment facial features efficiently.

## Features
- **Face Segmentation** using the Segformer model
- **Real-time Visualization** of segmentation masks
- **User-friendly UI** built with Streamlit
- **Supports GPU/CPU** for fast inference

## Installation
To run this project locally, follow these steps:

### 1. Clone the Repository
```sh
git clone https://github.com/Tahir-Abbas-555/FaceParse-AI.git
cd FaceParse-AI
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Run the Application
```sh
streamlit run app.py
```

## Dependencies
- `streamlit`
- `torch`
- `transformers`
- `PIL`
- `matplotlib`
- `numpy`

## Usage
1. Open the application in your browser.
2. Upload an image (JPG, PNG, JPEG).
3. View the segmented output with highlighted facial features.

## License
This project is open-source and available under the MIT License.

---
Developed by **Tahir Abbas Shaikh** 🚀


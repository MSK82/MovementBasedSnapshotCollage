# app.py
import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from timm import create_model
from PIL import Image as PILImage
import os
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gdown
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="ASL Video to Phrase", layout="wide", page_icon="hands")
st.title("ASL Video → Phrase + PDF Translator")
st.markdown("Upload a short ASL video → Get **snapshots**, **recognized phrase**, and **downloadable PDF**")

# --- Sidebar Controls ---
st.sidebar.header("Recognition Settings")
threshold = st.sidebar.slider("Motion Sensitivity", 100000, 2000000, 900000, 50000, help="Lower = more frames")
min_frames = st.sidebar.slider("Min Fallback Frames", 3, 10, 5)
max_frames = st.sidebar.slider("Max Frames in Collage", 5, 20, 15)
cols = st.sidebar.slider("Collage Columns", 3, 6, 5)

# --- Download Real Model ---
MODEL_URL = "https://drive.google.com/uc?id=1X8x2V3k9p7s8f1g2h3j4k5l6m7n8o9p0"  # Replace with your real model
MODEL_PATH = "asl_efficientnet_b0_wlasl100.pth"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading ASL model (~90MB)..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

model_path = download_model()

# --- Load Model ---
@st.cache_resource
def load_asl_model():
    model = create_model('efficientnet_b0', pretrained=False, num_classes=29)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_asl_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ASL Labels
asl_labels = [chr(65+i) for i in range(26)] + ['space', 'del', 'nothing']

# --- File Upload ---
uploaded_file = st.file_uploader("Upload ASL Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)

    # --- Extract Frames ---
    with st.spinner("Extracting key sign frames..."):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        selected_frames = []
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                prev_gray = gray
                selected_frames.append(frame)
                continue

            diff = cv2.absdiff(gray, prev_gray)
            motion = np.sum(diff > 30)
            if motion > threshold:
                selected_frames.append(frame)
                prev_gray = gray

        cap.release()

        # Fallback
        if len(selected_frames) <= 1:
            cap = cv2.VideoCapture(video_path)
            selected_frames = []
            interval = max(1, total_frames // max(min_frames, 2))
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    selected_frames.append(frame)
            cap.release()

        if len(selected_frames) > max_frames:
            step = len(selected_frames) // max_frames
            selected_frames = selected_frames[::step][:max_frames]

        st.success(f"Extracted {len(selected_frames)} distinct sign(s)")

    # --- Hand Detection & Recognition ---
    def crop_hand(img, padding=30):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        return img[y:y+h, x:x+w]

    with st.spinner("Recognizing ASL signs..."):
        predictions = []
        hand_crops = []

        for frame in selected_frames:
            hand = crop_hand(frame)
            if hand is None:
                predictions.append("nothing")
                hand_crops.append(None)
                continue

            hand_pil = PILImage.fromarray(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))
            input_tensor = transform(hand_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                idx = output.argmax(1).item()
                pred = asl_labels[idx]
            predictions.append(pred)
            hand_crops.append(hand)

        # Build phrase
        phrase = ""
        for p in predictions:
            if p == "space":
                phrase += " "
            elif p == "del" and phrase:
                phrase = phrase[:-1]
            elif p != "nothing":
                phrase += p

    # --- Display Collage ---
    rows = (len(selected_frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    axes = axes.flatten() if len(selected_frames) > 1 else [axes]

    for i, (frame, pred, hand) in enumerate(zip(selected_frames, predictions, hand_crops)):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        label = pred if pred not in ["space", "del", "nothing"] else "·"
        axes[i].set_title(label, fontsize=18, color='lime', fontweight='bold')
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"ASL Sequence → \"{phrase}\"", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    st.pyplot(fig)

    # --- Final Result ---
    st.markdown("### Predicted Phrase")
    st.success(f"**{phrase or '—'}**")

    # --- PDF Export ---
    pdf_path = f"ASL_Translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with st.spinner("Generating PDF..."):
        with PdfPages(pdf_path) as pdf:
            fig_pdf, ax = plt.subplots(1, 1, figsize=(11, 8.5))
            ax.axis('off')
            ax.text(0.5, 0.9, "ASL Video Translation Report", ha='center', va='center', fontsize=20, fontweight='bold')
            ax.text(0.5, 0.8, f"Video: {uploaded_file.name}", ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.7, f"Predicted: {phrase}", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
            ax.text(0.5, 0.6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=10)

            # Embed collage
            buf = fig.gca().figure
            buf.canvas.draw()
            img_data = np.frombuffer(buf.canvas.tostring_rgb(), dtype=np.uint8)
            img_data = img_data.reshape(buf.canvas.get_width_height()[::-1] + (3,))
            ax.imshow(img_data)
            plt.tight_layout()
            pdf.savefig(fig_pdf, bbox_inches='tight')
            plt.close()

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download PDF Study Sheet",
            data=f,
            file_name=pdf_path,
            mime="application/pdf"
        )

    # --- Download Collage ---
    buf = fig.gca().figure
    buf.canvas.draw()
    img = np.frombuffer(buf.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(buf.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    st.download_button(
        label="Download Annotated Collage (JPG)",
        data=buffer.tobytes(),
        file_name="asl_collage.jpg",
        mime="image/jpeg"
    )

    # Cleanup
    os.unlink(video_path)
    if os.path.exists(pdf_path):
        os.unlink(pdf_path)

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

# --- Page Config ---
st.set_page_config(page_title="ASL Video to Phrase", layout="wide")
st.title("ASL Video → Phrase Translator")
st.markdown("Upload a short ASL video → Get snapshots + recognized phrase")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Motion Sensitivity", 100000, 2000000, 900000, 50000)
min_frames = st.sidebar.slider("Min Fallback Frames", 3, 10, 5)
max_frames = st.sidebar.slider("Max Frames in Collage", 5, 20, 15)
cols = st.sidebar.slider("Collage Columns", 3, 6, 5)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload ASL Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    # --- Load ASL Model ---
    @st.cache_resource
    def load_model():
        model = create_model('efficientnet_b0', pretrained=False, num_classes=29)
        # In real app: load your trained .pth
        # model.load_state_dict(torch.load("asl_model.pth", map_location="cpu"))
        model.eval()
        return model

    model = load_model()
    device = "cpu"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    asl_labels = [chr(65+i) for i in range(26)] + ['space', 'del', 'nothing']

    # --- Extract Frames ---
    with st.spinner("Extracting distinct snapshots..."):
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

        # Limit
        if len(selected_frames) > max_frames:
            step = len(selected_frames) // max_frames
            selected_frames = selected_frames[::step][:max_frames]

        st.success(f"Extracted {len(selected_frames)} snapshot(s)")

    # --- Hand Crop & Predict ---
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
        return img[y:y+h+2*padding, x:x+w+2*padding]

    with st.spinner("Recognizing signs..."):
        predictions = []
        for frame in selected_frames:
            hand = crop_hand(frame)
            if hand is None:
                predictions.append("nothing")
                continue
            hand_pil = PILImage.fromarray(cv2.cvtColor(hand, cv2.COLOR_BGR2RGB))
            input_tensor = transform(hand_pil).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                idx = output.argmax(1).item()
                pred = asl_labels[idx]
            predictions.append(pred)

        # Build phrase
        phrase = ""
        for p in predictions:
            if p == "space":
                phrase += " "
            elif p == "del" and phrase:
                phrase = phrase[:-1]
            elif p != "nothing":
                phrase += p

    # --- Create Collage with Labels ---
    rows = (len(selected_frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if len(selected_frames) > 1 else [axes]

    for i, (frame, pred) in enumerate(zip(selected_frames, predictions)):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        label = pred if pred not in ["space", "del", "nothing"] else "·"
        axes[i].set_title(label, fontsize=16, color='lime')
        axes[i].axis('off')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Final Result ---
    st.markdown("### Predicted Phrase")
    st.success(f"**{phrase or '—'}**")

    # --- Download Collage ---
    buf = fig.gca().figure
    buf.canvas.draw()
    img = np.frombuffer(buf.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(buf.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img)
    st.download_button(
        label="Download Annotated Collage",
        data=buffer.tobytes(),
        file_name="asl_collage.jpg",
        mime="image/jpeg"
    )

    os.unlink(video_path)

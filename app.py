# app.py — Waggle (keep individual borders, remove overall/dashboard wrapper)
import streamlit as st
import cv2
import pandas as pd
import tempfile
from collections import Counter
import mediapipe as mp
import pickle

st.set_page_config(layout="wide", page_title="Waggle — Dog Mood Detector")

# ----------------- Load model and encoder -----------------
clf = None
le = None
try:
    with open("model/model.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("model/encoder.pkl", "rb") as f:
        le = pickle.load(f)
except Exception as e:
    st.error(f"Could not load model/encoder — make sure model/model.pkl and model/encoder.pkl exist. ({e})")

# ----------------- MediaPipe Pose init -----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# ----------------- Styling -----------------
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"] { background-color: #2a004f; }

    .stButton>button {
        background-color: #ffeb3b !important;
        color: #2a004f !important;
        font-weight: bold;
        border-radius: 8px;
    }

    /* Video box */
    .video-wrapper {
        text-align: center;
        margin-bottom: 20px;
    }
    .video-wrapper video {
        width: 100%;
        border-radius: 10px;
        border: 2px solid #ffeb3b;
        box-shadow: 0 0 12px #ffeb3b;
    }

    /* Compact prediction card */
    .prediction-box {
        background: rgba(255, 235, 59, 0.15);
        border: 2px solid #ffeb3b;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 0 18px #ffeb3b, 0 0 25px #2a004f;
    }
    .prediction-box h2 {
        color: #ffeb3b;
        font-size: 1.3rem;
        margin-bottom: 8px;
    }
    .prediction-box h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
    }

    /* Summary card */
    .summary-card {
        background: rgba(0,0,0,0.35);
        border: 2px solid #ffeb3b;
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
        color: white;
        box-shadow: 0 0 18px #ffeb3b, 0 0 25px #2a004f;
    }
    .summary-card h3 {
        color: #ffeb3b;
        text-align: center;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Header -----------------
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:30px;
                background-color:rgba(42,0,79,0.85);
                padding:25px 40px;border-radius:25px;
                border:4px solid #ffeb3b;
                box-shadow:0 0 40px #ffeb3b,0 0 60px #2a004f;
                margin-bottom:20px;">
        <img src="https://i.pinimg.com/originals/95/ae/20/95ae202770bd57e00aa61a65cbfd167a.gif"
             alt="dog gif"
             style="width:140px;height:140px;border-radius:50%;
             border:6px solid #ffeb3b;object-fit:cover;">
        <div><h1 style="font-size:3rem;color:#ffeb3b;
                        text-shadow:2px 2px 8px rgba(0,0,0,0.6);margin:0;">Waggle</h1></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="font-size:1.2rem;color:white;background:rgba(0,0,0,0.3);
                padding:12px 15px;border-radius:12px;max-width:1000px;
                margin:0 auto 20px auto;text-align:center;">
        Upload a short video of your dog to detect its mood.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- Lighting Tips -----------------
st.markdown(
    """
    <div style="font-size:1rem;color:#ffeb3b;background:rgba(0,0,0,0.4);
                padding:10px 15px;border-radius:10px;max-width:800px;
                margin:0 auto 20px auto;text-align:center;">
        Tips for best results 💡: Use good lighting, keep the dog in full view, and avoid heavy motion blur.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- File uploader -----------------
uploaded_file = st.file_uploader("Choose a dog video (MP4 only)", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # ----------------- Analyzing -----------------
    with st.spinner("Analyzing video frames..."):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 10 == 0:
                frames.append(frame)
            frame_num += 1
        cap.release()

        pose_rows = []
        for frame in frames:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            if result.pose_landmarks:
                row = {}
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    row[f'x{i}'] = lm.x
                    row[f'y{i}'] = lm.y
                    row[f'z{i}'] = lm.z
                    row[f'v{i}'] = lm.visibility
                pose_rows.append(row)

    if not pose_rows:
        st.warning("Unable to detect the dog's pose in the video. Try a clearer one.")
    else:
        test_df = pd.DataFrame(pose_rows).fillna(0)
        if clf is None or le is None:
            st.error("Model or encoder not loaded — cannot make predictions.")
        else:
            preds = clf.predict(test_df)
            labels = le.inverse_transform(preds)
            final = Counter(labels).most_common(1)[0][0]
            mood_counts = pd.Series(labels).value_counts(normalize=True) * 100
            confidence = float(mood_counts.get(final, 0.0))

            # ----------------- Dashboard Heading -----------------
            st.markdown(
                "<h2 style='color:#ffeb3b;text-align:center;margin-top:20px;'>Dashboard</h2>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
                st.video(video_path)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class='summary-card'>
                        <h3>Summary Stats</h3>
                        <p>Total Frames Analyzed: <b>{len(frames)}</b></p>
                        <p>Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"<div class='prediction-box'><h2>Final Prediction</h2><h1>{final.upper()}</h1></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("<h3 style='color:#ffeb3b;text-align:center;'>Mood Distribution</h3>",
                            unsafe_allow_html=True)
                st.bar_chart(mood_counts)


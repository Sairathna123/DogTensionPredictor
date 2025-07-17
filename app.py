import streamlit as st
import os
import cv2
import pandas as pd
import tempfile
from collections import Counter
import mediapipe as mp
import pickle

# Load model and label encoder
with open("model/model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# GLOBAL STYLING - enhanced violet box with hover + image beside title
st.markdown("""
    <style>
        html, body, [data-testid="stApp"] {
            background-color: #ffeb3b;
        }

        /* Violet card container with hover effect */
        [data-testid="stVerticalBlock"] {
            background-color: #2a004f;
            padding: 40px;
            border-radius: 20px;
            max-width: 900px;
            margin: 40px auto;
            color: white;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        [data-testid="stVerticalBlock"]:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }

        h1, h3, h4, label, p, .markdown-text-container {
            color: white !important;
        }

        .stButton>button {
            background-color: white;
            color: #2a004f;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Outer uploader wrapper stays default (within violet box) */
        [data-testid="stFileUploader"] {
            background-color: transparent !important;
            padding: 0;
        }

        /* Inner DROP ZONE becomes yellow */
        section[data-testid="stFileUploadDropzone"] > div {
            background-color: #ffeb3b !important;
            border-radius: 10px;
            padding: 10px;
            color: black !important;
        }

        /* Browse button also yellow */
        section[data-testid="stFileUploadDropzone"] button {
            background-color: #2a004f !important;
            color: black !important;
            font-weight: bold;
            border-radius: 6px !important;
        }

        .stVideo, .stChart {
            margin-top: 20px;
        }

        .stAlert {
            border-radius: 10px;
        }

        /* Title with image */
        .title-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .title-container img {
            width: 50px;
            height: 50px;
            object-fit: cover;
            border-radius: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# Custom title with image
st.markdown("""
    <div class="title-container">
        <img src="https://i.pinimg.com/originals/ff/25/c6/ff25c648ecaf7c9ba206d5d27910c2ff.gif" alt="dog gif">
        <h1>Waggle</h1>
    </div>
""", unsafe_allow_html=True)

st.write("Upload a short video of your dog to detect its mood — friendly, nervous, or aggressive.")

# Upload section
video_file = st.file_uploader("Choose a dog video (MP4 only)", type=["mp4"])

if video_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Display video
    st.video(video_path)

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % 10 == 0:  # Sample every 10th frame
            frames.append(frame)
        frame_num += 1
    cap.release()

    # Extract pose landmarks
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
        st.warning("⚠️ Unable to detect the dog's pose in the video. Try a clearer video.")
    else:
        # Convert to DataFrame and predict
        test_df = pd.DataFrame(pose_rows)
        preds = clf.predict(test_df)
        labels = le.inverse_transform(preds)

        # Final mood (most frequent)
        final = Counter(labels).most_common(1)[0][0]

        st.markdown(f"<h3>Final Predicted Mood: {final.upper()}</h3>", unsafe_allow_html=True)

        # Show mood distribution
        mood_counts = pd.Series(labels).value_counts()
        st.markdown("#### Mood Prediction Chart")
        st.bar_chart(mood_counts)

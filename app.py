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


# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Streamlit UI
st.title("Dog Mood Classifier")
st.write("Upload a short video of your dog to detect its mood: friendly, nervous, or aggressive.")

video_file = st.file_uploader("Upload Dog Video", type=["mp4"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)

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
        st.warning("Could not detect dog pose.")
    else:
        test_df = pd.DataFrame(pose_rows)
        preds = clf.predict(test_df)
        labels = le.inverse_transform(preds)
        final = Counter(labels).most_common(1)[0][0]

        st.success(f"Final Predicted Mood: **{final.upper()}**")

        mood_counts = pd.Series(labels).value_counts()
        st.bar_chart(mood_counts)

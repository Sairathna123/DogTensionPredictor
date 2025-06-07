# DogTensionPredictor

DogTensionPredictor is a video-based machine learning application that detects and classifies a dog's emotional state—friendly, nervous, or aggressive—based on body posture extracted using MediaPipe. It allows users to upload short dog videos and leverages pose keypoint analysis and a trained classifier to assess the dog's mood. The app provides real-time predictions and visual summaries, making it useful for behavioral monitoring, training, and welfare assessment by pet owners, researchers, and animal care professionals.

How It Works

- Upload a short dog video in .mp4 format.
- The app automatically extracts frames and detects pose keypoints using MediaPipe.
- A trained machine learning model predicts the dog's emotional state for each frame.
The final output includes:
- A single predicted mood (majority vote)
- A bar chart showing mood distribution across frames

Features

- Upload and analyze dog videos directly in the web interface
- Extract pose keypoints using MediaPipe
- Predict dog mood using a trained classification model
- Visualize prediction confidence via bar chart

Technologies Used

- Python
- Streamlit
- MediaPipe
- OpenCV
- Pandas
- Scikit-learn

  

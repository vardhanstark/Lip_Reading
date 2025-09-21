## PROJECT - 👄 Lip Reading 
## 📝 Overview

This project is a Lip Reading application that predicts spoken words from a person’s lip movements in a video. It combines Computer Vision for video processing and Deep Learning for sequence prediction. Users can interact with the project via a Streamlit web interface.

## ✨ Features

🎥 Upload a video and extract lip movements

🧠 Predict spoken words using the LipNet deep learning model

🖥️ User-friendly Streamlit interface

⏱️ Real-time transcription of lip movements
## 🛠️ Technologies Used

Deep Learning Framework: TensorFlow 🧠

Base Model: LipNet (CNN + LSTM for sequence modeling) 🔤

UI Framework: Streamlit 🖥️

Video Processing: FFmpeg 🎞️, imageio 🖼️

Custom Python Utilities:

utils.py → Data loading, preprocessing, and converting predictions (load_data, num_to_char) 📦

modelutil.py → Load the trained LipNet model (load_model) ⚙️

automated.py → Handles automated processing of videos and batch predictions ⚡

streamlitapp.py → Main Streamlit web interface for uploading videos and displaying predictions 🖥️


## 🚀 Usage
1.Run the Streamlit app: streamlit run streamlitapp.py
2.pload your video via the web interface 📤
3.The app will process the video and display the predicted text 📝

## 📂 File Structure
```
├── automated.py       # Handles automated processing and batch predictions
├── modelutil.py       # Loads the trained LipNet model
├── streamlitapp.py    # Streamlit app for video upload and text prediction
├── utils.py           # Utility functions for data preprocessing and prediction
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
```
## ⚠️ Notes
Videos should have clear lip visibility for accurate predictions 👁️

LipNet is trained on specific datasets; predictions may vary with new speakers or languages 🌍


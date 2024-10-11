import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the face mask detector model
model = load_model("mask_detector.h5")  # Change the path if necessary

# Function to start the video stream
def start_video_stream():
    # Initialize the front camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("Error: Could not open video stream.")
        return None
    
    return camera

# Function to capture frames from the camera
def capture_frame(camera):
    ret, frame = camera.read()
    
    if not ret:
        st.error("Error: Could not read frame.")
        return None
    
    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# Function to preprocess the frame for the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # Resize frame to model input size
    frame = np.expand_dims(frame, axis=0)   # Add batch dimension
    frame = frame / 255.0                    # Normalize pixel values
    return frame

# Streamlit app layout
st.title("Real-Time Mask Detection")

# Initialize session state for camera and running status
if "camera" not in st.session_state:
    st.session_state.camera = None
if "running" not in st.session_state:
    st.session_state.running = False

# Start/stop button
if st.button("Start Stream"):
    if st.session_state.camera is None:
        st.session_state.camera = start_video_stream()
        if st.session_state.camera:
            st.session_state.running = True
            st.write("Camera started.")
        else:
            st.write("Failed to start the camera.")
    else:
        st.write("Camera is already running.")

if st.button("Stop Stream"):
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.running = False
        st.write("Camera stopped.")
    else:
        st.write("Camera is not running.")

# Display the video stream
if st.session_state.running:
    stframe = st.empty()  # Placeholder for video stream
    
    while st.session_state.running:
        frame = capture_frame(st.session_state.camera)
        
        if frame is not None:
            # Preprocess the frame for model prediction
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            label = "Mask" if prediction[0][0] > 0.5 else "No Mask"

            # Display the prediction on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Mask" else (255, 0, 0), 2)

            # Show the frame in the Streamlit app
            stframe.image(frame, channels="RGB", use_column_width=True)
        else:
            st.error("Error: Failed to capture frame.")

    # Release camera on stop
    if st.session_state.camera:
        st.session_state.camera.release()
        st.session_state.camera = None

# End of the app
st.write("Press 'Start Stream' to begin video capture.")

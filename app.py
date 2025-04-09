import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import time
import numpy as np
from io import StringIO

st.set_page_config(page_title="Real-Time Object Detection", layout="centered")

st.title("🎥 Real-Time Object Detection with YOLOv8")

# Load YOLO model
model = YOLO("yolov8s.pt")

# Sidebar UI
st.sidebar.header("⚙️ Settings")
selected_classes = st.sidebar.multiselect(
    "Choose object classes to detect (optional)",
    options=model.model.names.values(),
    default=[]
)
save_frames = st.sidebar.checkbox("💾 Save frames with detections")

# Initialize CSV log as a list of dictionaries
detection_log = []

# Placeholder for video
frame_placeholder = st.empty()

# Start webcam
cap = cv2.VideoCapture(0)

st.sidebar.markdown("Press **Stop** to end the app.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not read from webcam.")
            break

        results = model(frame)[0]

        # Filter by selected class (if any)
        if selected_classes:
            results = results.filter(classes=[
                cls_id for cls_id, name in model.model.names.items() if name in selected_classes
            ])

        # Log detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.model.names[class_id]
            confidence = float(box.conf[0])
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            detection_log.append({
                "timestamp": timestamp,
                "class": class_name,
                "confidence": round(confidence, 2)
            })

        # Annotate + save if required
        annotated_frame = results.plot()
        if save_frames and len(results.boxes.cls) > 0:
            filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)

        # Convert to RGB for Streamlit
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

except Exception as e:
    st.error(f"🚫 Error: {e}")
finally:
    cap.release()

# ===== CSV Export =====
if detection_log:
    df = pd.DataFrame(detection_log)
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Detection Log (CSV)",
        data=csv,
        file_name="detection_log.csv",
        mime="text/csv"
    )

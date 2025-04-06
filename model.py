import numpy as np
import pandas as pd
import cv2
from collections import deque
from ultralytics import YOLO
import pickle
import threading
from fastapi import FastAPI
import time
import warnings
import asyncio
import uvicorn

warnings.filterwarnings("ignore")

# --- Load Pre-trained Model ---
with open("sensory_model.pkl", "rb") as f:
    sensory_model = pickle.load(f)
print("Model loaded from sensory_model.pkl")

image_model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
if not cap.isOpened():
    raise FileNotFoundError("Could not open camera stream")

# Image model variables
relative_speeds = deque(maxlen=10)
weights = {'traffic_density': 0.3, 'lane_discipline': 0.3, 'relative_speed': 0.4}
lk_params = dict(winSize=(10, 10), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# --- Global Variable for Latest Risk Score and Parameters ---
latest_risk_data = {
    "score": 0.0,
    "traffic_density": 0.0,
    "relative_velocity": 0.0,
    "lane_discipline": 0.0,
    "timestamp": None
}
lock = threading.Lock()

# --- Mock Sensor Input (No Hardware Needed) ---
def get_sensor_data():
    timestamp = pd.Timestamp.now()
    acc_x, acc_y, acc_z = np.random.normal(0, 1, 3)
    gyro_x, gyro_y, gyro_z = np.random.normal(0, 0.5, 3)
    return pd.DataFrame({
        'Timestamp': [timestamp],
        'AccX': [acc_x], 'AccY': [acc_y], 'AccZ': [acc_z],
        'GyroX': [gyro_x], 'GyroY': [gyro_y], 'GyroZ': [gyro_z]
    })

# --- Utility Functions for Dashcam Model ---
def region_of_interest(img):
    height, width = img.shape[:2]
    roi = np.array([[(width * 0.1, height), (width * 0.9, height),
                     (width * 0.6, height * 0.6), (width * 0.4, height * 0.6)]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame, scale=0.5):
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    gray = cv2.cvtColor(cv2.bitwise_and(small_frame, small_frame, mask=mask_white), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    roi_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)
    
    left_lane, right_lane = None, None
    if lines is not None:
        left_lines, right_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            if 0.3 < abs(slope) < 1.5:
                (left_lines if slope < 0 else right_lines).append(line[0])
        left_lane = np.mean(left_lines, axis=0, dtype=np.int32) / scale if left_lines else None
        right_lane = np.mean(right_lines, axis=0, dtype=np.int32) / scale if right_lines else None
    return left_lane, right_lane

def calculate_lane_discipline(boxes, left_lane, right_lane, width):
    if left_lane is None or right_lane is None or len(boxes) == 0:
        return 50.0
    
    lane_left_x = min(left_lane[0], left_lane[2])
    lane_right_x = max(right_lane[0], right_lane[2])
    lane_width = lane_right_x - lane_left_x
    
    discipline_score = 0
    total_vehicles = len(boxes)
    
    for box in boxes:
        x_left = box[0]
        x_right = box[2]
        vehicle_width = x_right - x_left
        x_center = (x_left + x_right) / 2
        
        if x_center < lane_left_x:
            distance_outside = lane_left_x - x_right
            discipline_factor = max(0, 1 - (distance_outside / vehicle_width))
        elif x_center > lane_right_x:
            distance_outside = x_left - lane_right_x
            discipline_factor = max(0, 1 - (distance_outside / vehicle_width))
        else:
            left_overlap = max(0, lane_left_x - x_left)
            right_overlap = max(0, x_right - lane_right_x)
            total_overlap = left_overlap + right_overlap
            discipline_factor = max(0, 1 - (total_overlap / vehicle_width))
        
        discipline_score += discipline_factor
    
    return (discipline_score / total_vehicles) * 100 if total_vehicles > 0 else 50.0

def calculate_traffic_density(boxes, frame_area):
    if not boxes.any():
        return 0
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    return min(sum(areas) / frame_area * 100, 100)

def calculate_relative_speed(prev_gray, curr_gray):
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=50, qualityLevel=0.3, minDistance=10, blockSize=5)
    if p0 is None or len(p0) == 0:
        return 0
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    if p1 is None or len(p1) == 0:
        return 0
    speed_m_s = np.mean(np.linalg.norm(p1 - p0, axis=1)) * fps / 20
    return speed_m_s * 3.6

def calculate_image_risk_score(traffic_density, lane_discipline_pct, relative_speed):
    td_risk = min(traffic_density / 50, 1.0)
    ld_risk = 1 - (lane_discipline_pct / 100)
    rs_risk = min(max(0, abs(relative_speed) - 20) / 80, 1.0)
    return 100 * (weights['traffic_density'] * td_risk + weights['lane_discipline'] * ld_risk + weights['relative_speed'] * rs_risk)

# --- Sensory Model Utilities ---
def risk_score(pred_class, class_prob):
    if pred_class == 0:   # Aggressive
        min_score, max_score = 80, 100
    elif pred_class == 1:  # Normal
        min_score, max_score = 0, 30
    elif pred_class == 2:  # Slow
        min_score, max_score = 30, 70
    return min_score + class_prob * (max_score - min_score)

def calculate_ewm_risk(input_data, model, alpha=0.3):
    X_input = input_data.drop(['Timestamp'], axis=1)
    y_pred_proba = model.predict_proba(X_input)
    risk_scores = [risk_score(np.argmax(proba), max(proba)) for proba in y_pred_proba]
    return pd.Series(risk_scores).ewm(alpha=alpha, adjust=False).mean().iloc[-1]

# --- Current Risk Score with History ---
SENSORY_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4
HISTORY_WEIGHT = 0.4
CURRENT_WEIGHT = 0.6
history_buffer = deque(maxlen=100)

def combine_risk_scores(sensory_score, image_score):
    return SENSORY_WEIGHT * sensory_score + IMAGE_WEIGHT * image_score

def get_current_risk_score(sensory_data, frame, prev_frame, last_boxes, last_lanes, sensory_model, image_model, alpha=0.3):
    global latest_risk_data
    sensory_ewm_score = calculate_ewm_risk(sensory_data, sensory_model, alpha)
    scale = 0.5
    small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    frame_area = small_frame.shape[0] * small_frame.shape[1]
    results = image_model(small_frame, conf=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy() / scale if results[0].boxes else np.array([])
    lanes = detect_lanes(frame, scale=scale)
    last_boxes = boxes if boxes.any() else last_boxes
    last_lanes = lanes if (lanes[0] is not None or lanes[1] is not None) else last_lanes

    curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        relative_speed = calculate_relative_speed(prev_gray, curr_gray)
        relative_speeds.append(relative_speed)

    traffic_density = calculate_traffic_density(last_boxes, frame_area * (1/scale)**2)
    lane_discipline_pct = calculate_lane_discipline(last_boxes, *last_lanes, frame.shape[1])
    avg_speed = np.mean(relative_speeds) if relative_speeds else 0
    image_risk_score = calculate_image_risk_score(traffic_density, lane_discipline_pct, avg_speed)

    current_score = combine_risk_scores(sensory_ewm_score, image_risk_score)
    historical_score = np.mean(history_buffer) if history_buffer else current_score
    final_score = CURRENT_WEIGHT * current_score + HISTORY_WEIGHT * historical_score
    history_buffer.append(final_score)

    with lock:
        latest_risk_data["score"] = final_score
        latest_risk_data["traffic_density"] = traffic_density
        latest_risk_data["relative.Microsoft.NETCore.Appvelocity"] = avg_speed
        latest_risk_data["lane_discipline"] = lane_discipline_pct
        latest_risk_data["timestamp"] = str(sensory_data['Timestamp'].iloc[0])

    return final_score, small_frame, last_boxes, last_lanes, traffic_density, avg_speed, lane_discipline_pct

# --- Real-Time Processing ---
def process_live_data(sensory_model, image_model, alpha=0.3):
    prev_frame = None
    last_boxes = np.array([])
    last_lanes = (None, None)

    cv2.namedWindow("Risk Assessment", cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            sensory_data = get_sensor_data()
            ret, frame = cap.read()
            if not ret:
                print("Camera feed ended")
                break

            current_score, prev_frame, last_boxes, last_lanes, traffic_density, relative_velocity, lane_discipline = get_current_risk_score(
                sensory_data, frame, prev_frame, last_boxes, last_lanes, sensory_model, image_model, alpha
            )

            # Draw Hough lines (lanes) on the frame
            if last_lanes[0] is not None:
                x1, y1, x2, y2 = last_lanes[0].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Left lane in blue
            if last_lanes[1] is not None:
                x1, y1, x2, y2 = last_lanes[1].astype(int)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Right lane in red

            # Display parameters on video feed
            cv2.putText(frame, f"Risk: {current_score:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Traffic Density: {traffic_density:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rel. Velocity: {relative_velocity:.1f} km/h", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Lane Discipline: {lane_discipline:.1f}%", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Risk Assessment", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.033)

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# --- FastAPI Setup ---
app = FastAPI()

@app.get("/risk")
async def get_risk():
    with lock:
        return latest_risk_data

# --- Run Everything ---
async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=5000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    processing_thread = threading.Thread(target=process_live_data, args=(sensory_model, image_model))
    processing_thread.daemon = True
    processing_thread.start()

    try:
        asyncio.run(run_server())
    except RuntimeError as e:
        if "running event loop" in str(e):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(run_server())
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("Stopped by user")
            else:
                loop.run_until_complete(run_server())
        else:
            raise e
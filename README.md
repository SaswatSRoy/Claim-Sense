# Claim Sense

![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Web-blue)
![Tech Stack](https://img.shields.io/badge/TechStack-Kotlin%2C%20YOLO%2C%20Firebase%2C%20ONNX%2C%20RandomForest-orange)

**Claim Sense** is an AI-powered SaaS solution designed for insurance companies and fleet managers to assess driving behavior and risk in real-time. It leverages on-device sensors and computer vision to generate driver risk scores, reduce insurance fraud, and improve road safety.

**GitHub Repository**: [https://github.com/anmol0705/Claim-Sense](https://github.com/anmol0705/Claim-Sense)

---

## 🚀 Features

- 📷 **Dashcam-Based Driving Analysis**
- 🧠 **Real-time Risk Prediction with On-Device AI (ONNX)**
- 📱 **Driver App for Viewing Risk Score, Claims, and Disputes**
- 🖥️ **Web Dashboard for Fleet Managers and Insurers**
- 🔐 **Google Authentication and Secure Data Handling via Firebase**
- ☁️ **Cloud Integration for Storage and Real-Time Sync**
- ✅ **User Consent-Driven Data Collection**

---

## 📱 Applications Overview

### 1. **Dashcam App**
- Captures real-time video and sensor data
- Runs YOLO and Random Forest locally for risk assessment
- Uploads results to Firebase with authentication

### 2. **Driver App**
- Displays personal driving behavior and risk score
- Allows drivers to raise disputes and initiate claims

### 3. **Fleet Manager / Insurer Dashboard**
- Monitor insured drivers or an entire fleet
- Analyze risk patterns, view driving history, manage claims

---

## 🧠 System Architecture

```
[ Android Dashcam App ]
  ├─ Captures video + sensor data
  ├─ YOLOv5 (visual) + Random Forest (sensor) → ONNX Runtime
  └─ Uploads risk score to Firebase

[ Driver App ]
  ├─ View scores & trip logs
  └─ Raise disputes / insurance claims

[ Web Dashboard ]
  ├─ Google Auth login (admin/fleet)
  └─ Monitor drivers & analytics

[ Firebase Backend ]
  ├─ Realtime Database
  ├─ Cloud Storage
  └─ Authentication
```

---

## 🧪 Machine Learning Models

### 1. **YOLOv5 (Visual Analysis)**
- **Input**: Dashcam video frames
- **Output**:
  - Traffic density
  - Lane discipline violations
  - Estimated relative speed of nearby vehicles

### 2. **Random Forest Classifier (Sensor Data)**
- **Input**: Accelerometer (Ax, Ay, Az) and Gyroscope (Gx, Gy, Gz)
- **Output**: Behavior-based risk component

### 3. **Risk Score Calculation**

```
Final Risk Score = α * Visual Risk + β * Sensor Risk
(α and β are tunable weight coefficients)
```

---

## 🔒 Privacy and Consent

Claim Sense asks for **explicit user consent** before collecting any sensor or camera data. We prioritize transparency and ensure data is only used for **risk assessment** and **claim resolution**.

---

## 🔧 Tech Stack

| Layer       | Tech Used                              |
|------------|-----------------------------------------|
| Frontend   | Kotlin (Jetpack Compose), React, HTML/CSS |
| Backend    | Firebase Realtime DB, Cloud Storage, Auth |
| ML Models  | YOLOv5 (Vision), Random Forest (Sensor)   |
| Inference  | ONNX Runtime (on-device)                 |
| Deployment | Android APKs, Firebase Hosting           |

---

## 🔨 Setup Instructions

### Prerequisites
- Android Studio
- Firebase account + CLI
- Node.js (for web dashboard)
- Python (optional for training)

### Firebase Setup
1. Go to Firebase Console
2. Create a new project
3. Enable:
   - Google Authentication
   - Realtime Database
   - Cloud Storage
4. Download `google-services.json` and add to both Android apps

### Clone and Run

```bash
git clone https://github.com/anmol0705/Claim-Sense.git
cd Claim-Sense
```

#### For Android Apps
- Open Dashcam and Driver apps in Android Studio
- Add `google-services.json` in `app/` directory
- Run on a physical Android device

#### For Web Dashboard

```bash
cd web-dashboard
npm install
firebase login
firebase deploy
```

---

## 📊 Example Use Case

1. Vehicle dashcam app captures sensor data and video.
2. On-device model processes and uploads risk score to Firebase.
3. Fleet manager views daily performance via web dashboard.
4. Driver raises claim using driver app after an incident.
5. Insurer validates claim with video + data.

---

## 📌 Future Enhancements

- Integration with OBD-II vehicle diagnostics
- Heatmaps of accident-prone zones
- AI-driven claim adjudication
- Real-time driver alerts & gamification

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature-xyz`
3. Commit your changes
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

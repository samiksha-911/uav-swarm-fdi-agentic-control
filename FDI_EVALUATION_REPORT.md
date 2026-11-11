# Intelligent Fault Detection and Identification (FDI) in UAV Missions
## Experimental Evaluation & Benchmarking Report

### 1. Executive Summary
This report presents the validation results of the **PX4-Native Fault Detection and Identification (FDI)** system running on a multirotor platform (Gazebo SITL / PX4 v1.18 Autopilot). 

The model relies on **Extra Trees Ensemble learning** fed by a 50-sample sliding window (5.0s temporal buffer at 10 Hz) extracting 12 statistical/temporal features across 61 raw telemetry channels (732 engineered features total).

---

### 2. Cross-Validation Results (Leave-One-Run-Out)
- **Validation Scheme**: Leave-One-Run-Out (LOOR) Group Cross-Validation
- **Total Flight Runs**: 30 independent flights (5 runs per class)
- **Evaluated Windows**: 455 total windows
- **LOOR Accuracy**: **100.00%**
- **LOOR Balanced Accuracy**: **100.00%**
- **LOOR Macro-F1**: **1.0000**

#### Detailed Classification Metrics:
| Fault Class | Precision | Recall | F1-Score | Evaluation Support |
| :--- | :---: | :---: | :---: | :---: |
| **Accelerometer Failure** | 1.00 | 1.00 | 1.00 | 75 windows |
| **Barometer Failure**     | 1.00 | 1.00 | 1.00 | 75 windows |
| **Compass Failure**       | 1.00 | 1.00 | 1.00 | 75 windows |
| **GPS Failure**           | 1.00 | 1.00 | 1.00 | 75 windows |
| **Gyroscope Failure**     | 1.00 | 1.00 | 1.00 | 75 windows |
| **No Failure (Healthy)**  | 1.00 | 1.00 | 1.00 | 80 windows |

---

### 3. Real-Time Online Streaming Verification
The model was tested live in real-time (`step71_live_px4_fdi.py`) against interactive simulated flight in Gazebo:
- **Baseline Healthy Flight**: Correctly reported as `No Failure` with **88% - 91% confidence** (zero false alarms).
- **Gyroscope Failure (`failure gyro off`)**: Correctly diagnosed with **78% confidence**.
- **Accelerometer Failure (`failure accel off`)**: Correctly diagnosed with **83% confidence**.
- **GPS Failure (`failure gps off`)**: Correctly diagnosed with **62% confidence**.
- **Compass Failure (`failure mag off`)**: Correctly diagnosed with **45% - 48% confidence**.
- **Barometer Failure (`failure baro off`)**: Correctly diagnosed with **48% - 50% confidence**.

---

### 4. Top Telemetry Discriminators
The most significant features separating the fault classes include:
- **angular_velocity_invalid_last** (Importance: 0.0304)
- **cs_baro_hgt_last** (Importance: 0.0264)
- **cs_mag_last** (Importance: 0.0262)
- **cs_mag_hdg_min** (Importance: 0.0249)
- **angular_velocity_invalid_max** (Importance: 0.0248)
- **cs_baro_hgt_min** (Importance: 0.0236)
- **cs_mag_hdg_last** (Importance: 0.0225)
- **cs_mag_min** (Importance: 0.0223)
- **angular_velocity_invalid_mean** (Importance: 0.0198)
- **cs_mag_mean** (Importance: 0.0184)

---

### 5. Generated Figures
- **Confusion Matrix**: `fdi_confusion_matrix.png`
- **Feature Importance Chart**: `fdi_top_features.png`

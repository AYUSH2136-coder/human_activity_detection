# Mobile Health (MHEALTH) Dataset Specification

The **MHEALTH (Mobile Health) dataset** is a benchmark dataset designed to facilitate research on human behavior analysis and mobile health. It comprises body motion and vital signs recordings from ten volunteers while performing twelve physical activities. Shimmer2 wearable sensors are placed on different body parts (chest, right wrist, left ankle) to collect inertial sensor data.

---

## 📍 Sensor Placement and Specifications

Sensors are strategically positioned to capture the dynamics of body movement from the torso and extremities:

1. **Chest**:
   - Acceleration (3-axis)
   - Electrocardiogram (ECG) — 2 leads (Note: In our pipeline, we focus primarily on inertial signals).
2. **Right Wrist**:
   - Acceleration (3-axis)
   - Angular velocity (gyroscope, 3-axis)
   - Magnetic field (magnetometer, 3-axis)
3. **Left Ankle**:
   - Acceleration (3-axis)
   - Angular velocity (gyroscope, 3-axis)
   - Magnetic field (magnetometer, 3-axis)

### Sensor Parameters
* **Sampling Rate**: 50 Hz
* **Subjects**: 10 healthy adult volunteers (gender: male/female, age: 20-30 years).
* **Format**: CSV format per subject, aggregated into a unified dataset.

---

## 🏃 Physical Activity Classes

The dataset records 12 distinct physical activities, covering both static postures and dynamic exercises:

| Class ID | Label (0-indexed) | Activity Description | Type |
| :---: | :---: | :--- | :--- |
| **1** | `0` | Standing still | Static |
| **2** | `1` | Sitting and relaxing | Static |
| **3** | `2` | Lying down | Static |
| **4** | `3` | Walking | Dynamic (Low Intensity) |
| **5** | `4` | Climbing stairs | Dynamic (Medium Intensity) |
| **6** | `5` | Waist bends forward | Dynamic (Low Intensity) |
| **7** | `6` | Frontal elevation of arms | Dynamic (Low Intensity) |
| **8** | `7` | Knees bending (crouching) | Dynamic (Medium Intensity) |
| **9** | `8` | Cycling | Dynamic (Medium Intensity) |
| **10** | `9` | Jogging | Dynamic (High Intensity) |
| **11** | `10` | Running | Dynamic (High Intensity) |
| **12** | `11` | Jump front & back | Dynamic (High Intensity) |

> [!NOTE]
> In raw records, a class ID of `0` denotes "Null Activity" (idle transitions between activities). Our preprocessing pipeline removes these transition intervals to focus on pure activity classification.

---

## 📊 Column Mapping

The raw dataset contains 23 columns (22 features and 1 label). In our modular codebase, we focus on the **12 core inertial sensor channels** from the right wrist and left ankle (accelerometer and gyroscope) to classify movements:

| Col Index | Sensor Component | Body Location | Axis | Column Name |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Accelerometer | Chest | X | `acc_chest_x` |
| **1** | Accelerometer | Chest | Y | `acc_chest_y` |
| **2** | Accelerometer | Chest | Z | `acc_chest_z` |
| **3** | ECG | Chest | Lead 1 | `ecg_lead_1` |
| **4** | ECG | Chest | Lead 2 | `ecg_lead_2` |
| **5** | Accelerometer | Right Wrist | X | `acc_wrist_x` |
| **6** | Accelerometer | Right Wrist | Y | `acc_wrist_y` |
| **7** | Accelerometer | Right Wrist | Z | `acc_wrist_z` |
| **8** | Gyroscope | Right Wrist | X | `gyro_wrist_x` |
| **9** | Gyroscope | Right Wrist | Y | `gyro_wrist_y` |
| **10** | Gyroscope | Right Wrist | Z | `gyro_wrist_z` |
| **11** | Magnetometer | Right Wrist | X | `mag_wrist_x` |
| **12** | Magnetometer | Right Wrist | Y | `mag_wrist_y` |
| **13** | Magnetometer | Right Wrist | Z | `mag_wrist_z` |
| **14** | Accelerometer | Left Ankle | X | `acc_ankle_x` |
| **15** | Accelerometer | Left Ankle | Y | `acc_ankle_y` |
| **16** | Accelerometer | Left Ankle | Z | `acc_ankle_z` |
| **17** | Gyroscope | Left Ankle | X | `gyro_ankle_x` |
| **18** | Gyroscope | Left Ankle | Y | `gyro_ankle_y` |
| **19** | Gyroscope | Left Ankle | Z | `gyro_ankle_z` |
| **20** | Magnetometer | Left Ankle | X | `mag_ankle_x` |
| **21** | Magnetometer | Left Ankle | Y | `mag_ankle_y` |
| **22** | Magnetometer | Left Ankle | Z | `mag_ankle_z` |
| **23** | Label | — | — | `Activity` |

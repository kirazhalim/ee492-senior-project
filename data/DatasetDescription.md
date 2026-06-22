# Dataset Description and Usage Guide

## 1. Overview
The dataset consists of multi-channel physiological recordings designed for cough analysis and state classification. It captures data from four distinct sensors simultaneously to correlate audio (cough sounds) with physical body movement (chest expansion and acceleration).

## 2. File Specifications
* **Format:** Comma-Separated Values (CSV).
* **Structure:** Headerless (data begins immediately on Row 1).
* **Naming Convention:** Files follow a `NNN_YYYYMMDD_Subject_Activity_Context.csv` format (e.g., `000_20250602_subject01_standing_clean.csv`).

## 3. Channel Mapping
The dataset contains four columns corresponding to raw sensor inputs:

| Col | Sensor Type | Description |
| :--- | :--- | :--- |
| **1** | Pulmonary Microphone | Captures internal chest sounds (lung acoustics). |
| **2** | Ambient Microphone | Captures external environmental audio. |
| **3** | Stretch Sensor (+ Label) | **Hybrid Channel:** Contains both chest expansion data and a binary cough label. |
| **4** | Accelerometer (Z-Axis) | Captures vertical body acceleration/movement. |

## 4. Data Decoding (Column 3)
The 3rd Column (Stretch Sensor) uses bitwise encoding to store the sensor value and the label in one integer.

**Decoding Logic:**
* **Extract Label (Cough Event):** Perform a bitwise `AND` with 1 (`Label = Raw_Value & 1`).
    * `1` = Cough event occurring.
    * `0` = No cough.
* **Extract Signal (Stretch Magnitude):** Perform a bitwise `Right Shift` by 1 (`Stretch_Signal = Raw_Value >> 1`).

## 5. Acquisition Parameters
* **Sampling Rate:** 4800 Hz.
* **Duration:** Approximately 20 seconds per recording.
* **Data Type:** Raw ADC values (integers).

## 6. Dataset Distribution Statistics
The dataset covers various activities and noise conditions. The distribution is as follows:

* **Activities Recorded:** Sitting, Walking, Running, Standing.
* **Noise Conditions / Contexts:** Clean, Cough noise, Music noise, Sneeze noise, Snooze noise, Door noise, Generic noise, False-positive.


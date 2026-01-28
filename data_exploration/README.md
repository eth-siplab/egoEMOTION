# egoEMOTION Dataset Information

*egoEMOTION* contains **per-participant sessions** with **synchronized and raw sensor streams** saved as `.npy`, and **video recordings** saved as `.mp4`. Each participant folder (e.g., `005/`) contains the same set of streams.


## Dataset structure

```
<dataset_root>/
├── 005/
│ ├── ecg_90fps.npy
│ ├── ecg.npy
│ ├── eda_90fps.npy
│ ├── eda.npy
│ ├── et.npy
│ ├── gaze_90fps.npy
│ ├── gaze.npy
│ ├── imu_right_90fps.npy
│ ├── imu_right.npy
│ ├── intensity_90fps.npy
│ ├── intensity.npy
│ ├── lbptop.npy
│ ├── ppg_ear_90fps.npy
│ ├── ppg_ear.npy
│ ├── ppg_nose_90fps.npy
│ ├── ppg_nose.npy
│ ├── pupils_90fps.npy
│ ├── pupils.npy
│ ├── rr_90fps.npy
│ ├── rr.npy
│ ├── Session_A_005.csv
│ ├── Session_B_005.csv
│ ├── pov.mp4
│ └── webcam.mp4
├── 006/
│ └── ...
└── ...
```

### Naming conventions
- `*_90fps.npy`: stream resampled to 90 Hz (intended to match the 90 FPS eye tracking timeline). Used for the deep learning-based approaches to have a unified sampling rate. Original frame rate is used for other approaches.
- `*.npy` (without `_90fps`): stream at its original sampling rate. 
- `pov.mp4`: first-person point-of-view video.
- `webcam.mp4`: external webcam face video.
- `Session_A_<id>.csv`, `Session_B_<id>.csv`: session-level self-reports.
- `personality_questionnaire_results`: participant mean and T-score big five personality results.
- `task_times.npy`: a dictionary that stores start and end indices (in the 90 Hz index space) for each participant's task segments. For example 
```
task_times = {
  "004": {
    "session_A": [36880, 149899],
    "session_B": [194815, 408520],
    "video_Neutral": [41056, 44691],
    "video_Disgust": [57584, 61254],
    "slenderman": [178880, 192900],
    "jenga": [210695, 240980],
    ...
  },
  "005": { ... }
}
```

Please note that all tasks indices are shifted by `Session_A[0]` for correct alignment during feature calculation (i.e., `task_idx_shifted = task_idx - Session_A[0]`) as the recording start index is `recording_start = Session_A[0]` and the recording end index is `recording_end = Session_B[1]`.

## Streams overview

The table below outlines all data streams captured in *egoEMOTION*. The parameter `N` signifies the length of the array (ignoring different sampling frequencies).

| Stream | File(s) | Description                                                                               | Original / Resampled Frequency | Shape |
|---|---|-------------------------------------------------------------------------------------------|---|---|
| ECG | `ecg.npy`, `ecg_90fps.npy` | Electrocardiogram waveform                                                                | 1024 / 90 Hz | `(N, 1)` |
| EDA | `eda.npy`, `eda_90fps.npy` | Electrodermal activity waveform                                                           | 256 / 90 Hz | `(N, 1)` |
| ET | `et.npy` | Eye-tracking videos                                                                       | 90 Hz | `(N, 48, 128)`|
| Gaze | `gaze_90fps.npy` | 2D gaze position `[x, y]`                                                                 | 90 Hz | `(N, 2)` |
| IMU (right) | `imu_right.npy`, `imu_right_90fps.npy` | Inertial measurement unit (right side) `[ax, ay, az, gx, gy, gz]`                         | 1000 / 90 Hz | `(N, 6)` |
| Intensity | `intensity_90fps.npy` | Eye-tracking video intensity signal                                                       | 90 Hz | `(N, 1)` |
| LBPTOP | `lbptop.npy` | 30 high-dimensional features for micro-expressions                                        | 9 Hz | `(N, 30)` |
| POV | `pov.mp4` | First-person point-of-view video                                                          | 10 FPS |`(N, 1408, 1408)` |
| PPG (ear) | `ppg_ear.npy`, `ppg_ear_90fps.npy` | Photoplethysmogram (ear)                                                                  | 256 / 90 Hz | `(N, 1)` |
| PPG (nose) | `ppg_nose.npy`, `ppg_nose_90fps.npy` | Photoplethysmogram (nose)                                                                 | 128 / 90 Hz | `(N,1)` |
| Pupils | `pupils_90fps.npy` | Pupil diameter `[left, right]`                                                            | 90 Hz | `(N, 2)` |
| Respiration rate (RR) | `rr.npy`, `rr_90fps.npy` | Respiration rate waveform                                                                 | 400 / 90 Hz | `(N, 1)` |
| Self-reports | `Session_A_<id>.csv`, `Session_B_<id>.csv` | Continuous affect (valence, arousal, dominance) and weighted Mikels' Wheel emotion labels | | |
| Webcam | `webcam.mp4` | External face webcam video                                                                | 60 FPS | `(N, 1280, 720)` |




## Data Availability & Quality Summary

The table below summarizes per-participant data availability and quality for each stream (✅ complete, ⚠️ partial, ❌ missing/unusable; ⚠️ entries include a short note describing the main issue).

| Participant | ECG | EDA | ET | IMU | POV | PPG <br> ear | PPG <br> nose | RR | Webcam | Comments |
|---|---|---|---|---|---|---|---|---|---|---|
|005|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|006|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|007|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|008|✅|✅|✅|✅|✅|✅|✅|✅|✅|ET glasses temporarily removed between Session A and B due to discomfort.|
|009|✅|✅|✅|✅|✅|✅|⚠️|✅|✅|EDA detached from finger briefly at minute 67. PPG (nose) complete but poor quality.|
|010|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|011|✅|✅|✅|✅|✅|✅|⚠️|✅|✅|PPG (nose) complete but poor quality.|
|012|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|013|✅|✅|✅|✅|✅|✅|⚠️|✅|✅|PPG (nose) complete but poor quality.|
|014|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|015|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|016|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|017|✅|⚠️|✅|✅|✅|✅|⚠️|✅|✅|EDA detached from finger at minutes 17-19 (3% total recording). PPG (nose) complete but poor quality.|
|018|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|019|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|021|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|022|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|023|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|024|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|025|✅|✅|✅|✅|✅|✅|✅|✅|⚠️|Webcam froze for all of Session B.|
|026|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|027|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|028|⚠️|✅|✅|✅|✅|✅|⚠️|✅|✅|ECG complete but poor quality after 40min (45% total recording). PPG (nose) complete but poor quality.|
|029|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|030|✅|✅|✅|✅|✅|✅|✅|✅|⚠️|Webcam froze for all but one task (Jenga) of Session B.|
|031|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|032|✅|✅|⚠️|✅|✅|✅|✅|✅|⚠️|ET glasses placed too low. Webcam recording 50s shorter.|
|033|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|034|✅|✅|✅|✅|✅|✅|⚠️|✅|✅|PPG (nose) complete but poor quality.|
|035|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|036|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|037|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|038|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|039|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|040|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|042|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|043|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|044|✅|✅|✅|✅|✅|✅|✅|✅|❌| Webcam recording did not work.|
|045|✅|✅|✅|✅|✅|✅|✅|✅|✅||
|046|✅|✅|✅|✅|✅|✅|✅|✅|✅||
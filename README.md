## egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-world Tasks (NeurIPS 2025)

[Matthias Jammot](https://matthiasjammot.com)\*, [Björn Braun](https://bjoernbraun.com/)\*,[Paul Streli](https://paulstreli.com), Rafael Wampfler, [Christian Holz](https://www.christianholz.net)<br/>
\* Equal contribution <br/>

[Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>

## :wave: egoEMOTION
*egoEMOTION* is the first dataset that couples egocentric visual and physiological signals with dense self-reports of emotion and personality across controlled and real-world scenarios. Participants completed emotion-elicitation tasks and naturalistic activities while self-reporting their affective state using the Circumplex Model and Mikels’ Wheel as well as their personality via the Big Five model. 

![Overview](assets/Figure_0_Overview.png)

## :movie_camera: egoEMOTION dataset

The *egoEMOTION* dataset includes over 50 hours of recordings from 43 participants, captured using Meta’s Project Aria glasses. Each session provides synchronized eye-tracking video, head-mounted photoplethysmography, inertial motion data, and physiological baselines for reference.

To download the dataset, please visit the following link: [egoEMOTION Dataset](https://polybox.ethz.ch/index.php/s/LSKXPye8rGJPHMj).

You have to sign a Data Transfer and Use Agreement (DTUA) form to agree to our terms of use. Please note that only members of an institution (e.g., a PI or professor) can sign this DTUA. After you have signed the DTUA, you will receive a download link via email. The dataset is around 380GB in size. The dataset is only for non-commercial, academic research purposes.

![Overview](assets/Figure_1_Sensors.png)

## :wrench: Setup

To create the environment that we used for our paper, simply run: 

```
conda env create -f environment.yml
```

## :file_folder: Code structure
Everything is running using the *source/main.py* file. The usage of *main.py* is explained in the file itself. Currently, all variables are set in the *main.py* file. We plan to release an updated version with config.yaml files to enable cleaner experiment setups.
At the moment, you can choose between different options for:

1) Signal processing-based approaches: 6 different feature selection methods, 3 scaling methods, and 7 different classifiers.
2) Deep learning-based approaches: 2 different architectures (one classical CNN and one transformer-based architecture).
3) The prediction target: continuous affect (arousal, valence, dominance), discrete emotions, or personality.
4) Choose the input modalities: ECG, EDA, RR, Pupils, IMU from the head, pixel intensity, Fisherfaces, micro-expressions, gaze, and nose PPG. You can flexibly combine these modalities.

You only have to run the feature calculation once at the beginning and can then specify to use the pre-calculated features. Depending on your setup, mind to specify how many processes to use for the feature calculation.

## :zap: Training and inference

## :bar_chart: Results for egoEMOTION
| **Benchmark** | **Model** | **Wearable Devices** | **Egocentric Glasses** | **All** |
|----------------|-----------|:--------------------:|:----------------------:|:-------:|
| **Continuous Affect** | Classical | 0.70 ± 0.14 | 0.74 ± 0.13 | 0.75 ± 0.13 |
|  | DCNN ([Ref](#)) | 0.63 ± 0.05 | 0.68 ± 0.05 | 0.68 ± 0.07 |
|  | WER ([Ref](#)) | 0.49 ± 0.21 | 0.65 ± 0.11 | 0.60 ± 0.16 |
| **Discrete Emotions** | Classical | 0.28 ± 0.08 | 0.52 ± 0.18 | 0.45 ± 0.17 |
|  | DCNN ([Ref](#)) | 0.12 ± 0.01 | 0.23 ± 0.03 | 0.22 ± 0.02 |
|  | WER ([Ref](#)) | 0.13 ± 0.02 | 0.22 ± 0.03 | 0.21 ± 0.04 |
| **Personality Traits** | Classical | 0.50 ± 0.48 | 0.57 ± 0.49 | 0.59 ± 0.49 |
|  | DCNN ([Ref](#)) | 0.43 ± 0.26 | 0.42 ± 0.20 | 0.41 ± 0.25 |
|  | WER ([Ref](#)) | 0.38 ± 0.28 | 0.47 ± 0.24 | 0.44 ± 0.28 |

> **Table:** Performance comparison between classical and deep learning approaches on the egoEMOTION dataset.


## :scroll: Citation
If you find our paper, code or dataset useful for your research, please cite our work.

```
TBD
```

## :heartbeat: :eye: egoPPG
Make sure to also check out [egoPPG](https://github.com/eth-siplab/egoPPG).
*egoPPG* is a novel vision task for egocentric vision systems to recover a person’s cardiac activity only from the eye tracking videos of egocentric systems to aid downstream vision tasks.
We demonstrate egoPPG’s downstream benefit for a key task on EgoExo4D, aexisting egocentric dataset for which we find PulseFormer’s estimates of HR to improve proficiency estimation by 14%.
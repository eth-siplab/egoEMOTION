import numpy as np

from scipy.stats import skew, kurtosis, iqr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from ..utils import pupil_filtering


def get_stats(data, as_dict=False):
    """
    Function defining the statistical measures considered for aggregation
    :return: (pd.DataFrame) data of aggregated featues with column 'num_samples'
    """
    data = np.asarray(data, dtype=np.float64)
    results = {
        'mean': np.nan,
        'median': np.nan,
        'std': np.nan,
        'min': np.nan,
        'q5': np.nan,                   # 5% quantil
        'max': np.nan,
        'q95': np.nan,                  # 95% quantil
        'range': np.nan,
        'iqrange': np.nan,              # inter quantil range
        'sum': np.nan,
        'energy': np.nan,
        'skewness': np.nan,
        'kurtosis': np.nan,
        'rms': np.nan,
        'lineintegral': np.nan,
    }

    if len(data) > 0:
        results['mean'] = np.mean(data)
        results['median'] = np.median(data)
        results['std'] = np.std(data)
        results['min'] = np.min(data)
        results['q5'] = np.quantile(data, 0.05)
        results['max'] = np.max(data)
        results['q95'] = np.quantile(data, 0.95)
        results['range'] = results['max'] - results['min']
        results['iqrange'] = iqr(data)
        results['sum'] = np.sum(data) #* scale_factor
        results['energy'] = np.sum([x**2 for x in data]) #* scale_factor
        results['skewness'] = skew(data)
        results['kurtosis'] = kurtosis(data)
        results['rms'] = np.sqrt(results['energy'] / len(data))
        results['lineintegral'] = np.abs(np.diff(data)).sum()

    if not as_dict:
        results = [v for k, v in results.items()]
    return results


def extract_imu_features(imu_splits):
    imu_features_splits = []
    for imu_split in imu_splits:
        imu_signal = np.sqrt(np.sum(np.square(np.vstack((imu_split[:, 0], imu_split[:, 1], imu_split[:, 2]))), axis=0))
        im_features_temp = get_stats(imu_signal, as_dict=False)
        imu_features_splits.append(im_features_temp)

    return imu_features_splits


def extract_lbptop_features(lbptop_splits):
    lbptop_features_splits = []
    for lbptop_split in lbptop_splits:
        lbptop_features_splits.append(np.mean(lbptop_split, axis=0))

    return lbptop_features_splits


def extract_gaze_features(gaze_splits):
    gaze_features_splits = []
    for gaze_split in gaze_splits:
        yaw, pitch = gaze_split[:, 0], gaze_split[:, 1]
        yaw_features, pitch_features = get_stats(yaw, as_dict=False), get_stats(pitch, as_dict=False)
        gaze_features_splits.append(yaw_features + pitch_features)

    return gaze_features_splits


def extract_pupils_features(pupil_splits, fs, cfg_dataset, participant):
    pupils_features_splits = []
    threshold = 250
    cutoff = 0.5
    for pupil_split in pupil_splits:
        if participant in cfg_dataset['pupils_side_to_use']:
            pupil_extracted = pupil_split[:, cfg_dataset['pupils_side_to_use'][participant]]
            pupil_filtered = pupil_filtering(pupil_extracted.copy(), fs, threshold=threshold, cutoff=cutoff)
            pupil_features_temp = get_stats(pupil_filtered, as_dict=False)
        else:
            left_pupil = pupil_split[:, 0]
            right_pupil = pupil_split[:, 1]
            n_cutoffs_left = np.sum(left_pupil > threshold)
            n_cutoffs_right = np.sum(right_pupil > threshold)
            if n_cutoffs_left > n_cutoffs_right:
                pupil_filtered = pupil_filtering(right_pupil.copy(), fs, threshold=threshold, cutoff=cutoff)
            else:
                pupil_filtered = pupil_filtering(left_pupil.copy(), fs, threshold=threshold, cutoff=cutoff)
            pupil_features_temp = get_stats(pupil_filtered, as_dict=False)
        pupils_features_splits.append(pupil_features_temp)

    return pupils_features_splits

def extract_intensity_features(intensity_splits):
    intensity_features_splits = []
    for intensity_split in intensity_splits:
        intensity_stats = get_stats(-intensity_split, as_dict=False)
        intensity_features_splits.append(np.mean(np.asarray([intensity_stats]), axis=0))

    return intensity_features_splits


def get_y_frames(fisherfaces_splits_train, emo_labels, label_category):
    y_frames = []
    for participant_clip_frames, participant_labels in zip(fisherfaces_splits_train, emo_labels):
        for i_clip in range(len(participant_clip_frames)):
            clip_frames = participant_clip_frames[i_clip]
            for frame in clip_frames:
                if label_category == 'personality':
                    y_frames.append(participant_labels)
                else:
                    y_frames.append(participant_labels[i_clip])
    return y_frames


def fit_PCA_model(fisherfaces_splits_train):
    X_frames = []
    for participant_clip_frames in fisherfaces_splits_train:
        for i_clip in range(len(participant_clip_frames)):
            clip_frames = participant_clip_frames[i_clip]
            for frame in clip_frames:
                X_frames.append(frame.flatten().astype("float32") / 255.0)

    X_frames = np.vstack(X_frames)

    # PCA and LDA
    pca = PCA(n_components=256)
    X_pca = pca.fit_transform(X_frames)

    return X_pca, pca


def extract_fisherfaces_features(fisherfaces_splits, pca, lda):
    feats_all = []
    for clip_frames in fisherfaces_splits:      # clip_frames = list of frames
        fish_vecs = []
        for fr in clip_frames:
            vec   = fr.flatten().astype("float32") / 255.0
            fish_v = lda.transform(pca.transform(vec.reshape(1, -1))).ravel()                      # (d,)
            fish_vecs.append(fish_v)
        fish_vecs = np.vstack(fish_vecs)            # (n_sub, d)
        clip_feat = get_stats(fish_vecs, as_dict=False)

        feats_all.append(np.hstack(clip_feat).astype("float32"))

    return feats_all


def extract_fisherfaces_features_overall(participant_test,
                                         session,
                                         cfg_dataset,
                                         cfg_features,
                                         df_all_labels,
                                         fisherfaces_labels,
                                         df_personality_means,
                                         fisherfaces_frames_splits):
    fisherfaces_frames_splits_train = []
    participants_train = [p for p in cfg_dataset['participants'] if p != participant_test]
    for participant in participants_train:
        fisherfaces_frames_splits_train.append(fisherfaces_frames_splits[participant])

    # Get test clips
    frames_splits_test = fisherfaces_frames_splits[participant_test]

    # Fit PCA
    print(f"Starting Fisherfaces feature extraction for participant {participant_test}")
    X_pca, pca = fit_PCA_model(fisherfaces_frames_splits_train)

    # Fit LDA and extract features
    fisherfaces_features_all = []
    for label in fisherfaces_labels:
        labels_temp = []
        print(f"Starting fisherfaces features for {participant_test} for {label}!")
        for participant in participants_train:
            if label == 'strongest_emotion':
                df_labels_copy = df_all_labels.copy()
                df_labels_copy = df_labels_copy[["participant"] + cfg_dataset['emotion_domains']]
                labels_temp.append(df_labels_copy[df_labels_copy["participant"] == participant].iloc[:, 1:].values.argmax(axis=1))
            elif label in cfg_dataset['personality_domains']:
                df_personality_means_copy = df_personality_means.copy()
                labels_temp.append(df_personality_means_copy.loc[participant, label])
            else:
                labels_temp.append(df_all_labels[df_all_labels['participant']==participant][label].values.astype(float))

        label_category = None
        if label in ['Liking', 'Familiarity']:
            thr = 2
            labels_temp = (np.asarray(np.asarray(labels_temp) >= thr)).astype(int)
        elif label in ['Arousal', 'Valence', 'Dominance']:
            thr = np.median(np.asarray(labels_temp).flatten())
            labels_temp = (np.asarray(labels_temp >= thr)).astype(int)
        elif label in cfg_dataset['emotion_domains']:
            labels_temp = (np.asarray(np.asarray(labels_temp) >= cfg_features['thr_emo_binary'])).astype(int)
        elif label in cfg_dataset['personality_domains']:
            thr = np.median(labels_temp)
            labels_temp = (np.asarray(labels_temp) >= thr).astype(int)
            label_category = 'personality'

        y_frames = get_y_frames(fisherfaces_frames_splits_train, labels_temp, label_category)

        # Fit LDA and extract features
        #  lda = LDA()
        lda = LDA(solver="eigen", shrinkage="auto")
        lda.fit(X_pca, y_frames)
        fisher_features = extract_fisherfaces_features(frames_splits_test, pca, lda)
        del lda

        np.save(f"{cfg_features['preprocessed_data_path']}/{participant_test}/"
                f"fisherfaces_features_session{session}_{cfg_features['chunk_len']}_{label}.npy",
                np.asarray(fisher_features),
                allow_pickle=True)

        fisherfaces_features_all.append(fisher_features)

        print(f"Fisherfaces features for {participant_test} extracted for {label}!")

    return fisherfaces_features_all
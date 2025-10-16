import neurokit2 as nk
import numpy as np
import os

from sp_feature_calculation_helper import filter_raw_modality
from sp_utils import butter_lowpass_filter, butter_highpass_filter, getBand_Power, detrend, getfreqs_power
from sp_utils import pupil_filtering

from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis, iqr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from concurrent.futures import ProcessPoolExecutor, as_completed


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
        #results['ratio'] = math.log(max(data) / statistics.median(data))

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


def extract_pupils_features(pupil_splits, fs, cfg, participant):
    pupils_features_splits = []
    threshold = 250
    cutoff = 0.5
    for pupil_split in pupil_splits:
        if participant in cfg['pupils_side_to_use']:
            pupil_extracted = pupil_split[:, cfg['pupils_side_to_use'][participant]]
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


def extract_rr_features(rr_splits, fs):
    """Return list of respiration-feature vectors (one per split)."""
    rr_features_splits = []
    band_edges = [(0.0 + i * 0.3, 0.3 + i * 0.3) for i in range(8)]
    for rr_split in rr_splits:
        rr_signal = rr_split

        # 1) NeuroKit processing: cycles, amplitude, rate
        signals_rsp, info_rsp = nk.rsp_process(rr_signal, sampling_rate=fs)
        sig_filt = signals_rsp["RSP_Clean"].values

        range_signal = sig_filt.max() - sig_filt.min()  # range of signal
        mean_deriv = np.mean(np.abs(np.diff(sig_filt)))  # mean of derivative

        # ----------------------------------------------------------------
        # 2) PEAKS (breaths) to get depth & rate -------------------------
        # ----------------------------------------------------------------
        peaks, _ = find_peaks(sig_filt, distance=fs * 0.8)   # ≥0.8 s apart
        # average PEAK-to-PEAK amplitude (depth)
        if len(peaks) > 1:
            depth = np.mean(np.diff(sig_filt[peaks]))
        else:
            depth = 0.0

        # breathing rate (Hz → breaths/min) via peak intervals
        if len(peaks) > 1:
            br_hz = 1.0 / np.mean(np.diff(peaks) / fs)
        else:
            freqs, psd = welch(sig_filt, fs=fs, nperseg=len(sig_filt))
            br_hz = freqs[np.argmax(psd[(freqs >= 0.05) & (freqs <= 1.0)])]
        breathing_rate_bpm = br_hz * 60.0

        # ----------------------------------------------------------------
        # 3) PSD & spectral features -------------------------------------
        # ----------------------------------------------------------------
        freqs, psd = getfreqs_power(sig_filt, fs=fs, nperseg=len(sig_filt), scaling='density')

        # band energy ratio (log-difference) 0.05–0.25 Hz vs 0.25–0.5 Hz
        low_E  = getBand_Power(freqs, psd, 0.05, 0.25)
        high_E = getBand_Power(freqs, psd, 0.25, 0.50)
        ber = np.log(low_E + 1e-9) - np.log(high_E + 1e-9)

        # breathing rhythm – spectral centroid (Hz)
        centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-9)

        # eight sub-band absolute powers (0–2.4 Hz by 0.3 Hz)
        subband_powers = [getBand_Power(freqs, psd, lo, hi) for lo, hi in band_edges]

        rr_features_temp = [ber, range_signal, mean_deriv, centroid, breathing_rate_bpm, depth] + subband_powers                      #  7-14  eight 0.3-Hz band powers
        rr_features_splits.append(rr_features_temp)

    return rr_features_splits


def extract_ecg_ppg_features(ecg_ppg_splits, fs, signal_type):
    ecg_ppg_features_splits = []
    for ecg_ppg_split in ecg_ppg_splits:
        signal_split = ecg_ppg_split

        # Filter ECG and get R-peaks
        if signal_type == 'ecg':
            signal_split = butter_highpass_filter(signal_split, 1.0, fs)
            signals_nk, info_nk = nk.ecg_process(signal_split, sampling_rate=fs)
            peaks = info_nk['ECG_R_Peaks']
        elif signal_type == 'ppg':
            signal_split = np.max(signal_split) - signal_split
            signal_split = signal_split - min(signal_split)
            signal_split = signal_split / max(signal_split)
            signals_nk, info_nk = nk.ppg_process(signal_split, sampling_rate=fs)
            peaks = info_nk['PPG_Peaks']
        else:
            raise ValueError

        # Get 60 spectral power in the range of 0 - 6 Hz
        freqs, power = getfreqs_power(signal_split, fs=fs, nperseg=signal_split.size, scaling='spectrum')
        power_0_6 = []
        for i in range(60):
            power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))

        # Calculate IBIs
        IBI = np.array([])
        for i in range(len(peaks) - 1):
            IBI = np.append(IBI, (peaks[i + 1] - peaks[i]) / fs)

        # Calculate heart rate
        heart_rate = np.array([])
        for i in range(len(IBI)):
            append_value = 60.0 / IBI[i] if IBI[i] != 0 else 0
            heart_rate = np.append(heart_rate, append_value)

        # IBI features
        mean_IBI = np.mean(IBI)
        rms_IBI = np.sqrt(np.mean(np.square(IBI)))
        std_IBI = np.std(IBI)
        skew_IBI = skew(IBI)
        kurt_IBI = kurtosis(IBI)
        per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
        per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)

        # IBI spectral power features
        freqs_, power_ = getfreqs_power(IBI, fs=1.0 / mean_IBI, nperseg=IBI.size, scaling='spectrum')
        power_000_004 = getBand_Power(freqs_, power_, lower=0., upper=0.04)  # VLF
        power_004_015 = getBand_Power(freqs_, power_, lower=0.04, upper=0.15)  # LF
        power_015_040 = getBand_Power(freqs_, power_, lower=0.15, upper=0.50)  # HF
        power_000_040 = getBand_Power(freqs_, power_, lower=0., upper=0.50)  # TF

        # IBI spectral power ratios
        LF_HF = power_004_015 / power_015_040
        LF_TF = power_004_015 / power_000_040
        HF_TF = power_015_040 / power_000_040
        nLF = power_004_015 / (power_000_040 - power_000_004)
        nHF = power_015_040 / (power_000_040 - power_000_004)

        # Heart rate features
        mean_heart_rate = np.mean(heart_rate)
        std_heart_rate = np.std(heart_rate)
        skew_heart_rate = skew(heart_rate)
        kurt_heart_rate = kurtosis(heart_rate)
        per_above_heart_rate = float(heart_rate[heart_rate >
                                                mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
        per_below_heart_rate = float(heart_rate[heart_rate <
                                                mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)

        ecg_ppg_features_temp = ([rms_IBI, mean_IBI] + power_0_6 +
                             [power_000_004, power_004_015, power_015_040, mean_heart_rate, std_heart_rate,
                              skew_heart_rate, kurt_heart_rate, per_above_heart_rate, per_below_heart_rate, std_IBI,
                              skew_IBI, kurt_IBI, per_above_IBI, per_below_IBI, LF_HF, LF_TF, HF_TF, nLF, nHF])
        ecg_ppg_features_splits.append(ecg_ppg_features_temp)

    return ecg_ppg_features_splits


def extract_eda_features(eda_splits, fs):
    eda_features_splits = []
    for eda_split in eda_splits:
        signal_split = eda_split
        der_signals = np.gradient(signal_split)

        con_signals = 1.0 / signal_split
        nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

        # Mean signal, mean derivative, mean negative derivative, negative derivative proportion
        mean = np.mean(signal_split)
        der_mean = np.mean(der_signals)
        neg_der_mean = np.mean(der_signals[der_signals < 0])
        neg_der_pro = float(der_signals[der_signals < 0].size) / float(der_signals.size)

        # Number of local minima of signal
        local_min = 0
        for i in range(signal_split.shape[0] - 1):
            if i == 0:
                continue
            if signal_split[i - 1] > signal_split[i] and signal_split[i] < signal_split[i + 1]:
                local_min += 1

        # Using SC calculates rising time
        det_nor_signals, trend = detrend(nor_con_signals)
        lp_det_nor_signals = butter_lowpass_filter(det_nor_signals, 0.5, fs)
        der_lp_det_nor_signals = np.gradient(lp_det_nor_signals)

        # Calculate rising time of signal
        rising_time = 0
        rising_cnt = 0
        for i in range(der_lp_det_nor_signals.size - 1):
            if der_lp_det_nor_signals[i] > 0:
                rising_time += 1
                if der_lp_det_nor_signals[i + 1] < 0:
                    rising_cnt += 1
        avg_rising_time = rising_time * (1. / fs) / rising_cnt

        # 10 spectral power in the range of 0-2.4 Hz
        freqs, power = getfreqs_power(signal_split, fs=fs, nperseg=signal_split.size, scaling='spectrum')
        power_0_24 = []
        for i in range(21):
            power_0_24.append(getBand_Power(freqs, power, lower=0 + (i * 0.8 / 7), upper=0.1 + (i * 0.8 / 7)))

        # Calculate low-pass filtered signal
        SCSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.2, fs))
        SCVSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.08, fs))

        # Calculate zero-crossing rate and mean peak value
        zero_cross_SCSR = 0
        zero_cross_SCVSR = 0
        peaks_cnt_SCSR = 0
        peaks_cnt_SCVSR = 0
        peaks_value_SCSR = 0.
        peaks_value_SCVSR = 0.

        zc_idx_SCSR = np.array([], int)  # must be int, otherwise it will be float
        zc_idx_SCVSR = np.array([], int)
        for i in range(nor_con_signals.size - 1):
            if SCSR[i] * next((j for j in SCSR[i + 1:] if j != 0), 0) < 0:
                zero_cross_SCSR += 1
                zc_idx_SCSR = np.append(zc_idx_SCSR, i + 1)
            if SCVSR[i] * next((j for j in SCVSR[i + 1:] if j != 0), 0) < 0:
                zero_cross_SCVSR += 1
                zc_idx_SCVSR = np.append(zc_idx_SCVSR, i)

        for i in range(zc_idx_SCSR.size - 1):
            peaks_value_SCSR += np.absolute(SCSR[zc_idx_SCSR[i]:zc_idx_SCSR[i + 1]]).max()
            peaks_cnt_SCSR += 1
        for i in range(zc_idx_SCVSR.size - 1):
            peaks_value_SCVSR += np.absolute(SCVSR[zc_idx_SCVSR[i]:zc_idx_SCVSR[i + 1]]).max()
            peaks_cnt_SCVSR += 1

        zcr_SCSR = zero_cross_SCSR / (nor_con_signals.size / fs)
        zcr_SCVSR = zero_cross_SCVSR / (nor_con_signals.size / fs)

        mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
        mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

        eda_features_temp = [mean, der_mean, neg_der_mean, neg_der_pro, local_min, avg_rising_time] + \
                             power_0_24 + [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR]
        eda_features_splits.append(eda_features_temp)

    return eda_features_splits


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


def extract_fisherfaces_features_overall(participant_test, participants, cfg, session_label, dfs_all, chunk_len,
                                         df_all_labels, fisherfaces_labels, thr_emo_binary, df_personality_means,
                                         fisherfaces_frames_splits):
    fisherfaces_frames_splits_train = []
    participants_train = [p for p in participants if p != participant_test]
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
                df_labels_copy = df_labels_copy[["participant"] + cfg['emotion_domains']]
                labels_temp.append(df_labels_copy[df_labels_copy["participant"] == participant].iloc[:, 1:].values.argmax(axis=1))
            elif label in cfg['personality_domains']:
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
        elif label in cfg['emotion_domains']:
            labels_temp = (np.asarray(np.asarray(labels_temp) >= thr_emo_binary)).astype(int)
        elif label in cfg['personality_domains']:
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

        np.save(f"{cfg['preprocessed_data_path']}/{participant_test}/"
                f"fisherfaces_features_session{session_label}_{chunk_len}_{label}.npy", np.asarray(fisher_features),
                allow_pickle=True)

        fisherfaces_features_all.append(fisher_features)

        print(f"Fisherfaces features for {participant_test} extracted for {label}!")

    return fisherfaces_features_all


def get_data_chunks(data, start, end, chunk_len, fs_data):
    relevant_data = data[start:end]
    n_chunks_task = 0
    if chunk_len == 'all':
        data_chunks = [relevant_data]
        n_chunks_task = 1
    elif isinstance(chunk_len, int):
        samples_per_chunk = chunk_len * fs_data
        data_chunks = []
        for i in range(0, len(relevant_data), samples_per_chunk):
            chunk = relevant_data[i:i + samples_per_chunk]
            if len(chunk) == samples_per_chunk:
                data_chunks.append(chunk)
                n_chunks_task += 1
            else:
                break
    elif isinstance(chunk_len, str) and chunk_len[:4] == 'last':
        secs = int(chunk_len.rsplit('_', 1)[-1])
        last_n = secs * fs_data  # number of samples
        if last_n > len(relevant_data):
            raise ValueError(f"Cannot take last {secs}s ({last_n} samples) from only {len(relevant_data)} samples")
        data_chunks = [relevant_data[-last_n:]]
        n_chunks_task = 1
    else:
        raise ValueError('Invalid chunk_len!')

    if len(data_chunks) == 0:
        raise ValueError('No data chunks created!')

    return data_chunks, n_chunks_task


def get_session_splits(data, dfs_all, fs_et, fs_data, session_label, task_times, participant, chunk_len, subsample=None):
    data_session = []
    n_chunks_session = []
    if session_label == 'A':
        for task in dfs_all[session_label][dfs_all[session_label]['participant'] == participant].loc[:, 'Video Emotion']:
            start = int((task_times[participant][f'video_{task}'][0] - task_times[participant][f'session_A'][0]) / fs_et * fs_data)
            end = int((task_times[participant][f'video_{task}'][1] - task_times[participant][f'session_A'][0]) / fs_et * fs_data)
            data_chunks, n_chunks_task = get_data_chunks(data, start, end, chunk_len, fs_data)
            if subsample is not None:
                data_chunks = [chunk[::subsample] for chunk in data_chunks]
            data_session.extend(data_chunks)
            n_chunks_session.append(n_chunks_task)
    elif session_label == 'B':
        for task in dfs_all[session_label][dfs_all[session_label]['participant'] == participant].loc[:, 'Activity Name']:
            start = int((task_times[participant][f'{task.lower()}'][0] - task_times[participant][f'session_A'][0]) / fs_et * fs_data)
            end = int((task_times[participant][f'{task.lower()}'][1] - task_times[participant][f'session_A'][0]) / fs_et * fs_data)
            data_chunks, n_chunks_task = get_data_chunks(data, start, end, chunk_len, fs_data)
            if subsample is not None:
                data_chunks = [chunk[::subsample] for chunk in data_chunks]
            data_session.extend(data_chunks)
            n_chunks_session.append(n_chunks_task)
    else:
        raise ValueError('Invalid session_label!')
    return data_session, n_chunks_session


def split_data(data, fs_et, fs_data, session_label, task_times, dfs_all, participant, chunk_len, subsample=None):
    data_out = []
    n_chunks = []
    if session_label in ['A', 'B']:
        data_out, n_chunks_session = get_session_splits(data, dfs_all, fs_et, fs_data, session_label, task_times,
                                                        participant, chunk_len, subsample=subsample)
        n_chunks = n_chunks_session
    elif session_label == 'A_and_B':
        data_out_a, n_chunks_session_a = get_session_splits(data, dfs_all, fs_et, fs_data, 'A', task_times, participant,
                                                          chunk_len, subsample=subsample)
        data_out_b, n_chunks_session_b = get_session_splits(data, dfs_all, fs_et, fs_data, 'B', task_times, participant,
                                                          chunk_len, subsample=subsample)
        data_out = data_out_a + data_out_b
        n_chunks = n_chunks_session_a + n_chunks_session_b
    return data_out, n_chunks


def get_frames(cfg, participant):
    frames = np.load(f"{cfg['original_data_path']}/{participant}/et.npy", allow_pickle=True)
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, axis=3)
    return frames


def _extract_single(participant, cfg, feature_modalities, do_new_feature_extraction, session_label, dfs_all, chunk_len,
                    participants, df_all_labels, fisherfaces_labels, do_new_fisherface_extraction, thr_emo_binary,
                    df_personality_means, fisherfaces_frames_splits, use_dl_method):
    """
    Extract features for exactly one participant.
    Returns (participant, features_matrix).
    """
    tasks_times = np.load(f"{cfg['original_data_path']}/task_times.npy", allow_pickle=True).item()

    if not os.path.exists(f"{cfg['preprocessed_data_path']}/{participant}"):
        os.makedirs(f"{cfg['preprocessed_data_path']}/{participant}")

    extracted_features_p = []
    feature_cols_per_mod = {}
    features_cols_n = 0

    if 'fisherfaces' in feature_modalities:
        col_fisher_n = 0
        if do_new_fisherface_extraction:
            fisherface_features_all = extract_fisherfaces_features_overall(participant, participants, cfg, session_label,
                                                                           dfs_all, chunk_len, df_all_labels,
                                                                           fisherfaces_labels, thr_emo_binary,
                                                                           df_personality_means,
                                                                           fisherfaces_frames_splits)
            for fisherface_feature_label in fisherface_features_all:
                extracted_features_p.append(fisherface_feature_label)
                col_fisher_n += len(fisherface_feature_label)
        else:
            for fisherface_label in fisherfaces_labels:
                fisherfaces_features = np.load(
                    f"{cfg['preprocessed_data_path']}/{participant}/"
                    f"fisherfaces_features_session{session_label}_{chunk_len}_{fisherface_label}.npy",
                    allow_pickle=True)
                extracted_features_p.append(fisherfaces_features)
                col_fisher_n += fisherfaces_features.shape[1]
        feature_cols_per_mod['fisherfaces'] = [0, col_fisher_n]
        features_cols_n = col_fisher_n

        # Has to be changed: Just load some features to get n_chunks
        modality_raw = np.load(f"{cfg['original_data_path']}/{participant}/rr.npy", allow_pickle=True)
        _, n_chunks = split_data(modality_raw, cfg['fs_all']['et'], cfg['fs_all']['rr'],
                                               session_label, tasks_times, dfs_all, participant, chunk_len)

    # Load data and extract features for each modality
    for modality in feature_modalities:
        if participant in cfg['sensors_missing'] and modality in cfg['sensors_missing'][participant]:
            continue

        if do_new_feature_extraction:
            if modality == 'fisherfaces':
                continue

            if use_dl_method:
                modality_raw = np.load(f"{cfg['original_data_path']}/{participant}/{modality}_90fps.npy",
                                       allow_pickle=True)
                modality_filtered = filter_raw_modality(modality_raw, modality, cfg['fs_all']['et'], cfg, participant)
                modality_splits, n_chunks = split_data(modality_filtered, cfg['fs_all']['et'], cfg['fs_all']['et'],
                                                       session_label, tasks_times, dfs_all, participant, chunk_len)
                try:
                    modality_splits = np.asarray(modality_splits)
                except:
                    raise ValueError("Set chunk len to X seconds.")
                if len(modality_splits.shape) == 2:
                    modality_splits = np.expand_dims(modality_splits, 1)
                elif len(modality_splits.shape) == 3:
                    modality_splits = np.transpose(modality_splits, (0, 2, 1))
                features_modality = modality_splits
                save_path_temp = f"{cfg['preprocessed_data_path']}/{participant}/CL_{chunk_len}"
                if not os.path.exists(save_path_temp):
                    os.makedirs(save_path_temp)
                np.save(f"{save_path_temp}/{modality}_raw_session{session_label}.npy", features_modality,
                        allow_pickle=True)
                np.save(f"{save_path_temp}/{modality}_n_chunks{session_label}.npy", n_chunks,
                        allow_pickle=True)
            else:
                modality_raw = np.load(f"{cfg['original_data_path']}/{participant}/{modality}.npy", allow_pickle=True)
                modality_splits, n_chunks = split_data(modality_raw, cfg['fs_all']['et'], cfg['fs_all'][modality],
                                                       session_label, tasks_times, dfs_all, participant, chunk_len)
                if modality == 'ecg':
                    features_modality = extract_ecg_ppg_features(modality_splits, cfg['fs_all'][modality], 'ecg')
                elif modality == 'ppg_nose':
                    features_modality = extract_ecg_ppg_features(modality_splits, cfg['fs_all'][modality], 'ppg')
                elif modality == 'eda':
                    features_modality = extract_eda_features(modality_splits, cfg['fs_all'][modality])
                elif modality == 'rr':
                    features_modality = extract_rr_features(modality_splits, cfg['fs_all'][modality])
                elif modality == 'imu_right':
                    features_modality = extract_imu_features(modality_splits)
                elif modality == 'gaze':
                    features_modality = extract_gaze_features(modality_splits)
                elif modality == 'pupils':
                    features_modality = extract_pupils_features(modality_splits, cfg['fs_all'][modality], cfg, participant)
                elif modality == 'intensity':
                    features_modality = extract_intensity_features(modality_splits)
                elif modality == 'lbptop':
                    features_modality = extract_lbptop_features(modality_splits)
                else:
                    raise ValueError(f"Unknown modality: {modality}")

                np.save(f"{cfg['preprocessed_data_path']}/{participant}/"
                        f"{modality}_features_session{session_label}_{chunk_len}.npy", features_modality,
                        allow_pickle=True)
                np.save(f"{cfg['preprocessed_data_path']}/{participant}/"
                        f"{modality}_n_chunks{session_label}_{chunk_len}.npy", features_modality,
                        allow_pickle=True)
            extracted_features_p.append(features_modality)
        else:
            if modality == 'fisherfaces':
                continue

            if use_dl_method:
                load_path = f"{cfg['preprocessed_data_path']}/{participant}/CL_{chunk_len}"
                features_modality = np.load(f"{load_path}/{modality}_raw_session{session_label}.npy", allow_pickle=True)
                n_chunks = np.load(f"{load_path}/{modality}_n_chunks{session_label}.npy", allow_pickle=True)
            else:
                features_modality = np.load(f"{cfg['preprocessed_data_path']}/{participant}/"
                                            f"{modality}_features_session{session_label}_{chunk_len}.npy",
                                            allow_pickle=True)
                n_chunks = np.load(f"{cfg['preprocessed_data_path']}/{participant}/"
                                            f"{modality}_n_chunks{session_label}_{chunk_len}.npy",
                                            allow_pickle=True)
            extracted_features_p.append(features_modality)
        feature_cols_per_mod[modality] = [features_cols_n, features_cols_n + np.asarray(features_modality).shape[1]]
        features_cols_n = features_cols_n + np.asarray(features_modality).shape[1]

    all_feats = np.hstack(extracted_features_p)
    return participant, all_feats, feature_cols_per_mod, n_chunks


def _load_frames_single(cfg, participant, session_label, tasks_times, dfs_all, chunk_len, subsample):
    frames = get_frames(cfg, participant)
    frames_splits = split_data(frames, cfg['fs_all']['et'], cfg['fs_all']['et'], session_label, tasks_times, dfs_all,
                               participant, chunk_len, subsample=subsample)
    return frames_splits[0], participant


def extract_features(cfg, participants, feature_modalities, do_new_feature_extraction, session_label, dfs_all,
                     chunk_len, scaler_name, df_all_labels, fisherfaces_labels, do_new_fisherface_extraction,
                     thr_emo_binary, df_personality_means, use_dl_method, n_jobs=None):
    """
    Parallel version of your extract_features.
    n_jobs = None  -> use as many workers as CPUs
    n_jobs = 1     -> fully serial
    n_jobs = k>1   -> k workers
    """
    extracted_features_all, feature_cols_per_mod_out, n_chunks_all = {}, {}, {}

    if 'fisherfaces' in feature_modalities and do_new_fisherface_extraction:
        subsample = 10  # as training otherwise takes too long
        fisherfaces_frames_splits = {}
        tasks_times = np.load(f"{cfg['original_data_path']}/task_times.npy", allow_pickle=True).item()
        with ProcessPoolExecutor(max_workers=len(participants)) as exe:
            futures = {exe.submit(_load_frames_single, cfg, p, session_label, tasks_times, dfs_all, chunk_len,
                                  subsample): p for p in participants}
            for fut in as_completed(futures):
                frames_splits, participant = fut.result()
                fisherfaces_frames_splits[participant] = frames_splits
    else:
        fisherfaces_frames_splits = {}

    if do_new_fisherface_extraction:
        n_jobs=1

    if n_jobs == 1:
        for participant in participants:
            participant, feats, feature_cols_per_mod, n_chunks = (
                _extract_single(participant, cfg, feature_modalities, do_new_feature_extraction, session_label, dfs_all,
                                chunk_len, participants, df_all_labels, fisherfaces_labels,
                                do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                                fisherfaces_frames_splits, use_dl_method))
            extracted_features_all[participant] = feats
            n_chunks_all[participant] = n_chunks
            if participant not in cfg['sensors_missing']:
                features_cols_per_mod_out = feature_cols_per_mod
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            futures = {exe.submit(_extract_single, p, cfg, feature_modalities, do_new_feature_extraction,
                       session_label, dfs_all, chunk_len, participants, df_all_labels, fisherfaces_labels,
                                  do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                                  fisherfaces_frames_splits, use_dl_method):
                           p for p in participants}
            for fut in as_completed(futures):
                participant, feats, feature_cols_per_mod, n_chunks = fut.result()
                extracted_features_all[participant] = feats
                n_chunks_all[participant] = n_chunks
                if participant not in cfg['sensors_missing']:
                    features_cols_per_mod_out = feature_cols_per_mod

    return extracted_features_all, features_cols_per_mod_out, n_chunks_all

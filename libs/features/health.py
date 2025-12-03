import neurokit2 as nk
import numpy as np

from scipy.signal import welch, find_peaks
from scipy.stats import skew, kurtosis, iqr

from ..utils import butter_lowpass_filter, butter_highpass_filter, getBand_Power, detrend, getfreqs_power


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
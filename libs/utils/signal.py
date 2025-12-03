import numpy as np
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch, find_peaks

from .general import normalize


def fisher_idx(num, features, labels):
    ''' Get idx sorted by fisher linear discriminant '''
    labels = np.array(labels)
    labels0 = np.where(labels < 1)
    labels1 = np.where(labels > 0)
    labels0 = np.array(labels0).flatten()
    labels1 = np.array(labels1).flatten()
    features0 = np.delete(features, labels1, axis=0)
    features1 = np.delete(features, labels0, axis=0)
    mean_features0 = np.mean(features0, axis=0)
    mean_features1 = np.mean(features1, axis=0)
    std_features0 = np.std(features0, axis=0)
    std_features1 = np.std(features1, axis=0)
    std_sum = std_features1**2 + std_features0**2
    fisher = (abs(mean_features0 - mean_features1)) / std_sum
    fisher_sorted = np.argsort(np.array(fisher))  # sort the fisher from small to large
    sorted_feature_idx = fisher_sorted[::-1]  # arrange from large to small
    return sorted_feature_idx[:num]


def butter_highpass_filter(data, cutoff, fs, order=5):
    ''' Highpass filter '''
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        [b, a] = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    ''' Lowpass filter '''
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def _band_power(freqs, psd, f_low, f_high):
    """Return absolute power between f_low & f_high (Hz)."""
    idx = np.logical_and(freqs >= f_low, freqs < f_high)
    return np.trapz(psd[idx], freqs[idx])


def getfreqs_power(signals, fs, nperseg, scaling):
    ''' Calculate power density or power spectrum density '''
    if scaling == "density":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='density')
        return freqs, power
    elif scaling == "spectrum":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='spectrum')
        return freqs, power
    else:
        return 0, 0


def getBand_Power(freqs, power, lower, upper):
    ''' Sum band power within desired frequency range '''
    low_idx = np.array(np.where(freqs <= lower)).flatten()
    up_idx = np.array(np.where(freqs > upper)).flatten()
    if len(up_idx) == 0:
        band_power = np.sum(power[low_idx[-1]:])
    else:
        band_power = np.sum(power[low_idx[-1]:up_idx[0]])

    return band_power


def detrend(data):
    ''' Detrend data with EMD '''
    emd = EMD()
    imfs = emd(data)
    detrended = np.sum(imfs[:int(imfs.shape[0] / 2)], axis=0)
    trend = np.sum(imfs[int(imfs.shape[0] / 2):], axis=0)

    return detrended, trend


def peak_artifact_removal(data, range_remove=15):
    data = normalize(data, 'zero_one')
    peaks = find_peaks(data, height=0.1 + np.mean(data))[0]
    for peak in peaks:
        if (peak - range_remove) < 0:
            data[:peak + range_remove] = np.repeat(data[peak+range_remove], peak+range_remove)
        elif (peak + range_remove) > data.shape[0]:
            data[peak - range_remove:] = np.repeat(data[peak-range_remove], range_remove + data.shape[0]-peak)
        else:
            data[peak - range_remove:peak + range_remove] = np.repeat(data[peak-range_remove], range_remove * 2)
    return data


def pupil_filtering(pupil_data, fs, threshold=250, cutoff=0.5):
    ''' Filter pupil diameter '''
    pupil_data[pupil_data > threshold] = threshold

    # Filtering option 1: Butterworth filter
    [b, a] = butter(4, cutoff, fs=fs, btype='lowpass')
    pupil_data = filtfilt(b, a, pupil_data)

    return pupil_data
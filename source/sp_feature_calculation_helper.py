import neurokit2 as nk
import numpy as np

from sp_utils import butter_highpass_filter, pupil_filtering


def filter_raw_modality(raw_modality, modality, fs, cfg, participant):
    """
    Filter the raw modality based on the specified modality.

    Args:
        raw_modality (list): The raw modality data.
        modality (str): The type of modality to filter by.

    Returns:
        list: Filtered raw modality data.
    """
    if modality == "ecg":
        return butter_highpass_filter(raw_modality, 1.0, fs)
    elif modality == "ppg_nose":
        raw_modality = np.max(raw_modality) - raw_modality
        raw_modality = raw_modality - min(raw_modality)
        raw_modality = raw_modality / max(raw_modality)
        signals_nk, info_nk = nk.ppg_process(raw_modality, sampling_rate=fs)
        return signals_nk["PPG_Clean"].values
    elif modality == "eda":
        return raw_modality
    elif modality == "rr":
        signals_rsp, info_rsp = nk.rsp_process(raw_modality, sampling_rate=fs)
        return signals_rsp["RSP_Clean"].values
    elif modality == "imu_right":
        return np.sqrt(np.sum(np.square(np.vstack((raw_modality[:, 0], raw_modality[:, 1], raw_modality[:, 2]))),
                              axis=0))
    elif modality == "intensity":
        return raw_modality
    elif modality == "gaze":
        return raw_modality
    elif modality == "pupils":
        threshold = 250
        cutoff = 0.5
        if participant in cfg['pupils_side_to_use']:
            pupil_extracted = raw_modality[:, cfg['pupils_side_to_use'][participant]]
            pupil_filtered = pupil_filtering(pupil_extracted.copy(), fs, threshold=threshold, cutoff=cutoff)
        else:
            left_pupil = raw_modality[:, 0]
            right_pupil = raw_modality[:, 1]
            n_cutoffs_left = np.sum(left_pupil > threshold)
            n_cutoffs_right = np.sum(right_pupil > threshold)
            if n_cutoffs_left > n_cutoffs_right:
                pupil_filtered = pupil_filtering(right_pupil.copy(), fs, threshold=threshold, cutoff=cutoff)
            else:
                pupil_filtered = pupil_filtering(left_pupil.copy(), fs, threshold=threshold, cutoff=cutoff)
        return pupil_filtered
    else:
        raise ValueError(f"Unsupported modality: {modality}")
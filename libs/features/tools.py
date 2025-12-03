import neurokit2 as nk
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..utils import butter_highpass_filter, pupil_filtering


def filter_raw_modality(raw_modality, modality, fs, cfg_dataset, participant):
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
    elif modality == "lbptop":
        return raw_modality
    elif modality == "pupils":
        threshold = 250
        cutoff = 0.5
        if participant in cfg_dataset['pupils_side_to_use']:
            pupil_extracted = raw_modality[:, cfg_dataset['pupils_side_to_use'][participant]]
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


def scale_minus1_plus1(df, cols=("Arousal", "Valence"), by="participant"):
    """
    Min-max scale *cols* to [-1, 1] within each group (participant).
    Returns a copy of *df* with scaled values.
    """
    out = df.copy()
    for pid, grp in df.groupby(by):
        for c in cols:
            lo, hi = grp[c].min(), grp[c].max()
            if hi == lo:                          # avoid divide-by-zero
                out.loc[grp.index, c] = 0.0
            else:
                out.loc[grp.index, c] = (grp[c] - lo) / (hi - lo) * 2 - 1
    return out


def scale_features(features_modality, use_dl_method, scaler_name='MinMax', feature_range=(-1, 1)):
    """Fit a Min-Max scaler on X[train_idx] and transform both splits."""
    if scaler_name == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler_name == 'standard':
        scaler = StandardScaler()
    elif scaler_name == 'no_scaling':
        return features_modality
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    if use_dl_method:
        features_modality_reshaped = features_modality.transpose(0, 2, 1).reshape(-1, features_modality.shape[1])
        features_modality_reshaped = scaler.fit_transform(features_modality_reshaped)
        features_modality = features_modality_reshaped.reshape(
            features_modality.shape[0], features_modality.shape[2], features_modality.shape[1]).transpose(0, 2, 1)
    else:
        features_modality = scaler.fit_transform(features_modality)
    return features_modality


def zero_out_bad_sensors(features_all, cfg, feature_cols_per_mod, in_place=False):
    sensors_to_exclude = cfg['dataset']['sensors_to_exclude']
    sensors_missing    = cfg['dataset']['sensors_missing']

    # copy if not in place
    feats_all = features_all if in_place else {
        p: X.copy() for p, X in features_all.items()
    }

    # precompute slices for each modality
    mod_slice = {
        m: slice(feature_cols_per_mod[m][0], feature_cols_per_mod[m][1])
        for m in feature_cols_per_mod
    }

    for participant, X in feats_all.items():
        # detect 2D vs 3D
        ndim = X.ndim
        if ndim not in (2, 3):
            raise ValueError(f"Unexpected array shape {X.shape}")

        # helper to zero out whole blocks
        def zero_block(start, stop):
            if ndim == 2:
                X[:, start:stop] = 0
            else:  # ndim == 3
                X[:, start:stop, :] = 0

        # --- exclude entire modality blocks ---
        for mod in sensors_to_exclude.get(participant, []):
            if mod not in mod_slice:
                continue
            start, stop = mod_slice[mod].start, mod_slice[mod].stop
            zero_block(start, stop)

        # --- replace missing‑sensor blocks with zeros via concat ---
        # (this preserves ordering if you wanted to actually *remove* columns;
        # for true zero‑in‑place you can just zero_block() here too)
        for mod in sensors_missing.get(participant, []):
            if mod not in mod_slice:
                continue
            start, stop = mod_slice[mod].start, mod_slice[mod].stop
            block_size = stop - start

            if ndim == 2:
                n_rows = X.shape[0]
                zero_blk = np.zeros((n_rows, block_size), dtype=X.dtype)
                X = np.concatenate([
                    X[:, :start],
                    zero_blk,
                    X[:, start:]
                ], axis=1)

            else:  # ndim == 3
                n_chunks, n_feats, chunk_len = X.shape
                zero_blk = np.zeros((n_chunks, block_size, chunk_len),
                                     dtype=X.dtype)
                X = np.concatenate([
                    X[:, :start, :],
                    zero_blk,
                    X[:, start:, :]
                ], axis=1)

        feats_all[participant] = X

    return feats_all


def get_X_groups(features_all, participants):
    X_list, groups = [], []
    for p in participants:
        X_p = features_all[p]
        X_list.append(X_p)
        groups.extend([p] * X_p.shape[0])
    X       = np.vstack(X_list)
    groups  = np.array(groups)

    return X, groups
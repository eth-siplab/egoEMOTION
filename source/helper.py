import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sp_feature_calculation import extract_features


def load_personality_data(participants, data_path, personality_domains):
    # Load personality questionnaire data and convert to DataFrame with numbers
    raw = pd.read_csv(f'{data_path}/personality_questionnaire_results.csv', sep=";", header=None, dtype=str)
    ids = raw.iloc[1, 1:].tolist()
    scales = raw.iloc[2:, 0].str.strip()
    mat = (raw.iloc[2:, 1:]
           .map(lambda x: str(x).replace(",", "."))  # comma → dot
           .apply(pd.to_numeric, errors="coerce")
           .set_axis(scales, axis=0)
           .set_axis(ids, axis=1))

    # Remove the first two rows and keep only participants wanted
    df_all_full = mat.T.copy()  # rows = participants, cols = all rows in the CSV
    df_all_full.columns.name = None  # remove the printed “0” column‑name label
    df_all_full = df_all_full.loc[participants]
    dup_mask = df_all_full.columns.duplicated(keep='first')

    # Split into means and T-scores and remove empty columns
    df_means = df_all_full.loc[:, ~dup_mask]  # all mean rows
    df_tscores = df_all_full.loc[:, dup_mask]  # all T‑score rows
    df_means = df_means.dropna(axis=1, how='all')
    df_tscores = df_tscores.dropna(axis=1, how='all')

    # Only keep desired personality domains
    df_means = df_means[personality_domains]
    df_tscores = df_tscores[personality_domains]

    return df_means, df_tscores


def load_affective_data(participants, data_path, session):
    dfs = []
    for p in participants:
        df = pd.read_csv(f"{data_path}/{p}/Session_{session}_{p}.csv")
        df['Valence'] = df['Valence'].astype(float) - 4  # Recenter valence
        df = df.drop(columns=['Session Start', 'Video Start', 'Q1 Start', 'Q2 Start', 'Q2 End', 'Session End'])
        df.insert(0, 'participant', p)  # Insert the participant column at position 0
        # Take mean of multiple same activity names
        if session == 'B':
            df = df.groupby(['participant', 'Activity Index', 'Activity Name', 'Activity ID'],
                            sort=False, as_index=False).mean()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


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


def get_df_labels(dfs_all, participants, session_label, n_chunks_all=None):
    df_parts = []
    for p in participants:
        if session_label in ('A', 'B'):
            dfp = dfs_all[session_label][dfs_all[session_label]['participant'] == p].copy()
        else:
            dfA = dfs_all['A'][dfs_all['A']['participant'] == p]
            dfB = dfs_all['B'][dfs_all['B']['participant'] == p]
            dfp = pd.concat([dfA, dfB], ignore_index=True)

        if n_chunks_all is not None:
            counts = n_chunks_all.get(p)
            if counts is None:
                raise KeyError(f"No n_chunks_all entry for participant {p}")
            if len(counts) != len(dfp):
                raise ValueError(
                    f"Participant {p}: expected {len(dfp)} tasks, "
                    f"but got counts array of length {len(counts)}"
                )
            dfp = dfp.loc[dfp.index.repeat(counts)].reset_index(drop=True)

        df_parts.append(dfp)
    df_all_labels = pd.concat(df_parts, ignore_index=True)
    return df_all_labels


def get_X_groups(features_all, participants):
    X_list, groups = [], []
    for p in participants:
        X_p = features_all[p]
        X_list.append(X_p)
        groups.extend([p] * X_p.shape[0])
    X       = np.vstack(X_list)
    groups  = np.array(groups)

    return X, groups


def zero_out_bad_sensors(features_all, cfg, feature_cols_per_mod, in_place=False):
    sensors_to_exclude = cfg['sensors_to_exclude']
    sensors_missing    = cfg['sensors_missing']

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

def get_features(participants, session_label, dfs_all, cfg, feature_modalities, do_new_feature_extraction,
                 chunk_len, scaler_name, df_all_labels, fisherfaces_labels, do_new_fisherface_extraction,
                 thr_emo_binary, df_personality_means, use_dl_method, per_session_scaling, n_jobs=1):
    if session_label in ['A', 'B']:
        print(f"Extracting features for session {session_label}...")
        features_all, feature_cols_per_mod, n_chunks_all = (
            extract_features(cfg, participants, feature_modalities, do_new_feature_extraction, session_label,
                             dfs_all, chunk_len, scaler_name, df_all_labels, fisherfaces_labels,
                             do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                             use_dl_method, n_jobs=n_jobs))
        features_all = {p: scale_features(features_all[p], use_dl_method, scaler_name=scaler_name) for p in participants}
    elif session_label == 'A_and_B':
        # Scale features separately for A and B and each participant
        if per_session_scaling:
            print("Extracting features for sessions A...")
            features_all_A, feature_cols_per_mod, n_chunks_all_A = (
                extract_features(cfg, participants, feature_modalities, do_new_feature_extraction, 'A', dfs_all,
                                 chunk_len, scaler_name, df_all_labels, fisherfaces_labels,
                                 do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                                 use_dl_method, n_jobs=n_jobs))
            print("Extracting features for sessions B...")
            features_all_B, feature_cols_per_mod, n_chunks_all_B = (
                extract_features(cfg, participants, feature_modalities, do_new_feature_extraction, 'B', dfs_all,
                                 chunk_len, scaler_name, df_all_labels, fisherfaces_labels,
                                 do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                                 use_dl_method, n_jobs=n_jobs))
            features_all_A = {p: scale_features(features_all_A[p], use_dl_method, scaler_name=scaler_name) for p in
                              participants}
            features_all_B = {p: scale_features(features_all_B[p], use_dl_method, scaler_name=scaler_name) for p in
                              participants}
            features_all = {p: np.concatenate((features_all_A[p], features_all_B[p]), axis=0) for p in participants}
            n_chunks_all = {p: np.concatenate((n_chunks_all_A[p], n_chunks_all_B[p]), axis=0) for p
                            in participants}

        # Scale features for A and B together
        else:
            print("Extracting features for sessions A and B...")
            features_all, feature_cols_per_mod, n_chunks_all = (
                extract_features(cfg, participants, feature_modalities, do_new_feature_extraction, 'A_and_B',
                                 dfs_all, chunk_len, scaler_name, df_all_labels, fisherfaces_labels,
                                 do_new_fisherface_extraction, thr_emo_binary, df_personality_means,
                                 use_dl_method, n_jobs=n_jobs))
            features_all = {p: scale_features(features_all[p], use_dl_method, scaler_name=scaler_name) for p in participants}
    else:
        raise ValueError(f"Unknown session label: {session_label}")

    return features_all, feature_cols_per_mod, n_chunks_all
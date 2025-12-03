import numpy as np
import os

from .glasses import extract_fisherfaces_features_overall, extract_imu_features, extract_lbptop_features
from .glasses import extract_pupils_features, extract_intensity_features, extract_gaze_features
from .health import extract_ecg_ppg_features, extract_eda_features, extract_rr_features
from .tools import filter_raw_modality, scale_features

from concurrent.futures import ProcessPoolExecutor, as_completed


# Get data chunks for a specific task
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


# Get data splits for a specific session
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


# Split data according to session label
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


# Load frames for Fisherfaces
def get_frames(cfg_features, participant):
    frames = np.load(f"{cfg_features['original_data_path']}/{participant}/et.npy", allow_pickle=True)
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, axis=3)
    return frames


# Extract features for a single participant
def _extract_single(participant,
                    session,
                    cfg_dataset,
                    cfg_features,
                    dfs_all,
                    df_all_labels,
                    fisherfaces_labels,
                    df_personality_means,
                    fisherfaces_frames_splits,
                    is_dl_method):
    """
    Extract features for exactly one participant.
    Returns (participant, features_matrix).
    """
    tasks_times = np.load(f"{cfg_features['original_data_path']}/task_times.npy", allow_pickle=True).item()

    if not os.path.exists(f"{cfg_features['preprocessed_data_path']}/{participant}"):
        os.makedirs(f"{cfg_features['preprocessed_data_path']}/{participant}")

    extracted_features_p = []
    feature_cols_per_mod = {}
    features_cols_n = 0

    chunk_len = cfg_features['chunk_len']
    fs_all = cfg_dataset['fs_all']
    feature_modalities = cfg_features['feature_modalities']

    if 'fisherfaces' in feature_modalities:
        col_fisher_n = 0
        if cfg_features['do_new_fisherface_extraction']:
            fisherface_features_all = extract_fisherfaces_features_overall(
                participant,
                session,
                cfg_dataset,
                cfg_features,
                df_all_labels,
                fisherfaces_labels,
                df_personality_means,
                fisherfaces_frames_splits)
            for fisherface_feature_label in fisherface_features_all:
                extracted_features_p.append(fisherface_feature_label)
                col_fisher_n += len(fisherface_feature_label)
        else:
            for fisherface_label in fisherfaces_labels:
                fisherfaces_features = np.load(
                    f"{cfg_features['preprocessed_data_path']}/{participant}/"
                    f"fisherfaces_features_session{session}_{chunk_len}_{fisherface_label}.npy",
                    allow_pickle=True)
                extracted_features_p.append(fisherfaces_features)
                col_fisher_n += fisherfaces_features.shape[1]
        feature_cols_per_mod['fisherfaces'] = [0, col_fisher_n]
        features_cols_n = col_fisher_n

        # Has to be changed: Just load some features to get n_chunks
        modality_raw = np.load(f"{cfg_features['original_data_path']}/{participant}/rr.npy", allow_pickle=True)
        _, n_chunks = split_data(modality_raw, fs_all['et'], fs_all['rr'],
                                               session, tasks_times, dfs_all, participant, chunk_len)

    # Load data and extract features for each modality
    for modality in feature_modalities:
        if participant in cfg_dataset['sensors_missing'] and modality in cfg_dataset['sensors_missing'][participant]:
            continue

        if cfg_features['do_new_feature_extraction']:
            if modality == 'fisherfaces':
                continue

            if is_dl_method:
                # lbptop always has 90 fps as same fps as video
                if modality == 'lbptop':
                    modality_raw = np.load(f"{cfg_features['original_data_path']}/{participant}/{modality}.npy",
                                           allow_pickle=True)
                else:
                    modality_raw = np.load(f"{cfg_features['original_data_path']}/{participant}/{modality}_90fps.npy",
                                           allow_pickle=True)
                modality_filtered = filter_raw_modality(modality_raw, modality, fs_all['et'], cfg_dataset, participant)
                modality_splits, n_chunks = split_data(modality_filtered, fs_all['et'], fs_all['et'],
                                                       session, tasks_times, dfs_all, participant, chunk_len)
                try:
                    modality_splits = np.asarray(modality_splits)
                except:
                    raise ValueError("Set chunk len to X seconds.")
                if len(modality_splits.shape) == 2:
                    modality_splits = np.expand_dims(modality_splits, 1)
                elif len(modality_splits.shape) == 3:
                    modality_splits = np.transpose(modality_splits, (0, 2, 1))
                features_modality = modality_splits
                save_path_temp = f"{cfg_features['preprocessed_data_path']}/{participant}/CL_{chunk_len}"
                if not os.path.exists(save_path_temp):
                    os.makedirs(save_path_temp)
                np.save(f"{save_path_temp}/{modality}_raw_session{session}.npy", features_modality,
                        allow_pickle=True)
                np.save(f"{save_path_temp}/{modality}_n_chunks{session}.npy", n_chunks,
                        allow_pickle=True)
            else:
                modality_raw = np.load(f"{cfg_features['original_data_path']}/{participant}/{modality}.npy",
                                       allow_pickle=True)
                modality_splits, n_chunks = split_data(modality_raw, fs_all['et'], fs_all[modality],
                                                       session, tasks_times, dfs_all, participant, chunk_len)
                if modality == 'ecg':
                    features_modality = extract_ecg_ppg_features(modality_splits, fs_all[modality], 'ecg')
                elif modality == 'ppg_nose':
                    features_modality = extract_ecg_ppg_features(modality_splits, fs_all[modality], 'ppg')
                elif modality == 'eda':
                    features_modality = extract_eda_features(modality_splits, fs_all[modality])
                elif modality == 'rr':
                    features_modality = extract_rr_features(modality_splits, fs_all[modality])
                elif modality == 'imu_right':
                    features_modality = extract_imu_features(modality_splits)
                elif modality == 'gaze':
                    features_modality = extract_gaze_features(modality_splits)
                elif modality == 'pupils':
                    features_modality = extract_pupils_features(modality_splits, fs_all[modality], cfg_dataset,
                                                                participant)
                elif modality == 'intensity':
                    features_modality = extract_intensity_features(modality_splits)
                elif modality == 'lbptop':
                    features_modality = extract_lbptop_features(modality_splits)
                else:
                    raise ValueError(f"Unknown modality: {modality}")

                np.save(f"{cfg_features['preprocessed_data_path']}/{participant}/"
                        f"{modality}_features_session{session}_{chunk_len}.npy", features_modality,
                        allow_pickle=True)
                np.save(f"{cfg_features['preprocessed_data_path']}/{participant}/"
                        f"{modality}_n_chunks{session}_{chunk_len}.npy", features_modality,
                        allow_pickle=True)
            extracted_features_p.append(features_modality)
        else:
            if modality == 'fisherfaces':
                continue

            if is_dl_method:
                load_path = f"{cfg_features['preprocessed_data_path']}/{participant}/CL_{chunk_len}"
                features_modality = np.load(f"{load_path}/{modality}_raw_session{session}.npy", allow_pickle=True)
                n_chunks = np.load(f"{load_path}/{modality}_n_chunks{session}.npy", allow_pickle=True)
            else:
                features_modality = np.load(f"{cfg_features['preprocessed_data_path']}/{participant}/"
                                            f"{modality}_features_session{session}_{chunk_len}.npy",
                                            allow_pickle=True)
                n_chunks = np.load(f"{cfg_features['preprocessed_data_path']}/{participant}/"
                                            f"{modality}_n_chunks{session}_{chunk_len}.npy",
                                            allow_pickle=True)
            extracted_features_p.append(features_modality)
        feature_cols_per_mod[modality] = [features_cols_n, features_cols_n + np.asarray(features_modality).shape[1]]
        features_cols_n = features_cols_n + np.asarray(features_modality).shape[1]

    all_feats = np.hstack(extracted_features_p)
    return participant, all_feats, feature_cols_per_mod, n_chunks


def _load_frames_single(participant, cfg_dataset, cfg_features, session_label, tasks_times, dfs_all, subsample):
    frames = get_frames(cfg_features, participant)
    frames_splits = split_data(frames, cfg_dataset['fs_all']['et'], cfg_dataset['fs_all']['et'], session_label,
                               tasks_times, dfs_all, participant, cfg_features['chunk_len'], subsample=subsample)
    return frames_splits[0], participant


# Main function to extract features in parallel
def extract_features(cfg_dataset,
                     cfg_features,
                     session,
                     dfs_all,
                     df_all_labels,
                     fisherfaces_labels,
                     df_personality_means,
                     is_dl_method
                     ):
    """
    Parallel version of extract_features.
    n_jobs = None  -> use as many workers as CPUs
    n_jobs = 1     -> fully serial
    n_jobs = k>1   -> k workers
    """

    n_jobs = cfg_features['n_jobs']
    extracted_features_all, feature_cols_per_mod_out, n_chunks_all = {}, {}, {}

    if 'fisherfaces' in cfg_features['feature_modalities'] and cfg_features['do_new_fisherface_extraction']:
        subsample = 10  # as training otherwise takes too long
        fisherfaces_frames_splits = {}
        tasks_times = np.load(f"{cfg_features['original_data_path']}/task_times.npy", allow_pickle=True).item()
        with ProcessPoolExecutor(max_workers=len(cfg_dataset['participants'])) as exe:
            futures = {exe.submit(_load_frames_single,
                                  p,
                                  cfg_dataset,
                                  cfg_features,
                                  session,
                                  tasks_times,
                                  dfs_all,
                                  subsample):
                           p for p in cfg_dataset['participants']}
            for fut in as_completed(futures):
                frames_splits, participant = fut.result()
                fisherfaces_frames_splits[participant] = frames_splits
    else:
        fisherfaces_frames_splits = {}

    if cfg_features['do_new_fisherface_extraction']:
        n_jobs=1

    if n_jobs == 1:
        for participant in cfg_dataset['participants']:
            participant, feats, feature_cols_per_mod, n_chunks = (
                _extract_single(participant,
                                session,
                                cfg_dataset,
                                cfg_features,
                                dfs_all,
                                df_all_labels,
                                fisherfaces_labels,
                                df_personality_means,
                                fisherfaces_frames_splits,
                                is_dl_method
                                ))
            extracted_features_all[participant] = feats
            n_chunks_all[participant] = n_chunks
            if participant not in cfg_dataset['sensors_missing']:
                features_cols_per_mod_out = feature_cols_per_mod
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as exe:
            futures = {exe.submit(_extract_single,
                                  p,
                                  session,
                                  cfg_dataset,
                                  cfg_features,
                                  dfs_all,
                                  df_all_labels,
                                  fisherfaces_labels,
                                  df_personality_means,
                                  fisherfaces_frames_splits,
                                  is_dl_method
                                  ):
                           p for p in cfg_dataset['participants']}
            for fut in as_completed(futures):
                participant, feats, feature_cols_per_mod, n_chunks = fut.result()
                extracted_features_all[participant] = feats
                n_chunks_all[participant] = n_chunks
                if participant not in cfg_dataset['sensors_missing']:
                    features_cols_per_mod_out = feature_cols_per_mod

    return extracted_features_all, features_cols_per_mod_out, n_chunks_all


def _extract_and_scale_session(session, cfg_dataset, cfg_features, dfs_all, df_all_labels, fisherfaces_labels,
                               df_personality_means, is_dl_method):
    """Helper: Handles extraction and scaling for a single specific session label."""
    print(f"Extracting features for session {session}...")

    # 1. Extract
    features, feat_cols, n_chunks = extract_features(
        cfg_dataset,
        cfg_features,
        session,  # Explicitly pass the session label here
        dfs_all,
        df_all_labels,
        fisherfaces_labels,
        df_personality_means,
        is_dl_method
    )

    # 2. Scale
    scaled_features = {
        p: scale_features(features[p], is_dl_method, scaler_name=cfg_features['scaler_name'])
        for p in cfg_dataset['participants']
    }

    return scaled_features, feat_cols, n_chunks


# Get features according to session configuration and scaling options
# def get_features(cfg, dfs_all, df_all_labels, fisherfaces_labels, df_personality_means):
def get_features(session, cfg_dataset, cfg_features, dfs_all, df_all_labels, fisherfaces_labels, df_personality_means,
                 is_dl_method):
    do_split_scaling = (session == 'A_and_B' and
                        cfg_features['per_session_scaling'])

    # Case 1: Split Scaling (A and B processed separately, then merged)
    if do_split_scaling:
        # Process A
        feats_A, cols, chunks_A = _extract_and_scale_session(
            'A', cfg_dataset, cfg_features, dfs_all, df_all_labels, fisherfaces_labels, df_personality_means,
            is_dl_method
        )
        # Process B
        feats_B, _, chunks_B = _extract_and_scale_session(
            'B', cfg_dataset, cfg_features, dfs_all, df_all_labels, fisherfaces_labels, df_personality_means,
            is_dl_method
        )

        # Merge Results
        participants = cfg_dataset['participants']
        features_all = {
            p: np.concatenate((feats_A[p], feats_B[p]), axis=0)
            for p in participants
        }
        n_chunks_all = {
            p: np.concatenate((chunks_A[p], chunks_B[p]), axis=0)
            for p in participants
        }
        feature_cols_per_mod = cols  # Columns remain the same

    # Case 2: Standard Processing (Single session OR A_and_B with global scaling)
    elif session in ['A', 'B', 'A_and_B']:
        features_all, feature_cols_per_mod, n_chunks_all = _extract_and_scale_session(
            session, cfg_dataset, cfg_features, dfs_all, df_all_labels, fisherfaces_labels, df_personality_means,
            is_dl_method
        )

    else:
        raise ValueError(f"Unknown session label: {session}")

    return features_all, feature_cols_per_mod, n_chunks_all
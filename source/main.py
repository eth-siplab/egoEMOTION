import heapq
import numpy as np
import torch
import yaml
import random

from AffectivePredictor import AffectivePredictor
from helper import load_personality_data, load_affective_data, get_df_labels, get_X_groups, zero_out_bad_sensors
from helper import get_features

from itertools import count


def set_random_seeds(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def main():
    participants = ['005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017',
                    '018', '019', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032',
                    '033', '034', '035', '036', '037', '038', '039', '040', '042', '043', '044', '045', '046']

    # Load configs
    with open('../configs/config_egoemotion.yml') as yf:
        cfg = yaml.load(yf, Loader=yaml.FullLoader)

    data_path = cfg['original_data_path']
    personality_domains = cfg['personality_domains']
    emotion_domains = cfg['emotion_domains']

    # Load emotions and personality domains
    df_personality_means, df_personality_T = load_personality_data(participants, data_path, personality_domains)
    df_A_all = load_affective_data(participants, data_path, 'A')
    df_B_all = load_affective_data(participants, data_path, 'B')
    dfs_all = {'A': df_A_all, 'B': df_B_all}

    # Define the three sessions
    sessions = {
        'A': ["Arousal", "Valence", "Dominance"],
        'B': ["Arousal", "Valence", "Dominance"],
        'A_and_B': ["Arousal", "Valence", "Dominance"]
    }

    # Parameters on how to run the analysis
    n_jobs = len(participants)
    do_new_feature_extraction = False
    do_new_fisherface_extraction = False  # specified separately since it takes long
    chunk_len = 'all'  # 'all', 'last_30', X (int in seconds): 20 was used for DL methods in the paper
    per_session_scaling = True
    splitter_name = 'LOGO'  # LOGO or GKF (used for DL methods in the paper)
    feature_modalities_list = [['ecg'], ['eda'], ['rr'], ['ecg', 'eda', 'rr'],
                               ['pupils'], ['ppg_nose'], ['imu_right'], ['intensity'], ['fisherfaces'], ['gaze'],
                               ['lbptop'],
                               ['pupils', 'ppg_nose', 'imu_right', 'intensity', 'fisherfaces', 'gaze', 'lbptop'],
                               ['ecg', 'eda', 'rr', 'pupils', 'imu_right', 'intensity', 'fisherfaces', 'gaze', 'lbptop']]
    feature_modalities_list = [['ecg', 'eda', 'rr']]

    # Hyperparameters: The following combinations were used in the paper:
    # Affect: standard, no_feature_selection, 10, RbfSVM
    # emotion_single_label: standard, mi, 10, RF/standard, no_feature_selection, X, WER
    # personality: no_scaling, mi, 10, RF; IMPORTANT: do_single_prediction=True for traditional and false for ML
    # DL methods: standard, no_feature_selection, X, WER/DCNN, GKF
    scaler_name = 'standard'  # standard, minmax (-1, 1), no_scaling
    feature_selection_methods = ['no_feature_selection']  # 'fisher', 'mi', 'pca', 'anova', 'lrc', 'no_feature_selection'
    nums_features = [10]  # number of features to select when feature selection is applied

    # Model and prediction targets
    # Classical ML classifiers: 'LogisticRegression', 'GaussianNB', 'LinearSVM', 'kNN', 'RF', 'RbfSVM', 'xGB'
    # Deep learning methods: 'WER', 'DCNN'
    clf_names = ['RbfSVM']
    use_dl_method = False  # True if using DL method, False if using traditional ML methods
    prediction_targets = ['affect']  # 'affect', 'emotion', 'personality'

    if use_dl_method:
        with open(f'../configs/ml/egoemotion_egoemotion_{clf_names[0]}.yml') as yf:
            cfg_dl = yaml.load(yf, Loader=yaml.FullLoader)
    else:
        cfg_dl = None

    # Other parameters
    thr_emo_binary = 0.1
    random_seed = 0
    set_random_seeds(random_seed)

    # Print stats
    print(f"Used parameters: \n")
    print(f"Chunk length: {chunk_len}")
    print(f"Per session scaling: {per_session_scaling}")
    print(f"Scaler: {scaler_name}")
    print(f"Feature selection method: {feature_selection_methods}")
    print(f"Number of features: {nums_features}")
    print(f"Classifier: {clf_names}")
    print(f"Random seed: {random_seed}")

    _uid = count()
    k = 10  # keep top k results
    topk = {t: [] for t in prediction_targets}  # each is a min‑heap
    results_f1_all = {session_label: [] for session_label in sessions.keys()}
    results_random = {session_label: [] for session_label in sessions.keys()}
    for i_fm, feature_modalities in enumerate(feature_modalities_list):
        print(f"Feature modalities: {feature_modalities}")
        for feature_selection_method in feature_selection_methods:
            print(f"\n===== Feature selection method: {feature_selection_method} =====")
            for num_features in nums_features:
                print(f"\n===== Number of features: {num_features} =====")
                for clf_name in clf_names:
                    print(f"\n===== Classifier: {clf_name} =====")
                    for session_label, affect_domains in sessions.items():

                        print(f"\n===== SESSION: {session_label} =====")
                        fisherfaces_labels = affect_domains + emotion_domains + personality_domains + ['strongest_emotion']
                        # fisherfaces_labels = ['Open-Mindedness']

                        # Get labels (only needed for Fisherfaces when not using raw signals)
                        df_all_labels = get_df_labels(dfs_all, participants, session_label)

                        # Get features
                        features_all, feature_cols_per_mod, n_chunks_all = get_features(
                            participants, session_label, dfs_all, cfg, feature_modalities, do_new_feature_extraction,
                            chunk_len, scaler_name, df_all_labels, fisherfaces_labels, do_new_fisherface_extraction,
                            thr_emo_binary, df_personality_means, use_dl_method, per_session_scaling, n_jobs=n_jobs)
                        features_all = zero_out_bad_sensors(features_all, cfg, feature_cols_per_mod, in_place=True)
                        X, groups = get_X_groups(features_all, participants)

                        if use_dl_method:
                            df_all_labels = get_df_labels(dfs_all, participants, session_label,
                                                          n_chunks_all=n_chunks_all)

                        # Initialize the AffectivePredictor
                        Affective_predictor = AffectivePredictor(X, groups, df_all_labels, participants, emotion_domains,
                                                                 personality_domains, feature_selection_method,
                                                                 num_features, clf_name, affect_domains, random_seed,
                                                                 cfg, session_label, dfs_all, chunk_len,
                                                                 feature_modalities, fisherfaces_labels,
                                                                 use_dl_method, splitter_name, cfg_dl=cfg_dl)

                        # Predict different affective states
                        for prediction_target in prediction_targets:
                            if prediction_target == 'affect':
                                result_temp, results_f1_temp, rand_f1_temp = Affective_predictor.predict_affect()
                            elif prediction_target == 'emotion':
                                result_temp, results_f1_temp, rand_f1_temp = Affective_predictor.predict_emotion(
                                    return_macro=True)
                            elif prediction_target == 'personality':
                                result_temp, results_f1_temp, rand_f1_temp = Affective_predictor.predict_personality(
                                    df_personality_means, df_personality_T, features_all,
                                    use_T=False, do_single_prediction=True)
                            else:
                                raise ValueError(f"Unknown prediction target: {prediction_target}")


                            # Add results_f1_temp to the results_df
                            results_f1_all[session_label].append(results_f1_temp)
                            results_random[session_label] = rand_f1_temp

                            score = result_temp
                            current_config = dict(feature_selection_method=feature_selection_method,
                                           num_features=num_features,
                                           clf_name=clf_name,
                                           scaler_name=scaler_name,
                                           session_label=session_label)
            
                            heap = topk[prediction_target]
                            entry = (score, next(_uid), current_config)
            
                            if len(heap) < k:
                                heapq.heappush(heap, entry)  # just fill the heap
                            elif score > heap[0][0]:  # better than current worst?
                                heapq.heapreplace(heap, entry)  # drop worst, keep best 5

        print(f"\nTop‑{k} results")
        for target, heap in topk.items():
            best = sorted(heap, key=lambda x: x[0], reverse=True)  # high → low
            print(f"\n▶ {target}")
            for rank, (s, _, cfg_temp) in enumerate(best, 1):
                print(f"  {rank}. {s:6.3f}  {cfg_temp}")


if __name__ == '__main__':
    main()

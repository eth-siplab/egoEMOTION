import argparse
import numpy as np
import torch
import random

from libs import AffectivePredictor, load_config
from libs.datasets import load_personality_data, load_affective_data, get_df_labels
from libs.features import zero_out_bad_sensors, get_X_groups, get_features

from pprint import pprint


def set_random_seeds(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def main(args):
    # Load and print configs
    cfg = load_config(
        model_file=args.cfg_model,
        dataset_file=args.cfg_dataset
    )
    pprint(cfg)

    # Load emotions and personality domains
    df_personality_means, df_personality_T = load_personality_data(cfg['dataset']['participants'],
                                                                   cfg['features']['original_data_path'],
                                                                   cfg['dataset']['personality_domains'])
    df_A_all = load_affective_data(cfg['dataset']['participants'], cfg['features']['original_data_path'], 'A')
    df_B_all = load_affective_data(cfg['dataset']['participants'], cfg['features']['original_data_path'], 'B')
    dfs_all = {'A': df_A_all, 'B': df_B_all}

    # Set random seeds
    set_random_seeds(cfg['random_seed'])

    # Get labels (only needed for Fisherfaces when not using raw signals)
    fisherfaces_labels = (cfg['dataset']['affect_domains'] +
                          cfg['dataset']['emotion_domains'] +
                          cfg['dataset']['personality_domains'] +
                          ['strongest_emotion'])
    df_all_labels = get_df_labels(dfs_all, cfg['dataset']['participants'], cfg['session'])

    # Get features
    features_all, feature_cols_per_mod, n_chunks_all = (
        get_features(cfg['session'],
                     cfg['dataset'],
                     cfg['features'],
                     dfs_all,
                     df_all_labels,
                     fisherfaces_labels,
                     df_personality_means,
                     cfg['model']['is_dl_method'])
    )
    features_all = zero_out_bad_sensors(features_all, cfg, feature_cols_per_mod, in_place=True)
    X, groups = get_X_groups(features_all, cfg['dataset']['participants'])

    if cfg['model']['is_dl_method']:
        df_all_labels = get_df_labels(dfs_all, cfg['dataset']['participants'], cfg['session'],
                                      n_chunks_all=n_chunks_all)

    # Initialize the AffectivePredictor
    predictor = AffectivePredictor(
        X=X,
        cfg=cfg,
        groups=groups,
        df_all_labels=df_all_labels,
        fisherface_labels=fisherfaces_labels
    )

    # Predict different affective states
    if cfg['prediction_target'] == 'affect':
        predictor.predict_affect()
    elif cfg['prediction_target'] == 'emotion':
        predictor.predict_emotion()
    elif cfg['prediction_target'] == 'personality':
        predictor.predict_personality(df_personality_means, df_personality_T, features_all, use_T=False)
    else:
        raise ValueError(f"Unknown prediction target: {cfg['prediction_target']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model for affective state prediction on the egoEMOTION dataset')
    parser.add_argument('cfg_dataset', metavar='FILE',
                        help='path to metadata file of egoemotion dataset')
    parser.add_argument('cfg_model', metavar='FILE',
                        help='path to config file of model to use')
    args = parser.parse_args()
    main(args)

import yaml

DEFAULTS = {
    # random seed for reproducibilit
    "random_seed": 0,
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    return config


def load_config(model_file, dataset_file, defaults=DEFAULTS):
    # 1. Start with the hardcoded defaults
    config = defaults.copy()  # Make a copy to avoid modifying the global DEFAULTS

    # 2. Load and merge the DATASET/METADATA file
    with open(dataset_file, "r") as fd:
        dataset_config = yaml.load(fd, Loader=yaml.FullLoader)
        # Ensure it's nested under 'dataset' key if your yaml is flat,
        # or just merge if your yaml already has 'dataset': { ... }
        # Assuming your yaml is just the raw data, we wrap it:
        if "dataset" not in dataset_config:
            dataset_config = {"dataset": dataset_config}

        _merge(dataset_config, config)

    # 3. Load and merge the MODEL/EXPERIMENT file
    with open(model_file, "r") as fd:
        model_config = yaml.load(fd, Loader=yaml.FullLoader)

        if model_config['model']['clf_name'] in ['WER', 'DCNN']:
            model_config['model']['is_dl_method'] = True
        else:
            model_config['model']['is_dl_method'] = False

        _merge(model_config, config)

    # 4. Run the internal transfer logic (e.g. moving fps_out to loader)
    config = _update_config(config)

    return config
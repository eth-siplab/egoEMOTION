import numpy as np
import yaml

from scipy.interpolate import CubicSpline


def quantile_artifact_removal(data, q1, q3, factor):
    sig_change = np.diff(data, axis=0)
    quantile_q1 = np.quantile(sig_change, q1)
    quantile_q3 = np.quantile(sig_change, q3)
    mean_change = (quantile_q1 + quantile_q3) / 2
    std_change = factor * 1.34896 * (quantile_q3 - quantile_q1)
    big_diffs = np.argwhere((sig_change > (mean_change + std_change)) |
                            (sig_change < (mean_change - std_change)))
    for i in range(len(big_diffs)):
        data[big_diffs[i][0] + 1:] = (data[big_diffs[i][0] + 1:] +
                                      (data[big_diffs[i][0]] - data[big_diffs[i][0] + 1]))

    return data


def quantile_artifact_removal2(data, q1, q3, factor):
    sig_change = np.diff(data, axis=0)
    quantile_q1 = np.quantile(sig_change, q1)
    quantile_q3 = np.quantile(sig_change, q3)
    mean_change = (quantile_q1 + quantile_q3) / 2
    std_change = factor * 1.34896 * (quantile_q3 - quantile_q1)
    big_diffs = np.argwhere((sig_change > (mean_change + std_change)) |
                            (sig_change < (mean_change - std_change)))
    for i in range(len(big_diffs)):
        data[big_diffs[i][0] + 1] = data[big_diffs[i][0]]

    return data


def get_adjusted_task_times(task_times, chunk_length):
    task_times = {task: [int(task_times[task][0] - task_times['video'][0]),
                         int(task_times[task][1] - task_times['video'][0])]
                  for task in task_times.keys()}
    task_times['running'][1] = (task_times['running'][1] // chunk_length) * chunk_length
    return task_times


def get_task_chunk_list(config, participant):
    cfg_path = f'./configs/preprocessing/config_preprocessing_{config.TRAIN.DATA.DATASET}.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)
    task_times_adjusted = get_adjusted_task_times(configs_pre['task_times'][participant],
                                                  config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)

    task_chunk_list = {'task_names': [], 'keep': []}
    for task in task_times_adjusted.keys():
        n_chunks = (((task_times_adjusted[task][1] - task_times_adjusted[task][0]) //
                     config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE + 1) // config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)
        if task in ['walking_1', 'walking_2', 'walking_3']:
            task_name = 'walking'
        else:
            task_name = task
        task_chunk_list['task_names'].extend([task_name] * n_chunks)

        if (task_name in config.TASKS_TO_USE) and (participant not in configs_pre['exclusion_list_l'][task_name]):
            task_chunk_list['keep'].extend([1] * n_chunks)
        else:
            task_chunk_list['keep'].extend([0] * n_chunks)
    return task_chunk_list


def get_ml_config_name(configs_g):
    ml_config_name = (f'CL{configs_g["clip_length"]}_W{configs_g["w"]}_H{configs_g["h"]}_'
                      f'Label{configs_g["label_type"]}_VideoType')
    for video_type in configs_g["video_types"]:
        ml_config_name += video_type
    return ml_config_name


def resample(data, new_len, interp_type):
    if interp_type == 'linear':  # same as sample
        interp_data = np.interp(np.linspace(0, data.shape[0] - 1, new_len), np.arange(data.shape[0]), data)
    elif interp_type == 'cubic':
        spl = CubicSpline(np.arange(data.shape[0]), data)
        interp_data = spl(np.linspace(0, data.shape[0] - 1, new_len))
    else:
        raise ValueError(f'Interpolation type {interp_type} not supported.')
    return interp_data


def resample_signal(signal, new_len, interp_type):
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal)
    if new_len == signal.shape[0]:
        return signal
    else:
        if len(signal.shape) == 1:
            interp_signal = resample(signal, new_len, interp_type)
        else:
            interp_signal = np.zeros((new_len, signal.shape[1]))
            for i in range(signal.shape[1]):
                interp_signal[:, i] = resample(signal[:, i], new_len, interp_type)
        return interp_signal


def upsample_video(frames, upsample_factor, interp_type):
    frames_upsampled = np.zeros((frames.shape[0] * upsample_factor - 2, frames.shape[1], frames.shape[2],
                                 frames.shape[3]))
    for row in range(frames.shape[1]):
        frames_upsampled[:, row, :, 0] = resample_signal(frames[:, row, :, 0], frames.shape[0] * upsample_factor - 2,
                                                         interp_type)
    return frames_upsampled


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def normalize(data, type_normalization):
    # No normalization
    if type_normalization is None:
        data = data
    # Around zero, zero mean and std = 1
    elif type_normalization == 'std':
        data = (data - data.mean()) / data.std()
        data[np.isnan(data)] = 0
    # Between 0 and 1
    elif type_normalization == 'zero_one':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data

def calculate_accuracy(y_pred, y):
    # y_pred and y are numpy arrays
    correct = (y_pred == y).sum()
    acc = correct / y.shape[0]
    return acc
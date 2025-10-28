"""
Neural data processing and formatting script for electrophysiology data
Processes spike and behavioral data for LFADS model training
"""
import os
import h5py
import sys
import numpy as np
import pickle
import torch
from datetime import datetime
from context_general_bci.config import DataKey, DatasetConfig

def days_between_date(str1, str2):
    date1 = datetime.strptime(str1, f'%Y%m%d')
    date2 = datetime.strptime(str2, f'%Y%m%d')

    delta = abs((date1 - date2).days)
    return delta


def slide_window(data, window_size=30, window_step=6, limit=0):
    """
    Apply sliding window segmentation to time series data
    
    Args:
        data: Input data list, each element represents one trial's data
        window_size: Window size in time points, default 30
        window_step: Window sliding step in time points, default 6
    
    Returns:
        window_data: List of segmented window data
    """
    window_data = []
    for trial in data:
        if limit != 0:
            trial = trial[:limit]
            window_size = limit
        window_trial = []
        length = len(trial)
        # Skip trial if length is smaller than window size
        if length < window_size:
            continue
        # Calculate number of windows for this trial
        for i in range(int((length - window_size) / window_step + 1)):
            # Extract and stack window data
            window_trial.append(trial[i * window_step: i * window_step + window_size, ...])
        window_data.append(torch.stack(window_trial))
    return window_data


def get_data(filename, limit=0, window_size=30, window_step=6):
    """
    Load and process electrophysiology data from lab data format
    
    Args:
        filename: Path to the data file
    
    Returns:
        spike: Processed and windowed spike data
        bhv: Processed and windowed behavioral data
    """
    # Load lab data file
    trial_file = glob(os.path.join(filename, '*.pth'))
    all_spikes = []
    all_bhv = []
    for trials in tqdm(trial_file):
        data = torch.load(trials)
        spikes = data[DataKey.spikes]['Leibniz-main'].squeeze()
        bhv = data[DataKey.bhvr_vel]
        all_spikes.append(spikes)
        all_bhv.append(bhv)
    
    spikes = slide_window(all_spikes, limit=limit, window_size=window_size, window_step=window_step)
    bhv = slide_window(all_bhv, limit=limit, window_size=window_size, window_step=window_step)
    
    return spikes, bhv

def propress_data(file_name, output_file='propressed_data', limit=0, window_size=30, window_step=6, date_start='20150730'):
    
    # Determine behavioral data type based on filename
    label = file_name.split('/')[-1]  # Extract session label from filename
    
    # Load and process data
    spike, bhv = get_data(file_name, limit=0, window_size=window_size, window_step=window_step)
    
    # Verify data consistency
    assert len(spike) == len(bhv)
    
    # Split data into training and validation sets
    trial_num = len(spike)
    trial_id = np.random.permutation(trial_num)  # Randomly shuffle trial indices
    val_ratio = 0.2  # Validation set ratio
    
    train_id = trial_id[:int(trial_num * (1 - val_ratio))]
    val_id = trial_id[int(trial_num * (1 - val_ratio)):]
    
    # Concatenate training and validation data
    train_data = np.concatenate([spike[i] for i in train_id])
    val_data = np.concatenate([spike[i] for i in val_id])
    train_bhv = np.concatenate([bhv[i] for i in train_id])
    val_bhv = np.concatenate([bhv[i] for i in val_id])
    
    # Create data dictionary for LFADS training
    data_dict = {
        'train_behavior': train_bhv,
        'train_data': train_data,
        'train_inds': train_id,
        'valid_behavior': val_bhv,
        'valid_data': val_data,
        'valid_inds': val_id
    }

    date = label.split('_')[0]
    delta_day = days_between_date(date_start, date)

    # Save data as HDF5 file for LFADS
    h5_filepath = os.path.join(output_file, 'chopped_data', 'day' + str(delta_day))
    os.makedirs(h5_filepath, exist_ok=True)
    with h5py.File(os.path.join(h5_filepath, 'lfads_' + label + '.h5'), 'w') as h5file:
        for key, value in data_dict.items():
            h5file.create_dataset(key, data=value)

    # Load and process data
    # spike, bhv = get_data(file_name, limit=limit, window_size=window_size, window_step=window_step)
    
    # Create separate data structure for pickle output
    data = {}
    data['spikes'] = [np.concatenate(np.array(spike[i]), axis=0) for i in range(len(spike))]
    data['bhv'] = [np.concatenate(np.array(bhv[i]), axis=0) for i in range(len(bhv))]
    # Save data as pickle file
    h5_filepath = os.path.join(output_file, 'trialized_data', 'day' + str(delta_day))
    os.makedirs(h5_filepath, exist_ok=True)
    with open(os.path.join(h5_filepath, label + '.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    return True


if __name__ == "__main__":
    from glob import glob
    # data_path = 'data/leibniz/20250711_rtt_001'
    # file_names = glob(os.path.join(data_path, '*.pth'))
    file_names = ['data/leibniz/20250711_rtt_001']
    from tqdm import tqdm
    for file_name in tqdm(file_names):
        propress_data(file_name, output_file='Leibniz_propressed_data', limit=0, window_size=30, window_step=6, date_start='20250711')

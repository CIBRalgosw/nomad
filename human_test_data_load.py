"""
Neural data processing and format conversion script
Converts neural spike signals and behavioral data to LFADS model training format
"""

import os
import h5py
import numpy as np
import pickle
import pynapple as nap
import os


def slide_window(data, window_size=30, window_step=6):
    """
    Apply sliding window segmentation to data
    
    Args:
        data: Input data list, each element represents one trial's data
        window_size: Window size in time points, default 30
        window_step: Window sliding step in time points, default 6
    
    Returns:
        window_data: List of segmented window data
    """
    window_data = []
    for trial in data:
        window_trial = []
        length = len(trial)
        # Skip trial if length is smaller than window size
        if length < window_size:
            continue
        # Calculate number of windows for this trial
        for i in range(int((length - window_size) / window_step + 1)):
            # Extract and stack window data
            window_trial.append(np.vstack(trial[i * window_step: i * window_step + window_size, ...]))
        window_data.append(np.stack(window_trial))
    return window_data


def get_data(filepath, filename):
    """
    Load and process neural data and behavioral data from NWB file
    
    Args:
        filepath: Path to the data file
        filename: Name of the data file
    
    Returns:
        spike: Spike signal data after sliding window processing
        bhv: Behavioral data after sliding window processing
    """
    # Load NWB data file
    data = nap.load_file(os.path.join(filepath, filename))
    
    # Extract data components
    binned_spikes = data['binned_spikes']  # Binned spike signals
    cursor_vel = data['cursor_vel']        # Cursor velocity (behavioral data)
    trials = data['trials']                # Trial information
    
    # Create trial time intervals
    behavior_trials = nap.IntervalSet(start=trials['start'], end=trials['end'], time_units='s')
    
    # Split spike signals and behavioral data by trial
    spike = [binned_spikes.restrict(behavior_trials[i]).as_array() for i in range(len(behavior_trials))]
    bhv = [cursor_vel.restrict(behavior_trials[i]).as_array() for i in range(len(behavior_trials))]
    
    # Apply sliding window processing
    spike = slide_window(spike)
    bhv = slide_window(bhv)
    
    return spike, bhv


if __name__ == "__main__":
    # Data path and file configuration
    data_path = '/home/yuezhifeng_lab/zhangzizuo/DATA/CIBR/019_pBCI_XZD/nomad/data/human_bci_T11'
    file_name = 'session2.nwb'
    label = file_name.split('.')[0]  # Extract session label from filename
    
    # Load and process data
    spike, bhv = get_data(data_path, file_name)
    
    # Verify data consistency
    assert len(spike) == len(bhv)
    
    # Scale behavioral data
    bhv = [i * 10 for i in bhv]
    
    # Split data into training and validation sets
    trial_num = len(spike)
    trial_id = np.random.permutation(trial_num)  # Randomly shuffle trial indices
    val_ratio = 0.2  # Validation set ratio
    
    train_id = trial_id[:int(trial_num * (1 - val_ratio))]
    val_id = trial_id[int(trial_num * (1 - val_ratio)):]
    
    # Concatenate training data
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
    
    # Create separate data structure for pickle output
    data = {}
    data['spikes'] = [np.concatenate(spike[i], axis=0) for i in val_id]
    data['bhv'] = [np.concatenate(bhv[i], axis=0) for i in val_id]
    
    # Save data as pickle file
    with open(label + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    # Save data as HDF5 file for LFADS
    with h5py.File('lfads_' + label + '.h5', 'w') as h5file:
        for key, value in data_dict.items():
            h5file.create_dataset(key, data=value)
"""
Neural data processing and formatting script for electrophysiology data
Processes spike and behavioral data for LFADS model training
"""
import os
import h5py
import sys
import numpy as np
import pickle
sys.path.append('/home/yuezhifeng_lab/zhangzizuo/DATA/programs/xds/xds_python')
from xds import lab_data
from datetime import datetime

def days_between_date(str1, str2):
    date1 = datetime.strptime(str1, f'%Y%m%d')
    date2 = datetime.strptime(str2, f'%Y%m%d')

    delta = abs((date1 - date2).days)
    return delta

def padding_units(elec_name, spike, max_unit=100, prefix='elec'):
    """
    Pad and reorder electrode units to standardized format
    
    Args:
        elec_name: List of original electrode unit names
        spike: List of spike count arrays for each trial
        max_unit: Maximum number of units to pad to, default 100
        prefix: Prefix for electrode naming convention, default 'elec'
    
    Returns:
        ordered_spike: Padded and reordered spike data with consistent unit ordering
    """
    elec_name = np.array(elec_name)
    ori_channel_num = len(elec_name)
    padding_id = ori_channel_num
    ordered_id = []
    
    # Create ordered unit indices with padding for missing units
    for i in range(max_unit):
        unit_name = prefix + str(i + 1)
        id = np.where(elec_name == unit_name)[0]
        if len(id) == 1:
            ordered_id.append(id[0])
        else:
            ordered_id.append(padding_id)
            padding_id += 1
    
    # Apply padding and reordering to spike data
    ordered_spike = []
    for trial in spike:
        time_len, _ = trial.shape
        # Pad with zeros for missing units
        padding_spike = np.concatenate((trial, np.zeros((time_len, max_unit - ori_channel_num))), axis=-1)
        ordered_spike.append(padding_spike[..., np.array(ordered_id)])
    return ordered_spike

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
            window_trial.append(np.vstack(trial[i * window_step: i * window_step + window_size, ...]))
        window_data.append(np.stack(window_trial))
    return window_data


def get_data(filepath, filename, bin_size=0.02, smooth_size=0, bhv_type='vel', limit=0, window_size=30, window_step=6):
    """
    Load and process electrophysiology data from lab data format
    
    Args:
        filepath: Path to the data file
        filename: Name of the data file
        bin_size: Bin size for spike counting in seconds, default 0.02
        smooth_size: Smoothing kernel size, 0 for no smoothing
        bhv_type: Type of behavioral data ('vel' for velocity, 'force' for force)
    
    Returns:
        spike: Processed and windowed spike data
        bhv: Processed and windowed behavioral data
    """
    # Load lab data file
    data = lab_data(filepath, filename)
    data.update_bin_data(bin_size)
    
    # Apply smoothing if specified
    if smooth_size:
        data.smooth_binned_spikes(bin_size, 'gaussian', smooth_size)
    
    # Extract spike data and electrode names
    spike = data.get_trials_data_spike_counts('R', 'gocue_time', 0, 'end_time', 0)
    # spike = data.get_trials_data_spike_counts('R', 'gocue_time', 0, 'gocue_time', 0.6)
    elec_name = data.unit_names
    
    # Process spike data with unit padding and windowing
    spike = padding_units(elec_name, spike)
    spike = slide_window(spike, limit=limit, window_size=window_size, window_step=window_step)
    
    # Extract behavioral data based on type
    if bhv_type == 'force':
        # import pdb;pdb.set_trace()
        force = data.get_trials_data_force('R', 'gocue_time', 0, 'end_time', 0)
        # force = data.get_trials_data_force('R', 'gocue_time', 0, 'gocue_time', 0.6)
        force = [i - i[0] for i in force]
        force = slide_window(force, limit=limit, window_size=window_size, window_step=window_step)
        bhv = force
    if bhv_type == 'vel':
        _, cursor_velocity, _ = data.get_trials_data_cursor('R', 'gocue_time', 0, 'end_time', 0)
        cursor_velocity = [i - i[0] for i in cursor_velocity]
        cursor_velocity = slide_window(cursor_velocity, limit=limit, window_size=window_size, window_step=window_step)
        bhv = cursor_velocity
    
    return spike, bhv

def propress_data(data_path, file_name, output_file='propressed_data', limit=0, window_size=30, window_step=6, date_start='20150730'):
    # Determine behavioral data type based on filename
    bhv_type = 'vel' if 'Che' in file_name else 'force'
    label = file_name.split('.')[0]  # Extract session label from filename
    
    # Load and process data
    spike, bhv = get_data(data_path, file_name, bhv_type=bhv_type, limit=0, window_size=window_size, window_step=window_step)
    
    # Verify data consistency
    assert len(spike) == len(bhv)
    
    # Calculate trial lengths for analysis
    trial_len = [len(spike[i]) for i in range(len(spike))]
    
    # Scale force data if applicable
    if bhv_type == 'force':
        bhv = [i / 1000 for i in bhv]
    
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
        # 'train_behavior': train_bhv,
        'train_data': train_data,
        'train_inds': train_id,
        # 'valid_behavior': val_bhv,
        'valid_data': val_data,
        'valid_inds': val_id
    }

    date = file_name.split('_')[1]
    delta_day = days_between_date(date_start, date)

    # Save data as HDF5 file for LFADS
    h5_filepath = os.path.join(output_file, 'chopped_data', 'day' + str(delta_day))
    os.makedirs(h5_filepath, exist_ok=True)
    with h5py.File(os.path.join(h5_filepath, 'lfads_' + label + '.h5'), 'w') as h5file:
        for key, value in data_dict.items():
            h5file.create_dataset(key, data=value)

    # Load and process data
    spike, bhv = get_data(data_path, file_name, bhv_type=bhv_type, limit=limit, window_size=window_size, window_step=window_step)
    
    # Create separate data structure for pickle output
    data = {}
    data['spikes'] = [np.concatenate(spike[i], axis=0) for i in range(len(spike))]
    data['bhv'] = [np.concatenate(bhv[i], axis=0) for i in range(len(bhv))]
    # import pdb;pdb.set_trace()
    # Save data as pickle file
    h5_filepath = os.path.join(output_file, 'trialized_data', 'day' + str(delta_day))
    os.makedirs(h5_filepath, exist_ok=True)
    with open(os.path.join(h5_filepath, label + '.pkl'), 'wb') as f:
        pickle.dump(data, f)
    
    return True


if __name__ == "__main__":
    # Data path and file configuration
    # data_path = '/home/yuezhifeng_lab/zhangzizuo/DATA/CIBR/019_pBCI_XZD/nomad/data/Jango_ISO_2015'
    data_path = 'data/Chewie_CO_2016'

    file_names = os.listdir(data_path)
    file_names = ['Chewie_20161104_001.mat']
    from tqdm import tqdm
    for file_name in tqdm(file_names):
        propress_data(data_path, file_name, output_file='C_temp_propressed_data', limit=23, window_size=20, window_step=6, date_start='20160927')

import os, pickle, h5py
from os import path, chmod
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from lfads_tf2.subclasses.behavior.models import BehaviorLFADS
from lfads_tf2.tuples import LoadableData, LFADSInput
from lfads_tf2.utils import load_posterior_averages, restrict_gpu_usage, unflatten
restrict_gpu_usage(0)

from nomad.models import AlignLFADS
from nomad.tuples import AlignInput
from nomad.defaults import get_cfg_defaults

import tensorflow as tf
import argparse

# 固定随机种子
np.random.seed(42)
tf.config.experimental_run_functions_eagerly(
    True
)

from utils import get_causal_model_output, generate_lagged_matrix, fit_and_eval_decoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', '-d', type=int, default=0)
    parser.add_argument('--monkey', type=str, default='C')  # J_day0_model
    args = parser.parse_args()
    monkey = args.monkey
    fig_path = f'{monkey}_figs_nomad'
    os.makedirs(fig_path, exist_ok=True)
    # train the LFADS Day 0 model
    if os.path.exists(f'{monkey}_day0_model/lfads_ckpts/'):
        model = BehaviorLFADS(model_dir=f'{monkey}_day0_model')
    else:
        model = BehaviorLFADS(cfg_path=f'demo/configs/lfads_monkey_{monkey}.yaml')
        # model = BehaviorLFADS(cfg_path='demo/configs/lfads.yaml')
        model.train()
    # load trialized data 
    data0 = glob(os.path.join(f'{monkey}_propressed_data/trialized_data/day0', '*.pkl'))[0]
    # if monkey == 'J':
    #     date0 = 'propressed_data/trialized_data/day0/Jango_20150730_001.pkl'
    # elif monkey == 'C':
    #     data0 = 'C_propressed_data/trialized_data/day0/Chewie_20160927_001.pkl'
    with open(data0, 'rb') as f:
        day0_trials= pickle.load(f)
    stack_spikes = day0_trials['spikes']
    behavior = day0_trials['bhv']

    behavior = [i * 1000 for i in behavior]

    day0_gen_states = stack_spikes
    from tqdm import tqdm
    day0_gen_states = []
    for tr in tqdm(range(len(stack_spikes))):
        day0_gen_states.append(get_causal_model_output(model, 
                                0.02, 
                                stack_spikes[tr], 
                                ['gen_states'], 
                                {
                                    'gen_states': model.cfg.MODEL.GEN_DIM
                                })['gen_states'])

    # train decoder 
    # keep_trials = [np.isnan(x).sum(axis=(0,1)) == 0 for x in behavior]
    # behavior = [behavior[keep_trials[i]] for i in range(len(keep_trials))]
    # day0_gen_states = [day0_gen_states[keep_trials[i]] for i in range(len(keep_trials))]
    ntrials = len(behavior)
    ntime, ndim = behavior[0].shape
    n_train_trials = int(0.8 * ntrials)
    n_test_trials = ntrials - n_train_trials
    train_trials = np.random.choice(ntrials, n_train_trials, replace=False)
    test_trials = np.setdiff1d(np.arange(ntrials), train_trials)
    decode_train_behavior = np.vstack([behavior[i] for i in train_trials])
    decode_test_behavior = np.vstack([behavior[i] for i in test_trials])
    decode_train_neural = np.vstack([day0_gen_states[i] for i in train_trials])
    decode_test_neural = np.vstack([day0_gen_states[i] for i in test_trials])
    # import pdb;pdb.set_trace()
    lagged_train_gen_states = generate_lagged_matrix(decode_train_neural, 3)
    lagged_test_gen_states = generate_lagged_matrix(decode_test_neural, 3)
    score, decoder, pred_force = fit_and_eval_decoder(
        lagged_train_gen_states,
        decode_train_behavior[3:, :],
        lagged_test_gen_states,
        decode_test_behavior[3:, :],
        return_preds=True
        )
    print('Day 0 score:', score)
    # plot decoder outputs 
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.set_aspect('equal')
    true_behavior = np.split(decode_test_behavior, n_test_trials, axis=0)
    true_plots = [plt.plot(tr[:, 0], tr[:, 1], label='True', color='k') for tr in true_behavior]
    predicted_behavior = np.split(np.pad(pred_force, ((3,0), (0,0)), constant_values=np.nan), n_test_trials, axis=0)
    pred_plots = [plt.plot(tr[:, 0], tr[:, 1], label='Pred', color='r', alpha=0.7) for tr in predicted_behavior]
    # make axes invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.legend([true_plots[0][0], pred_plots[0][0]], ['True', 'Pred'])
    plt.text(-700, 550, 'Force $R^2$ = {:.2f}'.format(score))
    plt.savefig(os.path.join(fig_path, 'day0.png'))

    # for date in [1, 2, 6, 7, 8, 9, 10, 21, 25, 26, 27, 28, 29, 32, 37, 38, 40, 91, 95]:
    for date in [args.day]:
        trialized_path = f'{monkey}_propressed_data/trialized_data/day' + str(date)
        try:
            file_path = glob(os.path.join(trialized_path, '*.pkl'))[0]
        except:
            print(date, 'file is not exist')
            continue
        # if we pass Day K data through the model 
        with open(file_path, 'rb') as f:
            dayk_trials= pickle.load(f)
        stack_spikes = dayk_trials['spikes']
        behavior = dayk_trials['bhv']
        behavior = [i * 1000 for i in behavior]

        day0_gen_states = stack_spikes
        from tqdm import tqdm
        day0_gen_states = []
        for tr in tqdm(range(len(stack_spikes))):
            day0_gen_states.append(get_causal_model_output(model, 
                                    0.02, 
                                    stack_spikes[tr], 
                                    ['gen_states'], 
                                    {
                                        'gen_states': model.cfg.MODEL.GEN_DIM
                                    })['gen_states'])

        # train decoder 
        ntrials = len(behavior)
        ntime, ndim = behavior[0].shape
        n_train_trials = int(0.8 * ntrials)
        n_test_trials = ntrials - n_train_trials
        train_trials = np.random.choice(ntrials, n_train_trials, replace=False)
        test_trials = np.setdiff1d(np.arange(ntrials), train_trials)
        decode_test_behavior = np.vstack([behavior[i] for i in test_trials])
        decode_test_neural = np.vstack([day0_gen_states[i] for i in test_trials])
        lagged_test_gen_states = generate_lagged_matrix(decode_test_neural, 3)

        score = decoder.score(lagged_test_gen_states, decode_test_behavior[3:, :])
        dayk_pred_force = decoder.predict(lagged_test_gen_states)
        print(f'Day {date} score:', score)

        # plot decoder outputs 
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_aspect('equal')
        true_behavior = np.split(decode_test_behavior, n_test_trials, axis=0)
        true_plots = [plt.plot(tr[:, 0], tr[:, 1], label='True', color='k') for tr in true_behavior]
        predicted_behavior = np.split(np.pad(dayk_pred_force, ((3,0), (0,0)), constant_values=np.nan), n_test_trials, axis=0)
        pred_plots = [plt.plot(tr[:, 0], tr[:, 1], label='Pred', color='r', alpha=0.7) for tr in predicted_behavior]
        # make axes invisible
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.legend([true_plots[0][0], pred_plots[0][0]], ['True', 'Pred'])
        plt.text(-700, 550, 'Force $R^2$ = {:.2f}'.format(score))
        plt.savefig(os.path.join(fig_path, f'day{date}.png'))

        # initialize nomad model
        align_dir = f'{monkey}_align_model/{monkey}_align_model' + str(date)
        if not os.path.exists(f'{align_dir}/align_ckpts/'):
            align_cfg = get_cfg_defaults()
            cfg_update_path = f'demo/configs/nomad_monkey_{monkey}.yaml'
            align_cfg.merge_from_file(cfg_update_path)
            align_cfg['TRAIN']['ALIGN_DIR'] = align_dir
            align_model = AlignLFADS(cfg_node=align_cfg)
        else:
            align_model = AlignLFADS(align_dir=f'{align_dir}/')

        # load the h5 files into the model
        h5_file0 = glob(os.path.join(f'{monkey}_propressed_data/chopped_data/day0', '*.h5'))[0]
        day0_datadict = h5py.File(h5_file0, 'r')
        h5_filek = glob(os.path.join(f'{monkey}_propressed_data/chopped_data/day{date}', '*.h5'))[0]
        dayk_datadict = h5py.File(h5_filek, 'r')
        align_input = AlignInput(
            day0_train_data=day0_datadict['train_data'],
            day0_valid_data=day0_datadict['valid_data'],
            dayk_train_data=dayk_datadict['train_data'],
            dayk_valid_data=dayk_datadict['valid_data'],
            day0_train_inds=day0_datadict['train_inds'],
            day0_valid_inds=day0_datadict['valid_inds'],
            dayk_train_inds=dayk_datadict['train_inds'],
            dayk_valid_inds=dayk_datadict['valid_inds'])
        align_model.load_datasets(align_input)

        if not os.path.exists(f'{align_dir}/align_ckpts/'):
            done = False
            while not done:
                results = align_model.train_epoch()
                done = results.get('done', False)

        # # get NoMAD outputs 
        # with open('demo/trialized_data/Jango_20150731_001.pkl', 'rb') as f:
        #     dayk_trials= pickle.load(f)
        # stack_spikes = dayk_trials['spikes']
        # behavior = dayk_trials['bhv']
        # behavior = [i * 1000 for i in behavior]

        from tqdm import tqdm
        day0_gen_states = []
        for tr in tqdm(range(len(stack_spikes))):
            day0_gen_states.append(get_causal_model_output(align_model.lfads_dayk, 
                                    0.02, 
                                    stack_spikes[tr], 
                                    ['gen_states'], 
                                    {
                                        'gen_states': model.cfg.MODEL.GEN_DIM
                                    })['gen_states'])

        # train decoder 
        # keep_trials = [np.isnan(x).sum(axis=(0,1)) == 0 for x in behavior]
        # behavior = [behavior[keep_trials[i]] for i in range(len(keep_trials))]
        # day0_gen_states = [day0_gen_states[keep_trials[i]] for i in range(len(keep_trials))]

        decode_test_behavior = np.vstack([behavior[i] for i in test_trials])
        decode_test_neural = np.vstack([day0_gen_states[i] for i in test_trials])
        lagged_test_gen_states = generate_lagged_matrix(decode_test_neural, 3)

        score = decoder.score(lagged_test_gen_states, decode_test_behavior[3:, :])
        dayk_pred_force = decoder.predict(lagged_test_gen_states)
        print(f'Day {date} NoMAD score:', score)
        # plot decoder outputs 
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.set_aspect('equal')
        true_behavior = np.split(decode_test_behavior, n_test_trials, axis=0)
        true_plots = [plt.plot(tr[:, 0], tr[:, 1], label='True', color='k') for tr in true_behavior]
        predicted_behavior = np.split(np.pad(dayk_pred_force, ((3,0), (0,0)), constant_values=np.nan), n_test_trials, axis=0)
        pred_plots = [plt.plot(tr[:, 0], tr[:, 1], label='Pred', color='r', alpha=0.7) for tr in predicted_behavior]
        # make axes invisible
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.legend([true_plots[0][0], pred_plots[0][0]], ['True', 'Pred'])
        plt.text(-700, 550, 'Force $R^2$ = {:.2f}'.format(score))
        plt.savefig(os.path.join(fig_path, f'day{date} nomad.png'))


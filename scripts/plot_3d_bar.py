import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import os.path as osp

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

path = osp.join(os.environ['HOME'],
                'workspaces/szeth_ws/src/szeth/data')

max_trials = 20

# CMAX
cmax_results = pickle.load(
    open(osp.join(path, 'cmax_3d_results.pkl'), 'rb'))
cmax_results_dict = cmax_results[1]

# CMAXPP
cmaxpp_results = pickle.load(
    open(osp.join(path, 'cmaxpp_3d_results.pkl'), 'rb'))
cmaxpp_results_dict = cmaxpp_results[1]

# ADAPTIVE
adaptive_cmaxpp_results = pickle.load(
    open(osp.join(path, 'adaptive_cmaxpp_3d_results.pkl'), 'rb'))
adaptive_cmaxpp_results_dict = adaptive_cmaxpp_results[4]

# Bar plot
# trials = [0, 5, 10, 15, 19]
trials = [0, 2, 4, 6, 8, 10, 12, 14, 16]
width = 0.5

cmax_trials_extracted = []
cmax_trials = []
cmax_counts = []

cmaxpp_trials_extracted = []
cmaxpp_trials = []
cmaxpp_counts = []

adaptive_cmaxpp_trials_extracted = []
adaptive_cmaxpp_trials = []
adaptive_cmaxpp_counts = []

for trial in trials:
    cmax_steps = 0
    count = 0
    for seed in cmax_results_dict.keys():
        if len(cmax_results_dict[seed]) > trial:
            cmax_steps += cmax_results_dict[seed][trial]
            count += 1
    if count != 0:
        cmax_steps = cmax_steps / count
        cmax_trials.append(trial)
        cmax_trials_extracted.append(cmax_steps)
        cmax_counts.append(count)

    cmaxpp_steps = 0
    count = 0
    for seed in cmaxpp_results_dict.keys():
        if len(cmaxpp_results_dict[seed]) > trial:
            cmaxpp_steps += cmaxpp_results_dict[seed][trial]
            count += 1
    if count != 0:
        cmaxpp_steps = cmaxpp_steps / count
        cmaxpp_trials.append(trial)
        cmaxpp_trials_extracted.append(cmaxpp_steps)
        cmaxpp_counts.append(count)

    adaptive_cmaxpp_steps = 0
    count = 0
    for seed in adaptive_cmaxpp_results_dict.keys():
        if len(adaptive_cmaxpp_results_dict[seed]) > trial:
            adaptive_cmaxpp_steps += adaptive_cmaxpp_results_dict[seed][trial]
            count += 1
    if count != 0:
        adaptive_cmaxpp_steps = adaptive_cmaxpp_steps / count
        adaptive_cmaxpp_trials.append(trial)
        adaptive_cmaxpp_trials_extracted.append(adaptive_cmaxpp_steps)
        adaptive_cmaxpp_counts.append(count)

# Convert into np arrays
cmax_trials, cmaxpp_trials, adaptive_cmaxpp_trials = np.array(
    cmax_trials), np.array(cmaxpp_trials), np.array(adaptive_cmaxpp_trials)

fig, ax = plt.subplots()
rects_cmax = ax.bar(cmax_trials - width,
                    cmax_trials_extracted, width, label='CMAX')
rects_cmaxpp = ax.bar(cmaxpp_trials,
                      cmaxpp_trials_extracted, width, label='CMAX++')
rects_adaptive_cmaxpp = ax.bar(adaptive_cmaxpp_trials + width,
                               adaptive_cmaxpp_trials_extracted, width,
                               label='Adaptive CMAX++')

ax.set_xlabel('Repetitions')
ax.set_ylabel('Average number of steps to goal')
ax.set_xticks(trials)
# ax.set_xticklabels([0, 5, 10, 15, 20])
ax.legend()
plt.show()

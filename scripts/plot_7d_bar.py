import ipdb
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
plt.rcParams.update({'font.size': 25})

path = osp.join(os.environ['HOME'],
                'workspaces/szeth_ws/src/szeth/data')

max_trials = 20
max_tsteps = 500

# CMAX
cmax_results = pickle.load(
    open(osp.join(path, 'cmax_7d_approximate_results.pkl'), 'rb'))
cmax_results_dict = cmax_results[4]

# CMAXPP
cmaxpp_results = pickle.load(
    open(osp.join(path, 'cmaxpp_7d_approximate_results.pkl'), 'rb'))
cmaxpp_results_dict = cmaxpp_results[4]

# ADAPTIVE
adaptive_cmaxpp_results = pickle.load(
    open(osp.join(path, 'adaptive_cmaxpp_7d_approximate_results_alpha_4.pkl'), 'rb'))
adaptive_cmaxpp_results_dict = adaptive_cmaxpp_results[4]

# Model
model_results_dict = pickle.load(
    open(osp.join(path, 'model_7d_approximate_results.pkl'), 'rb'))

# Knn
knn_results_dict = pickle.load(
    open(osp.join(path, 'knn_7d_approximate_results.pkl'), 'rb'))

# Qlearning
qlearning_results_dict = pickle.load(
    open(osp.join(path, 'qlearning_7d_approximate_results.pkl'), 'rb'))

# Bar plot
trials = [0, 5, 10, 15, 19]
# trials = [0, 4, 8, 12, 16, 19]
seeds = [0, 1, 5, 6, 9]
width = 0.5

cmax_trials_extracted_mean = []
cmax_trials_extracted_std = []
cmax_trials = []
cmax_counts = []

cmaxpp_trials_extracted_mean = []
cmaxpp_trials_extracted_std = []
cmaxpp_trials = []
cmaxpp_counts = []

adaptive_cmaxpp_trials_extracted_mean = []
adaptive_cmaxpp_trials_extracted_std = []
adaptive_cmaxpp_trials = []
adaptive_cmaxpp_counts = []

model_trials_extracted_mean = []
model_trials_extracted_std = []
model_trials = []
model_counts = []

knn_trials_extracted_mean = []
knn_trials_extracted_std = []
knn_trials = []
knn_counts = []

qlearning_trials_extracted_mean = []
qlearning_trials_extracted_std = []
qlearning_trials = []
qlearning_counts = []

for trial in trials:
    cmax_steps = []
    count = 0
    count_success = 0
    for seed in cmax_results_dict.keys():
        if seed not in seeds:
            continue
        if len(cmax_results_dict[seed]) > trial:
            cmax_steps.append(cmax_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     cmax_steps += max_tsteps
        #     count += 1
    if count != 0:
        # cmax_steps = cmax_steps / count
        cmax_mean_steps = np.mean(cmax_steps)
        cmax_std_steps = np.std(cmax_steps) / np.sqrt(count_success)
        cmax_trials.append(trial)
        cmax_trials_extracted_mean.append(cmax_mean_steps)
        cmax_trials_extracted_std.append(cmax_std_steps)
        cmax_counts.append(count_success)

    cmaxpp_steps = []
    count = 0
    count_success = 0
    for seed in cmaxpp_results_dict.keys():
        if seed not in seeds:
            continue
        if len(cmaxpp_results_dict[seed]) > trial:
            cmaxpp_steps.append(cmaxpp_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     cmaxpp_steps += max_tsteps
        #     count += 1
    if count != 0:
        # cmaxpp_steps = cmaxpp_steps / count
        cmaxpp_mean_steps = np.mean(cmaxpp_steps)
        cmaxpp_std_steps = np.std(cmaxpp_steps) / np.sqrt(count_success)
        cmaxpp_trials.append(trial)
        cmaxpp_trials_extracted_mean.append(cmaxpp_mean_steps)
        cmaxpp_trials_extracted_std.append(cmaxpp_std_steps)
        cmaxpp_counts.append(count_success)

    adaptive_cmaxpp_steps = []
    count = 0
    count_success = 0
    for seed in adaptive_cmaxpp_results_dict.keys():
        if seed not in seeds:
            continue
        if len(adaptive_cmaxpp_results_dict[seed]) > trial:
            adaptive_cmaxpp_steps.append(
                adaptive_cmaxpp_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     adaptive_cmaxpp_steps += max_tsteps
        #     count += 1
    if count != 0:
        # adaptive_cmaxpp_steps = adaptive_cmaxpp_steps / count
        adaptive_cmaxpp_mean_steps = np.mean(adaptive_cmaxpp_steps)
        adaptive_cmaxpp_std_steps = np.std(
            adaptive_cmaxpp_steps) / np.sqrt(count_success)
        adaptive_cmaxpp_trials.append(trial)
        adaptive_cmaxpp_trials_extracted_mean.append(
            adaptive_cmaxpp_mean_steps)
        adaptive_cmaxpp_trials_extracted_std.append(
            adaptive_cmaxpp_std_steps)
        adaptive_cmaxpp_counts.append(count_success)

    model_steps = []
    count = 0
    count_success = 0
    for seed in model_results_dict.keys():
        if seed not in seeds:
            continue
        if len(model_results_dict[seed]) > trial:
            model_steps.append(model_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     model_steps += max_tsteps
        #     count += 1
    if count != 0:
        # model_steps = model_steps / count
        model_mean_steps = np.mean(model_steps)
        model_std_steps = np.std(model_steps) / np.sqrt(count_success)
        model_trials.append(trial)
        model_trials_extracted_mean.append(model_mean_steps)
        model_trials_extracted_std.append(model_std_steps)
        model_counts.append(count_success)

    knn_steps = []
    count = 0
    count_success = 0
    for seed in knn_results_dict.keys():
        if seed not in seeds:
            continue
        if len(knn_results_dict[seed]) > trial:
            knn_steps.append(knn_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     knn_steps += max_tsteps
        #     count += 1
    if count != 0:
        # knn_steps = knn_steps / count
        knn_mean_steps = np.mean(knn_steps)
        knn_std_steps = np.std(knn_steps) / np.sqrt(count_success)
        knn_trials.append(trial)
        knn_trials_extracted_mean.append(knn_mean_steps)
        knn_trials_extracted_std.append(knn_std_steps)
        knn_counts.append(count_success)

    qlearning_steps = []
    count = 0
    count_success = 0
    for seed in qlearning_results_dict.keys():
        if seed not in seeds:
            continue
        if len(qlearning_results_dict[seed]) > trial:
            qlearning_steps.append(qlearning_results_dict[seed][trial])
            count += 1
            count_success += 1
        # else:
        #     qlearning_steps += max_tsteps
        #     count += 1
    if count != 0:
        # qlearning_steps = qlearning_steps / count
        qlearning_mean_steps = np.mean(qlearning_steps)
        qlearning_std_steps = np.std(qlearning_steps) / np.sqrt(count_success)
        qlearning_trials.append(trial)
        qlearning_trials_extracted_mean.append(qlearning_mean_steps)
        qlearning_trials_extracted_std.append(qlearning_std_steps)
        qlearning_counts.append(count_success)

# Convert into np arrays
cmax_trials, cmaxpp_trials, adaptive_cmaxpp_trials = np.array(
    cmax_trials), np.array(cmaxpp_trials), np.array(adaptive_cmaxpp_trials)
model_trials, knn_trials = np.array(model_trials), np.array(knn_trials)
qlearning_trials = np.array(qlearning_trials)

# ipdb.set_trace()

fig, ax = plt.subplots()
rects_cmax = ax.bar(cmax_trials - 2 * width,
                    cmax_trials_extracted_mean, width, label='\\textsc{Cmax}')
rects_cmaxpp = ax.bar(cmaxpp_trials - width,
                      cmaxpp_trials_extracted_mean, width, label='\\textsc{Cmax++}')
rects_adaptive_cmaxpp = ax.bar(adaptive_cmaxpp_trials,
                               adaptive_cmaxpp_trials_extracted_mean, width,
                               label='\\textsc{A-Cmax++}')
rects_model = ax.bar(model_trials + width, model_trials_extracted_mean,
                     width, label='NN Residual Model')
rects_knn = ax.bar(knn_trials + 2*width,
                   knn_trials_extracted_mean, width, label='KNN Residual Model')
rects_qlearning = ax.bar(qlearning_trials + 3 * width,
                         qlearning_trials_extracted_mean, width, label='Q-learning')


def autolabel(rects, labels):
    idx = 0
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(labels[idx]),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=20)
        idx += 1


autolabel(rects_cmax, cmax_counts)
autolabel(rects_cmaxpp, cmaxpp_counts)
autolabel(rects_adaptive_cmaxpp, adaptive_cmaxpp_counts)
autolabel(rects_model, model_counts)
autolabel(rects_knn, knn_counts)
autolabel(rects_qlearning, qlearning_counts)

ax.set_xlabel('Repetitions')
ax.set_ylabel('Average number of steps taken to reach goal')
ax.set_xticks(trials)
ax.set_xticklabels([1, 5, 10, 15, 20])
# ax.set_xticklabels([1, 4, 8, 12, 16, 20])
ax.set_yscale('log')
ax.legend(ncol=3, loc='upper left')
ax.set_title('7D Pick-and-Place Experiment')
ax.set_ylim(top=1e3)

# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/7d_bar.pdf'))

plt.show()

import pickle
import os
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

path = osp.join(os.environ['HOME'],
                'workspaces/szeth_ws/src/szeth/data')

max_trials = 20

plt.subplot(1, 2, 1)

# Read cmax
cmax_results_dict = pickle.load(
    open(osp.join(path, 'cmax_3d_results.pkl'), 'rb'))
cmax_results_dict = cmax_results_dict[1]
cmax_results_successful = []
cmax_results_unsuccessful = []
for seed in cmax_results_dict.keys():
    trial_returns = cmax_results_dict[seed]
    trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        cmax_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        cmax_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='blue',
        #          alpha=0.2)

cmax_mean_results_successful = np.mean(cmax_results_successful, axis=0)
cmax_std_results_successful = np.std(
    cmax_results_successful, axis=0) / np.sqrt(len(cmax_results_successful))

# Plot cmax
plt.plot(range(max_trials), cmax_mean_results_successful, color='blue', linestyle='-',
         label='\\textsc{Cmax} '+str(len(cmax_results_successful)) + '/' + str(len(cmax_results_dict.keys())))
plt.fill_between(range(max_trials),
                 cmax_mean_results_successful - cmax_std_results_successful,
                 cmax_mean_results_successful + cmax_std_results_successful,
                 color='blue', alpha=0.5)
plt.scatter(cmax_results_unsuccessful, [
            0 for _ in range(len(cmax_results_unsuccessful))], marker='x',
            color='blue')

# Read cmaxpp
cmaxpp_results_dict = pickle.load(
    open(osp.join(path, 'cmaxpp_3d_results.pkl'), 'rb'))
cmaxpp_results_dict = cmaxpp_results_dict[1]
cmaxpp_results_successful = []
cmaxpp_results_unsuccessful = []
for seed in cmaxpp_results_dict.keys():
    trial_returns = cmaxpp_results_dict[seed]
    trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        cmaxpp_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        cmaxpp_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='red',
        #          alpha=0.2)

cmaxpp_mean_results_successful = np.mean(cmaxpp_results_successful, axis=0)
cmaxpp_std_results_successful = np.std(
    cmaxpp_results_successful, axis=0) / np.sqrt(len(cmaxpp_results_successful))

# Plot cmaxpp
plt.plot(range(max_trials), cmaxpp_mean_results_successful, color='red', linestyle='-',
         label='\\textsc{Cmax++} '+str(len(cmaxpp_results_successful)) + '/' + str(len(cmaxpp_results_dict.keys())))
plt.fill_between(range(max_trials),
                 cmaxpp_mean_results_successful - cmaxpp_std_results_successful,
                 cmaxpp_mean_results_successful + cmaxpp_std_results_successful,
                 color='red', alpha=0.5)
plt.scatter(cmaxpp_results_unsuccessful, [
            0 for _ in range(len(cmaxpp_results_unsuccessful))], marker='x',
            color='red')

# Read adaptive_cmaxpp
adaptive_cmaxpp_results_dict = pickle.load(
    open(osp.join(path, 'adaptive_cmaxpp_3d_results.pkl'), 'rb'))
adaptive_cmaxpp_results_dict = adaptive_cmaxpp_results_dict[4]
adaptive_cmaxpp_results_successful = []
adaptive_cmaxpp_results_unsuccessful = []
for seed in adaptive_cmaxpp_results_dict.keys():
    trial_returns = adaptive_cmaxpp_results_dict[seed]
    trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        adaptive_cmaxpp_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        adaptive_cmaxpp_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='green',
        #          alpha=0.2)

adaptive_cmaxpp_mean_results_successful = np.mean(
    adaptive_cmaxpp_results_successful, axis=0)
adaptive_cmaxpp_std_results_successful = np.std(
    adaptive_cmaxpp_results_successful, axis=0) / np.sqrt(len(adaptive_cmaxpp_results_successful))

# Plot adaptive_cmaxpp
plt.plot(range(max_trials), adaptive_cmaxpp_mean_results_successful, color='green', linestyle='-',
         label='\\textsc{Adaptive Cmax++} '+str(len(adaptive_cmaxpp_results_successful)) + '/' + str(len(adaptive_cmaxpp_results_dict.keys())))
plt.fill_between(range(max_trials),
                 adaptive_cmaxpp_mean_results_successful -
                 adaptive_cmaxpp_std_results_successful,
                 adaptive_cmaxpp_mean_results_successful +
                 adaptive_cmaxpp_std_results_successful,
                 color='green', alpha=0.5)
plt.scatter(adaptive_cmaxpp_results_unsuccessful, [
            0 for _ in range(len(adaptive_cmaxpp_results_unsuccessful))], marker='x',
            color='green')

# plt.yscale('log')
plt.legend()

plt.subplot(1, 2, 2)

# Read cmax
cmax_results_dict = pickle.load(
    open(osp.join(path, 'cmax_3d_results.pkl'), 'rb'))
cmax_results_dict = cmax_results_dict[1]
cmax_results_successful = []
cmax_results_unsuccessful = []
for seed in cmax_results_dict.keys():
    trial_returns = cmax_results_dict[seed]
    # trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        cmax_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        cmax_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='blue',
        #          alpha=0.2)

cmax_mean_results_successful = np.mean(cmax_results_successful, axis=0)
cmax_std_results_successful = np.std(
    cmax_results_successful, axis=0) / np.sqrt(len(cmax_results_successful))

# Plot cmax
plt.plot(range(max_trials), cmax_mean_results_successful, color='blue', linestyle='-',
         label='\\textsc{Cmax} '+str(len(cmax_results_successful)) + '/' + str(len(cmax_results_dict.keys())))
plt.fill_between(range(max_trials),
                 cmax_mean_results_successful - cmax_std_results_successful,
                 cmax_mean_results_successful + cmax_std_results_successful,
                 color='blue', alpha=0.5)
plt.scatter(cmax_results_unsuccessful, [
            0 for _ in range(len(cmax_results_unsuccessful))], marker='x',
            color='blue')

# Read cmaxpp
cmaxpp_results_dict = pickle.load(
    open(osp.join(path, 'cmaxpp_3d_results.pkl'), 'rb'))
cmaxpp_results_dict = cmaxpp_results_dict[1]
cmaxpp_results_successful = []
cmaxpp_results_unsuccessful = []
for seed in cmaxpp_results_dict.keys():
    trial_returns = cmaxpp_results_dict[seed]
    # trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        cmaxpp_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        cmaxpp_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='red',
        #          alpha=0.2)

cmaxpp_mean_results_successful = np.mean(cmaxpp_results_successful, axis=0)
cmaxpp_std_results_successful = np.std(
    cmaxpp_results_successful, axis=0) / np.sqrt(len(cmaxpp_results_successful))

# Plot cmaxpp
plt.plot(range(max_trials), cmaxpp_mean_results_successful, color='red', linestyle='-',
         label='\\textsc{Cmax++} '+str(len(cmaxpp_results_successful)) + '/' + str(len(cmaxpp_results_dict.keys())))
plt.fill_between(range(max_trials),
                 cmaxpp_mean_results_successful - cmaxpp_std_results_successful,
                 cmaxpp_mean_results_successful + cmaxpp_std_results_successful,
                 color='red', alpha=0.5)
plt.scatter(cmaxpp_results_unsuccessful, [
            0 for _ in range(len(cmaxpp_results_unsuccessful))], marker='x',
            color='red')

# Read adaptive_cmaxpp
adaptive_cmaxpp_results_dict = pickle.load(
    open(osp.join(path, 'adaptive_cmaxpp_3d_results.pkl'), 'rb'))
adaptive_cmaxpp_results_dict = adaptive_cmaxpp_results_dict[4]
adaptive_cmaxpp_results_successful = []
adaptive_cmaxpp_results_unsuccessful = []
for seed in adaptive_cmaxpp_results_dict.keys():
    trial_returns = adaptive_cmaxpp_results_dict[seed]
    # trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        adaptive_cmaxpp_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        adaptive_cmaxpp_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='green',
        #          alpha=0.2)

adaptive_cmaxpp_mean_results_successful = np.mean(
    adaptive_cmaxpp_results_successful, axis=0)
adaptive_cmaxpp_std_results_successful = np.std(
    adaptive_cmaxpp_results_successful, axis=0) / np.sqrt(len(adaptive_cmaxpp_results_successful))

# Plot adaptive_cmaxpp
plt.plot(range(max_trials), adaptive_cmaxpp_mean_results_successful, color='green', linestyle='-',
         label='\\textsc{Adaptive Cmax++} '+str(len(adaptive_cmaxpp_results_successful)) + '/' + str(len(adaptive_cmaxpp_results_dict.keys())))
plt.fill_between(range(max_trials),
                 adaptive_cmaxpp_mean_results_successful -
                 adaptive_cmaxpp_std_results_successful,
                 adaptive_cmaxpp_mean_results_successful +
                 adaptive_cmaxpp_std_results_successful,
                 color='green', alpha=0.5)
plt.scatter(adaptive_cmaxpp_results_unsuccessful, [
            0 for _ in range(len(adaptive_cmaxpp_results_unsuccessful))], marker='x',
            color='green')

plt.yscale('log')
plt.show()

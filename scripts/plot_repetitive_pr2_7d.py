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

# Read cmax
cmax_results = pickle.load(
    open(osp.join(path, 'cmax_7d_approximate_results.pkl'), 'rb'))
cmax_results_dict = cmax_results[4]

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

# cmax_results = np.cumsum(cmax_results, axis=1)
# cmax_mean_results = np.mean(cmax_results, axis=0)
# cmax_std_results = np.std(cmax_results, axis=0) / \
#     np.sqrt(cmax_results.shape[0])

# Read cmaxpp
cmaxpp_results = pickle.load(
    open(osp.join(path, 'cmaxpp_7d_approximate_results.pkl'), 'rb'))
cmaxpp_results_dict = cmaxpp_results[4]

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
adaptive_cmaxpp_results = pickle.load(
    open(osp.join(path, 'adaptive_cmaxpp_7d_approximate_results.pkl'), 'rb'))
adaptive_cmaxpp_results_dict = adaptive_cmaxpp_results[1]

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
         label='Adaptive \\textsc{Cmax++} '+str(len(adaptive_cmaxpp_results_successful)) + '/' + str(len(adaptive_cmaxpp_results_dict.keys())))
plt.fill_between(range(max_trials),
                 adaptive_cmaxpp_mean_results_successful -
                 adaptive_cmaxpp_std_results_successful,
                 adaptive_cmaxpp_mean_results_successful +
                 adaptive_cmaxpp_std_results_successful,
                 color='green', alpha=0.5)
plt.scatter(adaptive_cmaxpp_results_unsuccessful, [
            0 for _ in range(len(adaptive_cmaxpp_results_unsuccessful))], marker='x',
            color='green')

# Read model
model_results_dict = pickle.load(
    open(osp.join(path, 'model_7d_approximate_results.pkl'), 'rb'))

model_results_successful = []
model_results_unsuccessful = []
for seed in model_results_dict.keys():
    trial_returns = model_results_dict[seed]
    # trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        model_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        model_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='green',
        #          alpha=0.2)

model_mean_results_successful = np.mean(
    model_results_successful, axis=0)
model_std_results_successful = np.std(
    model_results_successful, axis=0) / np.sqrt(len(model_results_successful))

# Plot model
plt.plot(range(max_trials), model_mean_results_successful, color='magenta', linestyle='-',
         label='NN Residual Model '+str(len(model_results_successful)) + '/' + str(len(model_results_dict.keys())))
plt.fill_between(range(max_trials),
                 model_mean_results_successful -
                 model_std_results_successful,
                 model_mean_results_successful +
                 model_std_results_successful,
                 color='magenta', alpha=0.5)
plt.scatter(model_results_unsuccessful, [
            0 for _ in range(len(model_results_unsuccessful))], marker='x',
            color='magenta')

# Read knn
knn_results_dict = pickle.load(
    open(osp.join(path, 'knn_7d_approximate_results.pkl'), 'rb'))

knn_results_successful = []
knn_results_unsuccessful = []
for seed in knn_results_dict.keys():
    trial_returns = knn_results_dict[seed]
    # trial_returns = np.cumsum(trial_returns)
    if len(trial_returns) == max_trials:
        # Finished all trials
        knn_results_successful.append(trial_returns)
    else:
        # Not finished all trials
        knn_results_unsuccessful.append(len(trial_returns))
        # plt.plot(range(len(trial_returns)), trial_returns, color='green',
        #          alpha=0.2)

knn_mean_results_successful = np.mean(
    knn_results_successful, axis=0)
knn_std_results_successful = np.std(
    knn_results_successful, axis=0) / np.sqrt(len(knn_results_successful))

# Plot knn
plt.plot(range(max_trials), knn_mean_results_successful, color='cyan', linestyle='-',
         label='KNN Residual Model '+str(len(knn_results_successful)) + '/' + str(len(knn_results_dict.keys())))
plt.fill_between(range(max_trials),
                 knn_mean_results_successful -
                 knn_std_results_successful,
                 knn_mean_results_successful +
                 knn_std_results_successful,
                 color='cyan', alpha=0.5)
plt.scatter(knn_results_unsuccessful, [
            0 for _ in range(len(knn_results_unsuccessful))], marker='x',
            color='cyan')

plt.legend()
plt.show()

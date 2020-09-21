import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import os.path as osp
import argparse

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

fig = plt.figure()
max_laps = 200
path = os.path.join(
    os.environ['HOME'],
    'workspaces/szeth_ws/src/szeth/data/')

# ADAPTIVE
load_path = os.path.join(path, 'adaptive_cmaxpp_ice_friction_results.pkl')
adaptive_cmaxpp_results = pickle.load(open(load_path, 'rb'))
adaptive_cmaxpp_lap_t_steps_dict = {}
adaptive_cmaxpp_lap_returns_dict = {}
for alpha in adaptive_cmaxpp_results.keys():
    adaptive_cmaxpp_lap_t_steps_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_t_steps']
    adaptive_cmaxpp_lap_returns_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_returns']


# CMAXPP
load_path = os.path.join(path, 'cmaxpp_ice_friction_results.pkl')
cmaxpp_results = pickle.load(open(load_path, 'rb'))
cmaxpp_results = cmaxpp_results[1]
cmaxpp_lap_t_steps = cmaxpp_results['lap_t_steps']
cmaxpp_lap_returns = cmaxpp_results['lap_returns']

# CMAX
load_path = os.path.join(path, 'cmax_ice_friction_results.pkl')
cmax_results = pickle.load(open(load_path, 'rb'))
cmax_results = cmax_results[1]
cmax_lap_t_steps = cmax_results['lap_t_steps']
cmax_lap_returns = cmax_results['lap_returns']


# Make them cumulative
adaptive_cmaxpp_lap_returns_dict[100] = [
    np.cumsum(l) for l in adaptive_cmaxpp_lap_returns_dict[100]]
cmaxpp_lap_returns = [np.cumsum(l) for l in cmaxpp_lap_returns]
cmax_lap_returns = [np.cumsum(l) for l in cmax_lap_returns]

# Make them cumulative
adaptive_cmaxpp_lap_t_steps_dict[100] = [
    np.cumsum(l) for l in adaptive_cmaxpp_lap_t_steps_dict[100]]
cmaxpp_lap_t_steps = [np.cumsum(l) for l in cmaxpp_lap_t_steps]
cmax_lap_t_steps = [np.cumsum(l) for l in cmax_lap_t_steps]

# Constants
RETURN_MAX = 1e7
LAP_MAX = 1e5

# Plot returns
# plt.subplot(1, 2, 1)

adaptive_cmaxpp_lap_returns_successful = []
adaptive_cmaxpp_laps_unsuccessful = []
adaptive_cmaxpp_unsuccessful_idxs = []
idx = 0
for lap_returns in adaptive_cmaxpp_lap_returns_dict[100]:
    # plt.plot(range(len(lap_returns)), lap_returns, alpha=0.2,
    #          color='red', linestyle='-')
    if len(lap_returns) == max_laps:
        # Finished all laps
        adaptive_cmaxpp_lap_returns_successful.append(lap_returns)
    else:
        # Did not finish all laps
        adaptive_cmaxpp_laps_unsuccessful.append(len(lap_returns) - 1)
        lap_returns[-1] = RETURN_MAX
        adaptive_cmaxpp_unsuccessful_idxs.append(idx)
        # plt.plot(range(len(lap_returns)), lap_returns,
        #          color = 'green', alpha = 0.2)
    idx += 1

mean_adaptive_cmaxpp_lap_returns_successful = np.mean(
    adaptive_cmaxpp_lap_returns_successful, axis=0)
std_adaptive_cmaxpp_lap_returns_successful = np.std(
    adaptive_cmaxpp_lap_returns_successful, axis=0) / np.sqrt(len(adaptive_cmaxpp_lap_returns_successful))
plt.plot(range(max_laps), mean_adaptive_cmaxpp_lap_returns_successful, color='green', linestyle='-',
         label='ADAPTIVE \\textsc{Cmax++} '+str(len(adaptive_cmaxpp_lap_returns_successful))+'/'+str(len(adaptive_cmaxpp_lap_returns_dict[100])))
plt.fill_between(range(max_laps),
                 mean_adaptive_cmaxpp_lap_returns_successful -
                 std_adaptive_cmaxpp_lap_returns_successful,
                 mean_adaptive_cmaxpp_lap_returns_successful +
                 std_adaptive_cmaxpp_lap_returns_successful,
                 color='green', alpha=0.5)
plt.scatter(adaptive_cmaxpp_laps_unsuccessful, [0 for _ in range(len(adaptive_cmaxpp_laps_unsuccessful))],
            marker='x', color='green')

# Plot cmaxpp
cmaxpp_lap_returns_successful = []
cmaxpp_laps_unsuccessful = []
cmaxpp_unsuccessful_idxs = []
idx = 0
for lap_returns in cmaxpp_lap_returns:
    # plt.plot(range(len(lap_returns)), lap_returns, alpha=0.2,
    #          color='red', linestyle='-')
    if len(lap_returns) == max_laps:
        # Finished all laps
        cmaxpp_lap_returns_successful.append(lap_returns)
    else:
        # Did not finish all laps
        cmaxpp_laps_unsuccessful.append(len(lap_returns) - 1)
        lap_returns[-1] = RETURN_MAX
        cmaxpp_unsuccessful_idxs.append(idx)
        # plt.plot(range(len(lap_returns)), lap_returns, color='red', alpha=0.2)
    idx += 1

mean_cmaxpp_lap_returns_successful = np.mean(
    cmaxpp_lap_returns_successful, axis=0)
std_cmaxpp_lap_returns_successful = np.std(
    cmaxpp_lap_returns_successful, axis=0) / np.sqrt(len(cmaxpp_lap_returns_successful))
plt.plot(range(max_laps), mean_cmaxpp_lap_returns_successful, color='red', linestyle='-',
         label='\\textsc{Cmax++} '+str(len(cmaxpp_lap_returns_successful))+'/'+str(len(cmaxpp_lap_returns)))
plt.fill_between(range(max_laps),
                 mean_cmaxpp_lap_returns_successful - std_cmaxpp_lap_returns_successful,
                 mean_cmaxpp_lap_returns_successful + std_cmaxpp_lap_returns_successful,
                 color='red', alpha=0.5)
plt.scatter(cmaxpp_laps_unsuccessful, [0 for _ in range(len(cmaxpp_laps_unsuccessful))],
            marker='x', color='red')

# Plot cmax
cmax_lap_returns_successful = []
cmax_laps_unsuccessful = []
cmax_unsuccessful_idxs = []
idx = 0
for lap_returns in cmax_lap_returns:
    # plt.plot(range(len(lap_returns)), lap_returns, alpha=0.2,
    #          color='blue', linestyle='-')
    if len(lap_returns) == max_laps:
        # Finished all laps
        cmax_lap_returns_successful.append(lap_returns)
    else:
        # Did not finish all laps
        cmax_laps_unsuccessful.append(len(lap_returns) - 1)
        lap_returns[-1] = RETURN_MAX
        cmax_unsuccessful_idxs.append(idx)
        # plt.plot(range(len(lap_returns)), lap_returns, color='blue', alpha=0.4)
    idx += 1

mean_cmax_lap_returns_successful = np.mean(
    cmax_lap_returns_successful, axis=0)
std_cmax_lap_returns_successful = np.std(
    cmax_lap_returns_successful, axis=0) / np.sqrt(len(cmax_lap_returns_successful))

plt.plot(range(max_laps), mean_cmax_lap_returns_successful, color='blue', linestyle='-',
         label='\\textsc{Cmax} '+str(len(cmax_lap_returns_successful))+'/'+str(len(cmax_lap_returns)))
plt.fill_between(range(max_laps),
                 mean_cmax_lap_returns_successful - std_cmax_lap_returns_successful,
                 mean_cmax_lap_returns_successful + std_cmax_lap_returns_successful,
                 color='blue', alpha=0.5)
plt.scatter(cmax_laps_unsuccessful, [0 for _ in range(len(cmax_laps_unsuccessful))],
            marker='x', color='blue')

plt.legend()
# plt.ylim([-1e3, 8e4])
# plt.ylim([-1.25e5, 1.25e6])
# plt.yscale('log')
plt.ylabel('Cumulative cost')
plt.xlabel('Number of laps')

# Plot t steps
# plt.subplot(1, 2, 2)

# # ADAPTIVE
# load_path = os.path.join(path, 'adaptive_cmaxpp_ice_friction_results.pkl')
# adaptive_cmaxpp_results = pickle.load(open(load_path, 'rb'))
# adaptive_cmaxpp_lap_t_steps_dict = {}
# adaptive_cmaxpp_lap_returns_dict = {}
# for alpha in adaptive_cmaxpp_results.keys():
#     adaptive_cmaxpp_lap_t_steps_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_t_steps']
#     adaptive_cmaxpp_lap_returns_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_returns']

# # CMAXPP
# load_path = os.path.join(path, 'cmaxpp_ice_friction_results.pkl')
# cmaxpp_results = pickle.load(open(load_path, 'rb'))
# cmaxpp_results = cmaxpp_results[1]
# cmaxpp_lap_t_steps = cmaxpp_results['lap_t_steps']
# cmaxpp_lap_returns = cmaxpp_results['lap_returns']

# # CMAX
# load_path = os.path.join(path, 'cmax_ice_friction_results.pkl')
# cmax_results = pickle.load(open(load_path, 'rb'))
# cmax_results = cmax_results[1]
# cmax_lap_t_steps = cmax_results['lap_t_steps']
# cmax_lap_returns = cmax_results['lap_returns']

# adaptive_cmaxpp_lap_returns_extracted = []
# cmaxpp_lap_returns_extracted = []
# cmax_lap_returns_extracted = []
# for idx in range(len(cmax_lap_returns)):
#     if idx not in cmax_unsuccessful_idxs:
#         adaptive_cmaxpp_lap_returns_extracted.append(
#             # np.cumsum(adaptive_cmaxpp_lap_returns_dict[100][idx])
#             adaptive_cmaxpp_lap_returns_dict[100][idx]
#         )
#         cmaxpp_lap_returns_extracted.append(
#             # np.cumsum(cmaxpp_lap_returns[idx])
#             cmaxpp_lap_returns[idx]
#         )
#         cmax_lap_returns_extracted.append(
#             # np.cumsum(cmax_lap_returns[idx])
#             cmax_lap_returns[idx]
#         )


# mean_adaptive_cmaxpp_lap_returns = np.mean(
#     adaptive_cmaxpp_lap_returns_extracted, axis=0)
# std_adaptive_cmaxpp_lap_returns = np.std(
#     adaptive_cmaxpp_lap_returns_extracted, axis=0) / np.sqrt(len(adaptive_cmaxpp_lap_returns_extracted))

# # Plot adaptive cmax++
# plt.plot(range(max_laps), mean_adaptive_cmaxpp_lap_returns, color='green', linestyle='-',
#          label='Adaptive \\textsc{Cmax++}')
# plt.fill_between(range(max_laps),
#                  mean_adaptive_cmaxpp_lap_returns - std_adaptive_cmaxpp_lap_returns,
#                  mean_adaptive_cmaxpp_lap_returns + std_adaptive_cmaxpp_lap_returns,
#                  color='green', alpha=0.5)

# mean_cmaxpp_lap_returns = np.mean(
#     cmaxpp_lap_returns_extracted, axis=0)
# std_cmaxpp_lap_returns = np.std(
#     cmaxpp_lap_returns_extracted, axis=0) / np.sqrt(len(cmaxpp_lap_returns_extracted))

# # Plot cmax++
# plt.plot(range(max_laps), mean_cmaxpp_lap_returns, color='red', linestyle='-',
#          label='\\textsc{Cmax++}')
# plt.fill_between(range(max_laps),
#                  mean_cmaxpp_lap_returns - std_cmaxpp_lap_returns,
#                  mean_cmaxpp_lap_returns + std_cmaxpp_lap_returns,
#                  color='red', alpha=0.5)

# mean_cmax_lap_returns = np.mean(
#     cmax_lap_returns_extracted, axis=0)
# std_cmax_lap_returns = np.std(
#     cmax_lap_returns_extracted, axis=0) / np.sqrt(len(cmax_lap_returns_extracted))

# # Plot cmax
# plt.plot(range(max_laps), mean_cmax_lap_returns, color='blue', linestyle='-',
#          label='\\textsc{Cmax}')
# plt.fill_between(range(max_laps),
#                  mean_cmax_lap_returns - std_cmax_lap_returns,
#                  mean_cmax_lap_returns + std_cmax_lap_returns,
#                  color='blue', alpha=0.5)

# plt.yscale('log')

# adaptive_cmaxpp_lap_t_steps_successful = []
# adaptive_cmaxpp_laps_unsuccessful = []
# for lap_t_steps in adaptive_cmaxpp_lap_t_steps_dict[100]:
#     # plt.plot(range(len(lap_t_steps)), lap_t_steps, alpha=0.2,
#     #          color='red', linestyle='-')
#     if len(lap_t_steps) == max_laps:
#         # Finished all laps
#         adaptive_cmaxpp_lap_t_steps_successful.append(lap_t_steps)
#     else:
#         # Did not finish all laps
#         adaptive_cmaxpp_laps_unsuccessful.append(len(lap_t_steps) - 1)
#         lap_t_steps[-1] = LAP_MAX
#         plt.plot(range(len(lap_t_steps)), lap_t_steps,
#                  color='green', alpha=0.2)

# mean_adaptive_cmaxpp_lap_t_steps_successful = np.mean(
#     adaptive_cmaxpp_lap_t_steps_successful, axis=0)
# std_adaptive_cmaxpp_lap_t_steps_successful = np.std(
#     adaptive_cmaxpp_lap_t_steps_successful, axis=0) / np.sqrt(len(adaptive_cmaxpp_lap_t_steps_successful))
# plt.plot(range(max_laps), mean_adaptive_cmaxpp_lap_t_steps_successful, color='green', linestyle='-',
#          label='ADAPTIVE \\textsc{Cmax++} '+str(len(adaptive_cmaxpp_lap_t_steps_successful))+'/'+str(len(adaptive_cmaxpp_lap_t_steps_dict[100])))
# plt.fill_between(range(max_laps),
#                  mean_adaptive_cmaxpp_lap_t_steps_successful -
#                  std_adaptive_cmaxpp_lap_t_steps_successful,
#                  mean_adaptive_cmaxpp_lap_t_steps_successful +
#                  std_adaptive_cmaxpp_lap_t_steps_successful,
#                  color='green', alpha=0.5)
# plt.scatter(adaptive_cmaxpp_laps_unsuccessful, [0 for _ in range(len(adaptive_cmaxpp_laps_unsuccessful))],
#             marker='x', color='green')

# # Plot cmaxpp
# cmaxpp_lap_t_steps_successful = []
# cmaxpp_laps_unsuccessful = []
# for lap_t_steps in cmaxpp_lap_t_steps:
#     # plt.plot(range(len(lap_t_steps)), lap_t_steps, alpha=0.2,
#     #          color='red', linestyle='-')
#     if len(lap_t_steps) == max_laps:
#         # Finished all laps
#         cmaxpp_lap_t_steps_successful.append(lap_t_steps)
#     else:
#         # Did not finish all laps
#         cmaxpp_laps_unsuccessful.append(len(lap_t_steps) - 1)
#         lap_t_steps[-1] = LAP_MAX
#         plt.plot(range(len(lap_t_steps)), lap_t_steps, color='red', alpha=0.2)

# mean_cmaxpp_lap_t_steps_successful = np.mean(
#     cmaxpp_lap_t_steps_successful, axis=0)
# std_cmaxpp_lap_t_steps_successful = np.std(
#     cmaxpp_lap_t_steps_successful, axis=0) / np.sqrt(len(cmaxpp_lap_t_steps_successful))
# plt.plot(range(max_laps), mean_cmaxpp_lap_t_steps_successful, color='red', linestyle='-',
#          label='\\textsc{Cmax++} '+str(len(cmaxpp_lap_t_steps_successful))+'/'+str(len(cmaxpp_lap_t_steps)))
# plt.fill_between(range(max_laps),
#                  mean_cmaxpp_lap_t_steps_successful - std_cmaxpp_lap_t_steps_successful,
#                  mean_cmaxpp_lap_t_steps_successful + std_cmaxpp_lap_t_steps_successful,
#                  color='red', alpha=0.5)
# plt.scatter(cmaxpp_laps_unsuccessful, [0 for _ in range(len(cmaxpp_laps_unsuccessful))],
#             marker='x', color='red')

# # Plot cmax
# cmax_lap_t_steps_successful = []
# cmax_laps_unsuccessful = []
# for lap_t_steps in cmax_lap_t_steps:
#     # plt.plot(range(len(lap_t_steps)), lap_t_steps, alpha=0.2,
#     #          color='blue', linestyle='-')
#     if len(lap_t_steps) == max_laps:
#         # Finished all laps
#         cmax_lap_t_steps_successful.append(lap_t_steps)
#     else:
#         # Did not finish all laps
#         cmax_laps_unsuccessful.append(len(lap_t_steps) - 1)
#         lap_t_steps[-1] = LAP_MAX
#         plt.plot(range(len(lap_t_steps)), lap_t_steps, color='blue', alpha=0.4)

# mean_cmax_lap_t_steps_successful = np.mean(
#     cmax_lap_t_steps_successful, axis=0)
# std_cmax_lap_t_steps_successful = np.std(
#     cmax_lap_t_steps_successful, axis=0) / np.sqrt(len(cmax_lap_t_steps_successful))

# plt.plot(range(max_laps), mean_cmax_lap_t_steps_successful, color='blue', linestyle='-',
#          label='\\textsc{Cmax} '+str(len(cmax_lap_t_steps_successful))+'/'+str(len(cmax_lap_t_steps)))
# plt.fill_between(range(max_laps),
#                  mean_cmax_lap_t_steps_successful - std_cmax_lap_t_steps_successful,
#                  mean_cmax_lap_t_steps_successful + std_cmax_lap_t_steps_successful,
#                  color='blue', alpha=0.5)
# plt.scatter(cmax_laps_unsuccessful, [0 for _ in range(len(cmax_laps_unsuccessful))],
#             marker='x', color='blue')

# plt.legend()
# # plt.ylim([-1e3, 8e4])
# # plt.ylim([-1e5, 2e6])
# # plt.yscale('log')
# plt.ylim([-6e3, 6e4])
# plt.ylabel('Cumulative steps to goal')
# plt.xlabel('Number of laps')

fig.set_size_inches(10, 8)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/car_racing.pdf'))

plt.show()

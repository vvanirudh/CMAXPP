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
plt.rcParams.update({'font.size': 25})

parser = argparse.ArgumentParser()
parser.add_argument('--count_all', action='store_true')
args = parser.parse_args()

max_laps = 200
max_tsteps = 10000
path = os.path.join(
    os.environ['HOME'],
    'workspaces/szeth_ws/src/szeth/data/')

# ADAPTIVE
load_path = os.path.join(path, 'adaptive_cmaxpp_ice_friction_results.pkl')
adaptive_cmaxpp_results = pickle.load(open(load_path, 'rb'))
# adaptive_cmaxpp_lap_t_steps_dict = {}
# adaptive_cmaxpp_lap_t_steps_dict = {}
# for alpha in adaptive_cmaxpp_results.keys():
#     adaptive_cmaxpp_lap_t_steps_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_t_steps']
#     adaptive_cmaxpp_lap_t_steps_dict[alpha] = adaptive_cmaxpp_results[alpha]['lap_t_steps']
# adaptive_cmaxpp_lap_t_steps = adaptive_cmaxpp_results[100]['lap_t_steps']
adaptive_cmaxpp_lap_t_steps = adaptive_cmaxpp_results[100]['lap_t_steps']


# CMAXPP
load_path = os.path.join(path, 'cmaxpp_ice_friction_results.pkl')
cmaxpp_results = pickle.load(open(load_path, 'rb'))
cmaxpp_results = cmaxpp_results[1]
# cmaxpp_lap_t_steps = cmaxpp_results['lap_t_steps']
cmaxpp_lap_t_steps = cmaxpp_results['lap_t_steps']

# CMAX
load_path = os.path.join(path, 'cmax_ice_friction_results.pkl')
cmax_results = pickle.load(open(load_path, 'rb'))
cmax_results = cmax_results[1]
# cmax_lap_t_steps = cmax_results['lap_t_steps']
cmax_lap_t_steps = cmax_results['lap_t_steps']


# Bar plot
laps = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
# laps = [0, 40, 80, 120, 160, 199]
width = 5

cmax_lap_t_steps_extracted = []
cmax_laps = []
cmax_counts = []
cmaxpp_lap_t_steps_extracted = []
cmaxpp_laps = []
cmaxpp_counts = []
adaptive_cmaxpp_lap_t_steps_extracted = []
adaptive_cmaxpp_laps = []
adaptive_cmaxpp_counts = []
for lap in laps:
    cmax_lap_return = 0
    count = 0
    count_success = 0
    for lap_t_steps in cmax_lap_t_steps:
        if len(lap_t_steps) > lap:
            cmax_lap_return += lap_t_steps[lap]
            count += 1
            count_success += 1
        else:
            cmax_lap_return += max_tsteps
            count += 1
    if count != 0:
        cmax_lap_return = cmax_lap_return / count
        cmax_laps.append(lap)
        cmax_lap_t_steps_extracted.append(cmax_lap_return)
        cmax_counts.append(count_success)

    cmaxpp_lap_return = 0
    count = 0
    count_success = 0
    for lap_t_steps in cmaxpp_lap_t_steps:
        if len(lap_t_steps) > lap:
            cmaxpp_lap_return += lap_t_steps[lap]
            count += 1
            count_success += 1
        else:
            cmaxpp_lap_return += max_tsteps
            count += 1
    if count != 0:
        cmaxpp_lap_return = cmaxpp_lap_return / count
        cmaxpp_laps.append(lap)
        cmaxpp_lap_t_steps_extracted.append(cmaxpp_lap_return)
        cmaxpp_counts.append(count_success)

    adaptive_cmaxpp_lap_return = 0
    count = 0
    count_success = 0
    for lap_t_steps in adaptive_cmaxpp_lap_t_steps:
        if len(lap_t_steps) > lap:
            adaptive_cmaxpp_lap_return += lap_t_steps[lap]
            count += 1
            count_success += 1
        else:
            adaptive_cmaxpp_lap_return += max_tsteps
            count += 1
    if count != 0:
        adaptive_cmaxpp_lap_return = adaptive_cmaxpp_lap_return / count
        adaptive_cmaxpp_laps.append(lap)
        adaptive_cmaxpp_lap_t_steps_extracted.append(
            adaptive_cmaxpp_lap_return)
        adaptive_cmaxpp_counts.append(count_success)

# Convert into np arrays
cmax_laps, cmaxpp_laps, adaptive_cmaxpp_laps = np.array(
    cmax_laps), np.array(cmaxpp_laps), np.array(adaptive_cmaxpp_laps)

fig, ax = plt.subplots()
rects_cmax = ax.bar(cmax_laps - width,
                    cmax_lap_t_steps_extracted, width, label='\\textsc{Cmax}')
rects_cmaxpp = ax.bar(cmaxpp_laps,
                      cmaxpp_lap_t_steps_extracted, width, label='\\textsc{Cmax}++')
rects_adaptive_cmaxpp = ax.bar(adaptive_cmaxpp_laps + width,
                               adaptive_cmaxpp_lap_t_steps_extracted, width,
                               label='\\textsc{A-Cmax}++')


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

# Create twin plot
# ax2 = ax.twinx()
# ax2.plot(cmax_laps, cmax_counts)
# ax2.grid(False)

ax.set_xlabel('Lap')
ax.set_ylabel('Average number of steps taken to finish lap')
ax.set_xticks(laps)
ax.set_xticklabels([1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
# ax.set_xticklabels([0, 40, 80, 120, 160, 200])
ax.legend()
ax.set_yscale('log')
ax.set_ylim(top=3e4)
ax.set_title('3D Mobile Robot Navigation Experiment')

fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/car_racing_bar.pdf'))
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import os.path as osp

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 25})

fig = plt.figure()
path = os.path.join(os.environ['HOME'],
                    'workspaces/szeth_ws/src/szeth/data/')
max_laps = 200
cmap = plt.get_cmap('tab10')

# EXP experiments
alphas = [10.0, 100.0, 1000.0]
exp_steps = [0.5, 0.7, 0.9]

idx = 0
size = len(alphas) * len(exp_steps)
for alpha in alphas:
    for exp in exp_steps:
        load_path = osp.join(
            path,
            'adaptive_cmaxpp_ice_friction_results_'+str(alpha)+'_exp_'+str(exp)+'.pkl')
        results = pickle.load(open(load_path, 'rb'))
        results = results[alpha]['lap_t_steps']
        results = [np.cumsum(l) for l in results]
        mean_results = np.mean(results, axis=0)
        std_results = np.std(results, axis=0) / np.sqrt(10)
        plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
                 label='EXP $\\beta_1$='+str(alpha)+' $\\rho$='+str(exp), linewidth=4)
        plt.fill_between(range(max_laps), mean_results - std_results,
                         mean_results + std_results,
                         color=cmap(idx / size), alpha=0.2)
        idx += 1

plt.ylabel('Cumulative Number of Steps taken to reach goal')
plt.xlabel('Laps')
plt.title(
    'Varying $\\beta_{i+1} = \\rho\\beta_i$ according to an exponential schedule and $\\alpha_i = 1 + \\beta_i$')
plt.legend()
plt.yscale('log')
# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/alpha_exp.pdf'))

# plt.show()()

# LINEAR experiments
fig = plt.figure()
alphas = [10.0, 100.0, 200.0]

idx = 0
size = len(alphas)
for alpha in alphas:
    load_path = osp.join(path,
                         'adaptive_cmaxpp_ice_friction_results_'+str(alpha)+'_linear_.pkl')
    results = pickle.load(open(load_path, 'rb'))
    results = results[alpha]['lap_t_steps']
    results = [np.cumsum(l) for l in results]
    mean_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0) / np.sqrt(10)
    step = alpha / max_laps
    plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
             label='LINEAR $\\beta_1$='+str(alpha)+' $\\eta$='+str(step), linewidth=4)
    plt.fill_between(range(max_laps), mean_results - std_results,
                     mean_results + std_results,
                     color=cmap(idx / size), alpha=0.2)
    idx += 1

plt.ylabel('Cumulative Number of Steps taken to reach goal')
plt.xlabel('Laps')
plt.title(
    'Varying $\\beta_{i+1} = \\beta_i - \\eta$ according to a linear schedule and $\\alpha_i = 1 + \\beta_i$')
plt.legend()
plt.yscale('log')
# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/alpha_linear.pdf'))

# plt.show()()

# TIME experiments
fig = plt.figure()
alphas = [10.0, 100.0, 1000.0]

idx = 0
size = len(alphas)
for alpha in alphas:
    load_path = osp.join(path,
                         'adaptive_cmaxpp_ice_friction_results_'+str(alpha)+'_time_.pkl')
    results = pickle.load(open(load_path, 'rb'))
    results = results[alpha]['lap_t_steps']
    results = [np.cumsum(l) for l in results]
    mean_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0) / np.sqrt(10)
    step = alpha / max_laps
    plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
             label='TIME $\\beta_1$='+str(alpha), linewidth=4)
    plt.fill_between(range(max_laps), mean_results - std_results,
                     mean_results + std_results,
                     color=cmap(idx / size), alpha=0.2)
    idx += 1

plt.ylabel('Cumulative Number of Steps taken to reach goal')
plt.xlabel('Laps')
plt.title(
    'Varying $\\beta_{i+1} = \\beta_1 / (i+1)$ according to a time decay schedule and $\\alpha_i = 1 + \\beta_i$')
plt.legend()
plt.yscale('log')
# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/alpha_time.pdf'))

# plt.show()()

# STEP experiments
fig = plt.figure()
alphas = [10.0, 100.0, 200.0]
steps = [5.0, 10.0, 20.0]

idx = 0
size = len(alphas) * len(exp_steps)
for alpha in alphas:
    for step in steps:
        load_path = osp.join(
            path,
            'adaptive_cmaxpp_ice_friction_results_'+str(alpha)+'_step_'+str(step)+'.pkl')
        results = pickle.load(open(load_path, 'rb'))
        results = results[alpha]['lap_t_steps']
        results = [np.cumsum(l) for l in results]
        mean_results = np.mean(results, axis=0)
        std_results = np.std(results, axis=0) / np.sqrt(10)
        decrease = alpha * step / max_laps
        plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
                 label='STEP $\\beta_1$='+str(alpha)+' $\\xi$='+str(step)+' $\\delta$='+str(decrease), linewidth=4)
        plt.fill_between(range(max_laps), mean_results - std_results,
                         mean_results + std_results,
                         color=cmap(idx / size), alpha=0.2)
        idx += 1

plt.ylabel('Cumulative Number of Steps taken to reach goal')
plt.xlabel('Laps')
plt.title(
    'Varying $\\beta_{i+1} = \\beta_i - \\delta$ if $i$ is a multiple of $\\xi$ according to a step schedule and $\\alpha_i = 1 + \\beta_i$')
plt.legend()
plt.yscale('log')
# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/alpha_step.pdf'))

# plt.show()

# BEST experiments
fig = plt.figure()
idx = 0
load_path = osp.join(
    path,
    'adaptive_cmaxpp_ice_friction_results_'+str(100.0)+'_exp_'+str(0.9)+'.pkl')
results = pickle.load(open(load_path, 'rb'))
results = results[100.0]['lap_t_steps']
results = [np.cumsum(l) for l in results]
mean_results = np.mean(results, axis=0)
std_results = np.std(results, axis=0) / np.sqrt(10)
plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
         label='EXP $\\beta_1$='+str(100.0)+' $\\rho$='+str(0.9), linewidth=4)
plt.fill_between(range(max_laps), mean_results - std_results,
                 mean_results + std_results,
                 color=cmap(idx / size), alpha=0.2)
idx += 1
load_path = osp.join(path,
                     'adaptive_cmaxpp_ice_friction_results_'+str(100.0)+'_linear_.pkl')
results = pickle.load(open(load_path, 'rb'))
results = results[100.0]['lap_t_steps']
results = [np.cumsum(l) for l in results]
mean_results = np.mean(results, axis=0)
std_results = np.std(results, axis=0) / np.sqrt(10)
step = 100.0 / max_laps
plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
         label='LINEAR $\\beta_1$='+str(100.0)+' $\\eta$='+str(step), linewidth=4)
plt.fill_between(range(max_laps), mean_results - std_results,
                 mean_results + std_results,
                 color=cmap(idx / size), alpha=0.2)
idx += 1
load_path = osp.join(path,
                     'adaptive_cmaxpp_ice_friction_results_'+str(100.0)+'_time_.pkl')
results = pickle.load(open(load_path, 'rb'))
results = results[100.0]['lap_t_steps']
results = [np.cumsum(l) for l in results]
mean_results = np.mean(results, axis=0)
std_results = np.std(results, axis=0) / np.sqrt(10)
step = 100.0 / max_laps
plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
         label='TIME $\\beta_1$='+str(100.0), linewidth=4)
plt.fill_between(range(max_laps), mean_results - std_results,
                 mean_results + std_results,
                 color=cmap(idx / size), alpha=0.2)
idx += 1
load_path = osp.join(
    path,
    'adaptive_cmaxpp_ice_friction_results_'+str(100.0)+'_step_'+str(5.0)+'.pkl')
results = pickle.load(open(load_path, 'rb'))
results = results[100.0]['lap_t_steps']
results = [np.cumsum(l) for l in results]
mean_results = np.mean(results, axis=0)
std_results = np.std(results, axis=0) / np.sqrt(10)
decrease = 100.0 * 5.0 / max_laps
plt.plot(range(max_laps), mean_results, color=cmap(idx / size),
         label='STEP $\\beta_1$='+str(100.0)+' $\\xi$='+str(5.0)+' $\\delta$='+str(decrease), linewidth=4)
plt.fill_between(range(max_laps), mean_results - std_results,
                 mean_results + std_results,
                 color=cmap(idx / size), alpha=0.2)

plt.ylabel('Cumulative Number of Steps taken to reach goal')
plt.xlabel('Laps')
plt.title('Comparing the best choices among all schedules')
plt.legend()
plt.yscale('log')
# fig.tight_layout()
fig.set_size_inches(16, 10)
fig.savefig(osp.join(
    os.environ['HOME'], 'workspaces/szeth_ws/src/szeth/plot/alpha_best.pdf'))
plt.show()

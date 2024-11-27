import os
import ioh
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_instance = 500
pbs = [1, 3, 4, 5, 16, 23]
epsilons = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.5]

for problem_id in pbs:
    for instance_id in range(num_instance):
        prob = ioh.get_problem(problem_id, instance_id, dimension=2,
                               problem_class=ioh.ProblemClass.BBOB)
        xopt = prob.optimum.x
        lb = [-5. for _ in range(2)]
        ub = [5. for _ in range(2)]
        abs_diff_with_neg5 = np.abs(xopt - lb)
        abs_diff_with_5 = np.abs(xopt - ub)
        min_abs_diff = np.minimum(abs_diff_with_neg5, abs_diff_with_5)
        sorted_indices = np.argsort(min_abs_diff)
        for epsilon in epsilons[:-1]:
            fpath = f"{problem_id}_{instance_id}_{epsilon}.txt"
            fpath = "data/search_region/" + fpath
            if os.path.exists(fpath):
                continue
            search_region = np.array([[lb[i], ub[i]] for i in range(2)])
            for j in sorted_indices:
                if np.abs(xopt[j] - lb[j]) <= np.abs(xopt[j] - ub[j]):
                    search_region[j][0] = xopt[j] - \
                        (ub[j] - xopt[j]) * epsilon / (1 - epsilon)
                    search_region[j][1] = ub[j]
                else:
                    search_region[j][0] = lb[j]
                    search_region[j][1] = xopt[j] + \
                        (xopt[j] - lb[j]) * epsilon / (1 - epsilon)
            np.savetxt(fpath, search_region)


n_samples = 25000
points = np.random.uniform(size=(n_samples, 2), low=-5, high=5)
norm_points = (points + 5) / 10
for epsilon in epsilons:
    for problem_id in pbs:
        fs = np.zeros((n_samples, num_instance))
        for instance_id in range(num_instance):
            fpath = f"{problem_id}_{instance_id}_{epsilon}.txt"
            fpath = "data/search_region/" + fpath
            f = ioh.get_problem(problem_id, dimension=2, instance=instance_id)
            bounds = np.loadtxt(fpath)
            f.bounds.lb = bounds.T[0]
            f.bounds.ub = bounds.T[1]
            # print(f.bounds.lb, f"{problem_id}_{instance_id}_{epsilon}.txt")
            x = norm_points * (f.bounds.ub - f.bounds.lb) + f.bounds.lb
            y = np.array(f(x))
            fs[:, instance_id] = y - f.optimum.y
        print(f"Saving: data/{problem_id}_{epsilon}_full.npy")
        np.save(f"data/{problem_id}_{epsilon}_full.npy", fs)


for epsilon in epsilons:
    for problem_id in pbs:
        fs = np.load(f"data/{problem_id}_{epsilon}_full.npy")
        l_fs = np.clip(np.log(fs), -10, 25)
        mean_flog = np.mean(l_fs, axis=1)
        dt = pd.DataFrame(points, columns=['x0', 'x1'])
        dt['f'] = mean_flog
        print(f"Saving: data/{problem_id}_{epsilon}_dt.csv")
        dt.to_csv(f"data/{problem_id}_{epsilon}_dt.csv")


fig, axs = plt.subplots(nrows=6, ncols=7, figsize=(15, 15),
                        sharex=True, sharey=True)
for fidx, ax in enumerate(axs.flat):
    problem_id = pbs[int(fidx / 7)]
    epsilon = epsilons[fidx % 7]
    dt = pd.read_csv(f"data/{problem_id}_{epsilon}_dt.csv")
    # plt.figure(figsize=(16,16))
    # ax.scatter(data = dt, x='x0', y='x1', c='f', alpha=0.1)
    sns.scatterplot(data=dt, x='x0', y='x1', hue='f',
                    ax=ax, legend=None, alpha=0.2)
    ax.axhline(5, c='k', ls=':')
    ax.axhline(-5, c='k', ls=':')
    ax.axvline(5, c='k', ls=':')
    ax.axvline(-5, c='k', ls=':')
    ax.set_title(f"fid: {problem_id}, epsilon: {epsilon}")
# plt.subplots_adjust(left=0.01,right=0.01,up=0.01,down=0.01)
plt.tight_layout()
plt.savefig(f"Overall_scatter_avg_v3.png")

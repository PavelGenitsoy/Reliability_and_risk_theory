from scipy import stats
import numpy as np


def count_empty_blocks(x, y):
    return len([1 for i in range(1, len(x)) if not any((x[i - 1] < y) & (x[i] > y))])


def task_1_A_empty_boxes(lyambda, size):
    sample_exp_x = np.sort(np.random.exponential(size=size[0]))
    sample_exp_y = np.random.exponential(scale=1 / lyambda, size=size[1])

    r_val = int(size[1] / size[0])
    count_zero_boxes = count_empty_blocks(sample_exp_x, sample_exp_y)
    limit = size[0] / (1 + r_val) + np.sqrt(size[0]) * (r_val / np.power(1 + r_val, 3/2)) * stats.norm.ppf(1 - gamma)

    print(f"\tn = {size[0]}, m = {size[1]}", file=file)
    if count_zero_boxes < limit:
        print(f"\t\tcount_zero_blocks = {count_zero_boxes}; critical_limit = {limit} || Statistics do not "
              f"contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tcount_zero_blocks = {count_zero_boxes}; critical_limit = {limit} || An alternative "
              f"hypothesis should be accepted H_1", file=file)


#######################################################################################################################


def v_frequencies(samples, u):
    v_arr = np.zeros((len(samples), len(u)))
    for it, list_val in enumerate(samples):
        for i in range(len(u) - 1):
            v_arr[it][i] = len(list_val[(list_val >= u[i]) & (list_val < u[i + 1])])
    return v_arr


def task_1_B_chi_square(lyambda, size):
    sample_exp_x = np.random.exponential(size=size[0])
    sample_exp_y = np.random.exponential(size=size[1])
    sample_exp_z = np.random.exponential(scale=1 / lyambda, size=size[2])
    samples = [sample_exp_x, sample_exp_y, sample_exp_z]

    r_interval = int(20 * np.array(size).sum() / 1000)
    u_interval = np.linspace(0, np.maximum(np.maximum(np.max(samples[0]), np.max(samples[1])), np.max(samples[2])),
                             r_interval)

    z_value = stats.chi2.ppf(1 - gamma, (r_interval - 1) * (len(samples) - 1))

    v_j_frequencies_arr = np.array(v_frequencies(samples, u_interval))

    v_j = v_j_frequencies_arr.sum(axis=0)
    n_i = v_j_frequencies_arr.sum(axis=1)  # n_array (for every sample)
    n_val = n_i.sum()

    tmp = 0
    for i in range(len(samples)):
        for j in range(r_interval - 1):
            tmp += (v_j_frequencies_arr[i][j] - n_i[i] * v_j[j] / n_val)**2 / (n_i[i] * v_j[j] + 1e-10)
    delta = n_val * tmp

    print(f"\tn = {size[0]}, m = {size[1]}, k = {size[2]}", file=file)
    if delta < z_value:
        print(f"\t\tdelta = {delta}; z = {z_value} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tdelta = {delta}; z = {z_value} || An alternative hypothesis should be accepted H_1", file=file)


#######################################################################################################################


if __name__ == '__main__':
    gamma = 0.05
    lyambda = [1, 1.1]
    value_param = {0: [[500, 1000], [5000, 10000], [50000, 100000]],  # n, m
                   1: [[200, 600, 400], [2000, 6000, 4000], [20000, 60000, 40000]]}  # n, m, k
    dict_for_func = {0: task_1_A_empty_boxes, 1: task_1_B_chi_square}
    test = ['Task_1_A', 'Task_1_B']

    for it, name in enumerate(test):
        with open("Results\\" + name + "_result.txt", "w") as file:
            print(f"{name}:", file=file)
            for size in value_param[it]:
                dict_for_func[it](lyambda[1], size)
            print("\n", file=file)
    print("ALL DONE!")

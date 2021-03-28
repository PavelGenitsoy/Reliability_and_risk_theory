from scipy import stats
import numpy as np
from operator import itemgetter


def count_empty_blocks(x, y):
    return len([1 for i in range(1, len(x)) if not any((x[i - 1] < y) & (x[i] > y))])


def task_1_A_empty_boxes(lyambda, size):
    sample_exp_x = np.sort(np.random.exponential(size=size[0]))
    sample_exp_y = np.random.exponential(scale=1 / lyambda, size=size[1])

    r_val = int(size[1] / size[0])
    count_zero_boxes = count_empty_blocks(sample_exp_x, sample_exp_y)
    limit = size[0] / (1 + r_val) + np.sqrt(size[0]) * (r_val / np.power(1 + r_val, 3 / 2)) * stats.norm.ppf(1 - gamma)

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
    return v_arr.astype(int)


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
        for j in range(r_interval):
            if v_j[j] != 0.0:
                tmp += (v_j_frequencies_arr[i][j] - n_i[i] * v_j[j] / n_val) ** 2
                tmp /= (n_i[i] * v_j[j])
    delta = n_val * tmp

    print(f"\tn = {size[0]}, m = {size[1]}, k = {size[2]}", file=file)
    if delta < z_value:
        print(f"\t\tdelta = {delta}; z = {z_value} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tdelta = {delta}; z = {z_value} || An alternative hypothesis should be accepted H_1", file=file)


#######################################################################################################################


def v_arr_from_intervals(u_x_interval, x_sample, v_y_interval, y_sample):
    v_arr = np.zeros((len(u_x_interval) - 1, len(v_y_interval) - 1))
    for i in range(len(u_x_interval) - 1):
        for j in range(len(v_y_interval) - 1):
            v_arr[i][j] = len(x_sample[(x_sample >= u_x_interval[i]) & (x_sample < u_x_interval[i + 1])]) + \
                          len(y_sample[(y_sample >= v_y_interval[j]) & (y_sample < v_y_interval[j + 1])])
    return v_arr.astype(int)


def task_2_A_chi_square(_, size):
    sample_x = np.random.uniform(size=size)
    sample_y = np.random.uniform(low=-1.0, high=1.0, size=size) + sample_x

    if size == 50000:
        r_x_interval = 100
        k_y_interval = 105
    else:
        r_x_interval = int(10 * size / 1000)
        k_y_interval = int(11 * size / 1000)

    u_x_interval = np.linspace(0, np.max(sample_x), r_x_interval)
    v_y_interval = np.linspace(-1, np.max(sample_y), k_y_interval)

    z_value = stats.chi2.ppf(1 - gamma, (r_x_interval - 1) * (k_y_interval - 1))

    v_i_j_arr = v_arr_from_intervals(u_x_interval, sample_x, v_y_interval, sample_y)

    v_j = v_i_j_arr.sum(axis=0)
    v_i = v_i_j_arr.sum(axis=1)

    tmp = 0.0
    for i in range(r_x_interval - 1):
        for j in range(k_y_interval - 1):
            tmp += (v_i_j_arr[i][j] - v_i[i] * v_j[j] / size) ** 2
            tmp /= (v_i[i] * v_j[j])

    delta = size * np.abs(tmp)

    print(f"\tn = {size}, r = {r_x_interval}, k = {k_y_interval}", file=file)
    if delta < z_value:
        print(f"\t\tdelta = {delta}; z = {z_value} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tdelta = {delta}; z = {z_value} || An alternative hypothesis should be accepted H_1", file=file)

#######################################################################################################################


def task_2_B_spearman(_, size):
    sample_x = np.random.uniform(size=size)
    sorted_x = list(np.sort(sample_x))
    sample_y = np.random.uniform(low=-1.0, high=1.0, size=size) + sample_x
    sorted_y = list(np.sort(sample_y))

    z_value = stats.norm.ppf(1 - gamma / 2) / np.sqrt(size)

    list_of_tuples = list(zip([sorted_x.index(elem) for elem in sample_x], [sorted_y.index(elem) for elem in sample_y]))
    r_index, s_index = zip(*sorted(list_of_tuples, key=itemgetter(0)))

    p_value = 1 - (6 / (size * (size**2 - 1)) * np.sum((np.array(r_index) - np.array(s_index)) ** 2))

    print(f"\tn = {size}", file=file)
    if np.abs(p_value) < z_value:
        print(f"\t\tp_value = {np.abs(p_value)}; z = {z_value} || Statistics do not contradict the hypothesis H_0",
              file=file)
    else:
        print(f"\t\tp_value = {np.abs(p_value)}; z = {z_value} || An alternative hypothesis should be accepted H_1",
              file=file)


#######################################################################################################################


def task_2_C_kendell(_, size):
    sample_x = np.random.uniform(size=size)
    sorted_x = list(np.sort(sample_x))
    sample_y = np.random.uniform(low=-1.0, high=1.0, size=size) + sample_x
    sorted_y = list(np.sort(sample_y))

    z_value = 2 * stats.norm.ppf(1 - gamma / 2) / (3 * np.sqrt(size))

    list_of_tuples = list(zip([sorted_x.index(elem) for elem in sample_x], [sorted_y.index(elem) for elem in sample_y]))
    r_index, v_array = zip(*sorted(list_of_tuples, key=itemgetter(0)))
    v_array = np.array(v_array)

    count_of_all_pairs = []
    for i in range(size):
        count_of_all_pairs.append(len(v_array[i + 1:][v_array[i] < v_array[i + 1:]]))

    tau_value = (4 * int(np.sum(count_of_all_pairs))) / (size * (size - 1)) - 1

    print(f"\tn = {size}", file=file)
    if np.abs(tau_value) < z_value:
        print(f"\t\ttau_value = {np.abs(tau_value)}; z = {z_value} || Statistics do not contradict the hypothesis H_0",
              file=file)
    else:
        print(f"\t\ttau_value = {np.abs(tau_value)}; z = {z_value} || An alternative hypothesis should be accepted H_1",
              file=file)


#######################################################################################################################


if __name__ == '__main__':
    gamma = 0.05
    lyambda = [1, 1.1]
    value_param = {0: [[500, 1000], [5000, 10000], [50000, 100000]],  # n, m
                   1: [[200, 600, 400], [2000, 6000, 4000], [20000, 60000, 40000]],  # n, m, k
                   2: [500, 5000, 50000]}  # n
    dict_for_func = {0: task_1_A_empty_boxes, 1: task_1_B_chi_square, 2: task_2_A_chi_square, 3: task_2_B_spearman,
                     4: task_2_C_kendell}
    test = ['Task_1_A', 'Task_1_B', 'Task_2_A', 'Task_2_B', 'Task_2_C']

    for it, name in enumerate(test):
        with open("Results\\" + name + "_result.txt", "w") as file:
            print(f"{name}:", file=file)
            for size in value_param[2 if it > 1 else it]:
                dict_for_func[it](lyambda[1], size)
            print("\n", file=file)
    print("ALL DONE!")

from scipy import stats
import numpy as np


def expcdf(t, lyambda=1):
    return 1.0 - np.exp(-lyambda * t)


def kolmogorov(lyambda, size):
    sample = np.random.exponential(scale=1/lyambda, size=size)
    sample = np.sort(sample)

    k_it = np.array([x for x in range(1, size + 1)])

    d_value = (np.maximum(expcdf(sample) - (k_it - 1) / size, (k_it / size) - expcdf(sample))).max()

    if np.sqrt(size) * d_value < z:
        print(f"\t\tD = {d_value} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tD = {d_value} || An alternative hypothesis should be accepted H_1", file=file)


##############################################################################################################


def v_frequencies(cdf, r):
    left_pointer, right_pointer = 0, 1/r
    result_count = []
    for i in range(r):
        tmp = len(cdf[(cdf >= left_pointer) & (cdf < right_pointer)])
        left_pointer += 1/r
        right_pointer += 1/r
        result_count.append(tmp)
    return result_count


def chi_square(lyambda, size):
    sample_exp_cdf = expcdf(np.random.exponential(scale=1/lyambda, size=size))

    r_interval = int(20 * size / 1000)
    z_value = stats.chi2.ppf(1 - gamma, r_interval - 1)

    v_frequencies_arr = np.array(v_frequencies(sample_exp_cdf, r_interval))  # frequency

    delta = (v_frequencies_arr**2/(size/r_interval)).sum() - size

    if delta < z_value:
        print(f"\t\tdelta = {delta} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tdelta = {delta} || An alternative hypothesis should be accepted H_1", file=file)


##############################################################################################################


if __name__ == '__main__':
    z = 1.36  # at gamma = 0.05
    gamma = 0.05
    n_quantity = [1000, 10000, 100000]
    lyambda = [1, 1.3]
    name_list_criterion = ['kolmogorov', 'chi_square']
    dict_for_func = {0: kolmogorov, 1: chi_square}

    for it, name in enumerate(name_list_criterion):
        with open("Results\\" + name + "_result.txt", "w") as file:
            for l in lyambda:
                print(f"H_0: X_i belongs to F(u, 1), when real is X_i belongs to F(u, {l})", file=file)
                for n in n_quantity:
                    print(f"\tN = {n}", file=file)
                    dict_for_func[it](l, n)
                print("\n", file=file)
    print("ALL DONE!")

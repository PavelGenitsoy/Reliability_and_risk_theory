from scipy import stats
import numpy as np


def expcdf(t, lyambd=1):
    return 1.0 - np.exp(-lyambd * t)


def kolmogorov(lyambda, size):
    sample = np.sort(np.random.exponential(scale=1/lyambda, size=size))

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
        print(f"\t\tdelta = {delta}; z = {z_value} || Statistics do not contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tdelta = {delta}; z = {z_value} || An alternative hypothesis should be accepted H_1", file=file)


##############################################################################################################


def critical_limit(r):
    return r * np.exp(-2) + stats.norm.ppf(1 - gamma) * np.sqrt(r * np.exp(-2) * (1 - 3 * np.exp(-2)))


def empty_boxes(lyambda, size):
    sample_exp_cdf = expcdf(np.random.exponential(scale=1 / lyambda, size=size))

    r_interval = int(size / 2)
    count_zero_boxes = v_frequencies(sample_exp_cdf, r_interval).count(0)
    t_alpha = critical_limit(r_interval)

    if count_zero_boxes < t_alpha:
        print(f"\t\tcount_zero_boxes = {count_zero_boxes}; critical_limit = {t_alpha} || Statistics do not "
              f"contradict the hypothesis H_0", file=file)
    else:
        print(f"\t\tcount_zero_boxes = {count_zero_boxes}; critical_limit = {t_alpha} || An alternative "
              f"hypothesis should be accepted H_1", file=file)


##############################################################################################################


def smirnov(lyambda, size):
    sample_n = np.sort(np.random.exponential(scale=1, size=size))
    m_size = int(size / 2)
    sample_m = np.sort(np.random.exponential(scale=1 / lyambda, size=m_size))

    k_it = np.array([x for x in range(1, m_size + 1)])

    d_value = (np.maximum((k_it / m_size) - expcdf(sample_m), expcdf(sample_m) - (k_it - 1 / m_size))).max()
    critical_val = z * np.sqrt((1 / size) + (1 / m_size))

    if d_value <= critical_val:
        print(f"\t\tD = {d_value}; critical_val = {critical_val} || Statistics do not contradict the hypothesis H_0",
              file=file)
    else:
        print(f"\t\tD = {d_value}; critical_val = {critical_val}|| An alternative hypothesis should be accepted H_1",
              file=file)


##############################################################################################################


if __name__ == '__main__':
    z = 1.36  # at gamma = 0.05
    gamma = 0.05
    dict_for_func = {0: kolmogorov, 1: chi_square, 2: empty_boxes, 3: smirnov}

    for it, name in enumerate(['kolmogorov', 'chi_square', 'empty_boxes', 'smirnov']):
        with open("Results\\" + name + "_result.txt", "w") as file:
            for lyambda in [1, 1.3]:
                print(f"H_0: X_i belongs to F(u, 1), when real is X_i belongs to F(u, {lyambda})", file=file)
                for n in [1000, 10000, 100000]:
                    print(f"\tN = {n}", file=file)
                    dict_for_func[it](lyambda, n)
                print("\n", file=file)
    print("ALL DONE!")

from scipy import stats
import numpy as np


def first_A_task(arr, length):
    """
    Confidence interval of mean which have a normal distribution but the variance is unknown
    """

    df = length - 1  # degree of freedom
    t = stats.t.ppf(1 - alpha / 2, df)  # t-score, ppf - Percent point function
    s = np.std(arr, ddof=1)  # sample standard deviation

    lower = np.mean(arr) - (t * s / np.sqrt(length))
    upper = np.mean(arr) + (t * s / np.sqrt(length))
    return [lower, upper]


def second_B_task(arr, length):
    """
    Confidence interval of mean which have a unknown distribution
    """

    z = stats.norm.ppf(alpha / 2)  # z-score
    s = np.std(arr, ddof=1)  # sample standard deviation

    upper = np.mean(arr) - (z * s / np.sqrt(length))
    lower = np.mean(arr) + (z * s / np.sqrt(length))
    return [lower, upper]


def third_C_task(arr, length):
    """
    Confidence interval of variance which have a normal distribution
    """

    s2 = np.var(arr, ddof=1)
    df = length - 1

    lower = df * s2 / stats.chi2.ppf(1 - alpha / 2, df)
    upper = df * s2 / stats.chi2.ppf(alpha / 2, df)
    return [lower, upper]


if __name__ == '__main__':
    # np.random.seed(42)

    gamma = 0.99
    alpha = 1 - gamma   # 0.01
    number = [pow(10, 2), pow(10, 4), pow(10, 6)]

    for size in number:
        arr = np.random.normal(size=size)
        print(f"N = {size}\n",
              f"first_A_task:  {first_A_task(arr, size)}\n",
              f"second_B_task: {second_B_task(arr, size)}\n",
              f"third_C_task:  {third_C_task(arr, size)}\n")

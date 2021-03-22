from scipy import stats
import numpy as np


def create_ksi(alpha, size):
    uniform_val = np.random.uniform(size=size)
    return (1 / alpha) * np.power(-np.log(uniform_val), 0.25)


def create_etha(size):
    uniform_val = np.random.uniform(size=size)
    return np.power(-np.log(uniform_val), 0.5)


def quantity_of_steps(arr_qu):
    qu_mean = np.mean(arr_qu)
    sigma = np.std(arr_qu, ddof=1)
    return np.power((2.575 * sigma) / 0.01 * qu_mean, 2)


def method_Monte_Carlo(alpha, init_val):
    n = init_val
    while True:
        etha = create_etha(n)
        ksi = create_ksi(alpha, n)

        # val_qu = np.array([(val_etha - val_ksi) > 0 for i, val_etha in enumerate(etha) for j, val_ksi
        #                    in enumerate(ksi) if i == j], dtype=np.uint8)
        val_qu = ((np.array(etha) - np.array(ksi)) > 0).astype(np.uint8)
        if n >= quantity_of_steps(val_qu):
            print(f"Q: {np.mean(val_qu)}", f"num_of_steps(length)={n}")
            break
        #print(n)
        n += 1


if __name__ == '__main__':
    gamma = 0.99
    error = 0.01
    alpha = [1, 0.3, 0.1]
    test_list_init = [1000, 10052400, 5]
    print("first")
    for i, value in enumerate(alpha):
        print(f"alpha: {value}\n", f"method_1(Monte-Carlo): {method_Monte_Carlo(value, test_list_init[i])}")

    #method_Monte_Carlo(1)
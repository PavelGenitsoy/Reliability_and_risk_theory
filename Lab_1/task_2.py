from scipy import stats
import numpy as np


def create_ksi(alpha, size):
    return (1 / alpha) * np.power(-np.log(np.random.uniform(size=size)), 0.25)


def create_etha(size):
    return np.power(-np.log(np.random.uniform(size=size)), 0.5)


def quantity_of_steps(arr_qu):
    return np.power((2.575 * np.std(arr_qu, ddof=1)) / (0.01 * np.mean(arr_qu)), 2)


def method_Monte_Carlo_1(alpha, init_val):
    n = init_val

    if n < 100000000:
        while True:
            etha = create_etha(n)
            ksi = create_ksi(alpha, n)

            # val_qu = np.array([(val_etha - val_ksi) > 0 for i, val_etha in enumerate(etha) for j, val_ksi
            #                    in enumerate(ksi) if i == j], dtype=np.uint8)

            val_qu = ((np.array(etha) - np.array(ksi)) > 0).astype(np.uint8)
            if n >= quantity_of_steps(val_qu):
                print(f"\tmethod_1(Monte-Carlo):\n\t\tQ: {np.mean(val_qu)}", f"|| num_of_steps(length)={n}")
                break
            n += 1
    else:
        print("\tmethod_1(Monte-Carlo):\n\t\tlack of resources due to sample length!!")


def method_2(alpha, init_val):
    n = init_val
    while True:
        ksi = create_ksi(alpha, n)

        val_qu = np.exp(-np.power(ksi, 2))
        if n >= quantity_of_steps(val_qu):
            print(f"\tmethod_2:\n\t\tQ: {np.mean(val_qu)}", f"|| num_of_steps(length)={n}")
            break
        n += 1


def method_3(alpha, init_val):
    n = init_val
    while True:
        etha = create_etha(n)

        val_qu = 1 - np.exp(-np.power(alpha * etha, 4))
        if n >= quantity_of_steps(val_qu):
            print(f"\tmethod_3:\n\t\tQ: {np.mean(val_qu)}", f"|| num_of_steps(length)={n}")
            break
        n += 1


def method_4(alpha, init_val):
    n = init_val
    while True:
        betta = np.sqrt((-np.log(np.random.uniform(size=(3, n)))).sum(axis=0))

        val_qu = 2 * (1 - np.exp(-np.power(alpha * betta, 4))) / np.power(betta, 4)
        if n >= quantity_of_steps(val_qu):
            print(f"\tmethod_4:\n\t\tQ: {np.mean(val_qu)}", f"|| num_of_steps(length)={n}")
            break
        n += 1


if __name__ == '__main__':
    gamma = 0.99
    error = 0.01
    alpha = [1, 0.3, 0.1]
    test_list_init_1 = [78000, 4200000, 330000000]
    test_list_init_2 = [11000, 1035605, 82550000]
    test_list_init_3 = [42500, 282500, 325000]
    test_list_init_4 = [61000, 50, 5]

    for i, value in enumerate(alpha):
        print(f"alpha: {value}")
        method_Monte_Carlo_1(value, test_list_init_1[i])
        method_2(value, test_list_init_2[i])
        method_3(value, test_list_init_3[i])
        method_4(value, test_list_init_4[i])

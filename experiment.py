import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def e(x):
    return 1 / (1 + np.exp(-np.dot(g, x)))


def sigma(x):
    return np.sqrt(1 + 1.25 * (x[0] ** 2))


def estimate_global_propensity(precision=20):
    res = 0
    for x1 in np.linspace(0, 1, precision):
        for x2 in np.linspace(0, 1, precision):
            for x3 in np.linspace(0, 1, precision):
                for x4 in np.linspace(0, 1, precision):
                    x = np.array([x1, x2, x3, x4])
                    res += e(x)
    return res / (precision ** 4)





def r(x):
    return (1 - p_treated) * e(x) / ((1 - e(x)) * p_treated)


def evar0(x, alpha, delta, lower=False):
    if lower:
        return np.dot(b0, x) - delta * sigma(x) - np.sqrt(-2 * np.log(alpha)) * sigma(x)
    return np.dot(b0, x) - delta * sigma(x) + np.sqrt(-2 * np.log(alpha)) * sigma(x)

def evar1(x, alpha, delta, lower=False):
    if lower:
        return np.dot(b1, x) - np.sqrt(-2 * np.log(alpha)) * sigma(x) - delta * sigma(x)
    return np.dot(b1, x) + np.sqrt(-2 * np.log(alpha)) * sigma(x) - delta * sigma(x)


def estimate_bound(rho, lower, delta, precision=20):
    alpha = np.exp(-rho)
    res = 0
    res_l = 0
    res_u = 0
    n = 15_000
    for x1 in np.linspace(0, 1, precision):
        for x2 in np.linspace(0, 1, precision):
            for x3 in np.linspace(0, 1, precision):
                for x4 in np.linspace(0, 1, precision):
                    x = np.array([x1, x2, x3, x4])
                    res += 1/r(x) * (1-e(x)) / (1 - p_treated) * (evar1(x, alpha, delta, lower) - np.dot(b0, x) + delta * sigma(x))
                    res_l += 1/r(x) * (1-e(x)) / (1 - p_treated) * (evar1(x, alpha, delta, lower) - np.dot(b0, x) + delta * sigma(x)) - 2 * sigma(x)
                    res_u += 1/r(x) * (1-e(x)) / (1 - p_treated) * (evar1(x, alpha, delta, lower) - np.dot(b0, x) + delta * sigma(x)) + 2 * sigma(x)
    return res / (precision ** 4), res_u / (precision ** 4), res_l / (precision ** 4)

if __name__ == '__main__':
    g = np.array([-0.531, 0.126, -0.312, 0.018])
    delta = 0.5
    b0 = np.array([-0.531, -0.126, -0.312, 0.671])
    b1 = np.array([0.531, 1.126, -0.312, 0.671])
    precision = 20
    p_treated = estimate_global_propensity(precision)
    rhos = np.linspace(0, 1, 20)
    upper = []
    pos_confidence_upper = []
    neg_confidence_upper = []
    lower = []
    pos_confidence_lower = []
    neg_confidence_lower = []
    for rho in tqdm(rhos):
        # delta = np.
        # sqrt(2 * rho)
        upper_value, upper_conf_pos, upper_cong_neg = estimate_bound(rho, False, delta, precision)
        lower_value, lower_conf_pos, lower_cong_neg = estimate_bound(rho, True, delta, precision)
        upper.append(upper_value)
        pos_confidence_upper.append(upper_conf_pos)
        neg_confidence_upper.append(upper_cong_neg)
        lower.append(lower_value)
        pos_confidence_lower.append(lower_conf_pos)
        neg_confidence_lower.append(lower_cong_neg)
    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2.1, 2.5, 2.9, 3.1, 3.2, 3.4, 3.5], color='green', label='from paper')
    plt.plot([0.05, 1], [1.25, 1.25], color='blue', label='ground truth')
    plt.plot([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.6, 1.25, 1, 0.7, 0.45, 0.25, 0.1], color='green')
    plt.plot(rhos, lower, color='red', label='closed form')
    plt.fill_between(rhos, lower, pos_confidence_lower, color='red', alpha=0.5)
    plt.fill_between(rhos, lower, neg_confidence_lower, color='red', alpha=0.5)
    plt.plot(rhos, upper, color='red')
    plt.fill_between(rhos, upper, pos_confidence_upper, color='red', alpha=0.5)
    plt.fill_between(rhos, upper, neg_confidence_upper, color='red', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Rho')
    plt.ylabel('ATC')
    plt.title("Reproduction of the ATC result made by authors.")
    plt.show()
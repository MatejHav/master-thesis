import numpy as np
import matplotlib.pyplot as plt

from data.generators import *
from sensitivities import *

c = lambda p, v: 1 - (0.5 * (v + 0.5) + 0.5 * p)


def create_generator(u_prob, x_effect, t_effect, y_effect):
    # General settings
    n_rows = 300_000
    n_jobs = 30
    sizes = {
        "U": 1,
        "X": 1,
        "T": 1,
        "Y": 1
    }
    base_x_prob = 0.5
    base_t_prob = 0.5

    # Generators
    u_gen = lambda noise: [0 if np.random.rand() >= u_prob else 1]
    x_gen = lambda u, noise: [0 if np.random.rand() >= x_effect * u[0] + base_x_prob else 1 for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [0 if np.random.rand() >= t_effect * u[0] - 0.1 * x[0] + base_t_prob else 1]
    y_gen = lambda u, x, t, noise: [x[0] + y_effect * u[0] + 2 * t[0] + noise]
    generators = {
        "U": u_gen,
        "X": x_gen,
        "T": t_gen,
        "Y": y_gen
    }
    # Noise generators
    noise = {
        "U": lambda: 0,
        "X": lambda: 0,
        "T": lambda: 0,
        "Y": lambda: np.random.choice(np.arange(-2, 2, 1))
    }

    generator = RegularGenerator(generators=generators, noise_generators=noise, sizes=sizes)
    path = f"./csv_files/data_u{int(100 * u_prob)}_x{int(100 * x_effect)}_t{int(100 * t_effect)}_y{int(100 * y_effect)}.csv"
    df = generator.generate(num_rows=n_rows, n_jobs=n_jobs, path=path)
    return df, path


def plot_setting_distribution(metric_dict, metric_name, logscale=False):
    bins = {}
    fig, ax = plt.subplots(6, 1)
    fig.set_figheight(30)
    fig.set_figwidth(15)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax
    for setting in metric_dict:
        p, x, t, y = setting
        metric = round(metric_dict[setting] if not logscale else np.log(metric_dict[setting]), 1)
        radius = 0.05
        y_coord = radius
        if metric in bins:
            y_coord = bins[metric] + 2 * radius
        bins[metric] = y_coord
        circle = plt.Circle((metric, y_coord), radius, facecolor=(c(p, x), p, p), linewidth=0.05, edgecolor='black')
        ax1.add_patch(circle)
        circle = plt.Circle((metric, y_coord), radius, facecolor=(p, c(p, t), p), linewidth=0.05, edgecolor='black')
        ax2.add_patch(circle)
        circle = plt.Circle((metric, y_coord), radius, facecolor=(p, p, c(p, y)), linewidth=0.05, edgecolor='black')
        ax3.add_patch(circle)
        circle = plt.Circle((metric, y_coord), radius, facecolor=(0.5 - x, 0.5 - t, 0), linewidth=0.05,
                            edgecolor='black')
        ax4.add_patch(circle)
        circle = plt.Circle((metric, y_coord), radius, facecolor=(0, 0.5 - t, 0.5 - y), linewidth=0.05,
                            edgecolor='black')
        ax5.add_patch(circle)
        circle = plt.Circle((metric, y_coord), radius, facecolor=(0.5 - x, 0, 0.5 - y), linewidth=0.05,
                            edgecolor='black')
        ax6.add_patch(circle)
    ax1.set_title("P and X")
    ax2.set_title("P and T")
    ax3.set_title("P and Y")
    ax4.set_title("X and T")
    ax5.set_title("T and Y")
    ax6.set_title("X and Y")
    for axs in ax:
        axs.autoscale_view()
        axs.set_xlabel(metric_name if not logscale else f"log({metric_name})")
        axs.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    rosenbaum = RosenbaumSensitivityModel("Rosenbaum Sensitivity Model")
    rosenbaum_metrics = {}
    msm = MarginalSensitivityModel("Marginal Sensitivity Model")
    msm_metrics = {}
    fmsm = FMarginalSensitivityModel("MSM with f-divergence")
    fmsm_metrics = {}
    fdmsm = FdMarginalSensitivityModel("MSM with f-divergence in derivative")
    fdmsm_metrics = {}
    fm = FSensitivityModel("f-sensitivity Model", lambda t: t * np.log(t))
    fm_metrics = {}
    ps = np.linspace(0.05, 0.95, 5)
    xs = np.linspace(-0.5, 0.5, 5)
    ts = np.linspace(-0.5, 0.5, 5)
    ys = np.linspace(-0.5, 0.5, 5)
    bar = tqdm(range(len(ps) * len(xs) * len(ts) * len(ys)))
    for p in ps:
        for x_effect in xs:
            for t_effect in ts:
                for y_effect in ys:
                    setting = (p, x_effect, t_effect, y_effect)
                    df, _ = create_generator(*setting)
                    rosenbaum_metric = rosenbaum.sensitivity_measure(df)
                    rosenbaum_metrics[setting] = rosenbaum_metric
                    msm_metric = msm.sensitivity_measure(df)
                    msm_metrics[setting] = msm_metric
                    fm_metric = fm.sensitivity_measure(df)
                    fm_metrics[setting] = fm_metric
                    fmsm_metric = fmsm.sensitivity_measure(df)
                    fmsm_metrics[setting] = fmsm_metric
                    fdmsm_metric = fdmsm.sensitivity_measure(df)
                    fdmsm_metrics[setting] = fdmsm_metric
                    bar.update()
    fig, ax = plt.subplots()
    r = 0.05
    # p to X
    for p in ps:
        for x in xs:
            circle = plt.Circle((p, x), r, facecolor=(c(p, x), p, p), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on X")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[X|U=1] - E[X|U=0]")
    plt.show()
    # p to T
    fig, ax = plt.subplots()
    for p in ps:
        for t in ts:
            circle = plt.Circle((p, t), r, facecolor=(p, c(p, t), p), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on T")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    plt.show()
    # p to Y
    fig, ax = plt.subplots()
    for p in ps:
        for y in ys:
            circle = plt.Circle((p, y), r, facecolor=(p, p, c(p, y)), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on P(U=1) and effect of U on Y")
    ax.set_xlabel("P(U=1)")
    ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    plt.show()
    # X to T
    fig, ax = plt.subplots()
    for x in xs:
        for t in ts:
            circle = plt.Circle((x, t), r, facecolor=(0.5 - x, 0.5 - t, 0), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on effect of U on X and T")
    ax.set_xlabel("E[X|U=1] - E[X|U=0]")
    ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    plt.show()
    # T to Y
    fig, ax = plt.subplots()
    for t in ts:
        for y in ys:
            circle = plt.Circle((t, y), r, facecolor=(0, 0.5 - t, 0.5 - y), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on effect of U on T and Y")
    ax.set_xlabel("E[T|U=1, X] - E[T|U=0, X]")
    ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    plt.show()
    # X to Y
    fig, ax = plt.subplots()
    for x in xs:
        for y in ys:
            circle = plt.Circle((x, y), r, facecolor=(0.5 - x, 0, 0.5 - y), linewidth=1, edgecolor='black')
            ax.add_patch(circle)
    ax.autoscale_view()
    ax.set_title("Simple settings colored based on effect of U on X and Y")
    ax.set_xlabel("E[X|U=1] - E[X|U=0]")
    ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    plt.show()

    # Plot rosenbaum sensitivity distribution
    # plot_setting_distribution(rosenbaum_metrics, "Γ")
    plot_setting_distribution(rosenbaum_metrics, "Γ", logscale=True)
    # plot_setting_distribution(msm_metrics, "Λ")
    plot_setting_distribution(msm_metrics, "Λ", logscale=True)
    # plot_setting_distribution(fm_metrics, "ρ")
    plot_setting_distribution(fm_metrics, "ρ", logscale=True)
    # plot_setting_distribution(fmsm_metrics, "Γ")
    plot_setting_distribution(fmsm_metrics, "Γ", logscale=True)
    # plot_setting_distribution(fdmsm_metrics, "Γ")
    plot_setting_distribution(fdmsm_metrics, "Γ", logscale=True)

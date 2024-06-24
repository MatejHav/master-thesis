import os.path

import numpy as np
from causalml.inference.tree import CausalRandomForestRegressor
import matplotlib.pyplot as plt
# CausalML changes the style of PLT
plt.style.use("default")

from data.generators import *
from sensitivities import *
from models.SensitivityModels import bounds_creator, closed_form_f_sensitivity, gaussian_mixture_model, get_author_algorithm_bounds
from sklearn.mixture import GaussianMixture

c = lambda p, v: 1 - (0.5 * (v + 0.5) + 0.5 * p)


def create_generator(u_prob, x_effect, t_effect, y_effect, dim=1, base_x_prob=0.45,  base_t_prob=0.5):
    path = f"./csv_files/data_u{int(100 * u_prob)}_x{int(100 * x_effect)}_t{int(100 * t_effect)}_y{int(100 * y_effect)}_k{dim}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, path
    # General settings
    n_rows = 3_000_000
    n_jobs = 30
    sizes = {
        "U": 1,
        "X": dim,
        "T": 1,
        "Y": 1
    }

    # Generators
    u_gen = lambda noise: [0 if np.random.rand() >= u_prob else 1]
    x_gen = lambda u, noise: [0 if np.random.rand() >= x_effect * u[0] + base_x_prob else 1 for _ in range(sizes["X"])]
    t_gen = lambda u, x, noise: [0 if np.random.rand() >= t_effect * u[0] + 0.25 * sum(x)/dim + base_t_prob else 1]
    y_gen = lambda u, x, t, noise: [round(sum(x)/dim + y_effect * u[0] + 2 * t[0] + noise, 1)]
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
        "Y": lambda: 0.1 * np.random.randn() - 1
    }

    generator = RegularGenerator(generators=generators, noise_generators=noise, sizes=sizes)
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


def main2():
    # Generate dataset
    # df, path = create_generator(0.25, 0, -0.3, 1)
    df, path = create_generator(0.25, 0, -0.3, 1, dim=3)
    # Generate counterfactuals and save
    # path = './csv_files/synthetic_data_random_forest_dim3.csv'
    # if not os.path.exists(path):
    #     # Train Random Forest
    #     X = df['X0'].to_numpy().reshape(-1, 1)
    #     T = df['T0'].to_numpy()
    #     Y = df['Y0'].to_numpy()
    #     random_forest = CausalRandomForestRegressor()
    #     random_forest.fit(X=X, treatment=T, y=Y)
    #     copy_of_data = pd.DataFrame([], columns=df.columns)
    #     predictions = random_forest.predict(X, with_outcomes=True)
    #     for index, row in df.iterrows():
    #         u = row['U0']
    #         x = row['X0']
    #         t = row['T0']
    #         y = row['Y0']
    #         y0 = predictions[index][0] if t == 1 else y
    #         y1 = predictions[index][1] if t == 0 else y
    #         copy_of_data.loc[len(copy_of_data)] = [u, x, 0, round(y0, 1)]
    #         copy_of_data.loc[len(copy_of_data)] = [u, x, 1, round(y1, 1)]
    #     copy_of_data.to_csv(path)
    # # data = pd.read_csv('./csv_files/synthetic_data_random_forest.csv')
    data = pd.read_csv(path)
    # # Train Gaussian Mixture models
    # k = 1
    # treated_data0 = data[(data['X0'] == 0) & (data['T0'] == 1)]
    # treated_gauss0 = GaussianMixture(n_components=k, covariance_type='spherical')
    # treated_gauss0.fit(treated_data0['Y0'].to_numpy().reshape(-1, 1))
    # treated_data1 = data[(data['X0'] == 1) & (data['T0'] == 1)]
    # treated_gauss1 = GaussianMixture(n_components=k, covariance_type='spherical')
    # treated_gauss1.fit(treated_data1['Y0'].to_numpy().reshape(-1, 1))
    # control_data0 = data[(data['X0'] == 0) & (data['T0'] == 0)]
    # control_gauss0 = GaussianMixture(n_components=k, covariance_type='spherical')
    # control_gauss0.fit(control_data0['Y0'].to_numpy().reshape(-1, 1))
    # control_data1 = data[(data['X0'] == 1) & data['T0'] == 0]
    # control_gauss1 = GaussianMixture(n_components=k, covariance_type='spherical')
    # control_gauss1.fit(control_data1['Y0'].to_numpy().reshape(-1, 1))
    # means = {
    #     'treated': [treated_gauss0.means_[0], treated_gauss1.means_[0]],
    #     'control': [control_gauss0.means_[0], control_gauss1.means_[0]]
    # }
    # variances = {
    #     'treated': [treated_gauss0.covariances_, treated_gauss1.covariances_],
    #     'control': [control_gauss0.covariances_, control_gauss1.covariances_]
    # }
    # Generate data based on the gaussian
    # path = './csv_files/synthetic_data_gaussian.csv'
    # if not os.path.exists(path):
    #     copy_of_data = pd.DataFrame([], columns=df.columns)
    #     for index, row in df.iterrows():
    #         u = row['U0']
    #         x = int(row['X0'])
    #         t = row['T0']
    #         y = row['Y0']
    #         y0 = np.sum(variances["control"][x] * np.random.randn(k) + means["control"][x]) if t == 1 else y
    #         y1 = np.sum(variances["treated"][x] * np.random.randn(k) + means["treated"][x]) if t == 0 else y
    #         copy_of_data.loc[len(copy_of_data)] = [u, x, 0, round(y0, 3)]
    #         copy_of_data.loc[len(copy_of_data)] = [u, x, 1, round(y1, 3)]
    #     copy_of_data.to_csv(path)
    # data = pd.read_csv(path)
    # Compute bounds
    rhos = np.linspace(0, 1.3, 13)
    f_upper = []
    f_lower = []
    closed_f_upper = []
    closed_f_lower = []
    author_lower = []
    author_upper = []
    author_lower95 = []
    author_lower05 = []
    author_upper95 = []
    author_upper05 = []
    gaussian_lower = []
    gaussian_upper = []
    # num_runs = 10
    # for rho in tqdm(rhos):
    #     lower_temp = []
    #     upper_temp = []
    #     for run in range(num_runs):
    #         lower, upper = get_author_algorithm_bounds(data, rho)
    #         upper_temp.append(upper)
    #         lower_temp.append(lower)
    #     author_upper.append(np.mean(upper_temp))
    #     author_lower.append(np.mean(lower_temp))
    #     author_lower05.append(np.quantile(lower_temp, 0.05))
    #     author_upper05.append(np.quantile(upper_temp, 0.05))
    #     author_lower95.append(np.quantile(lower_temp, 0.95))
    #     author_upper95.append(np.quantile(upper_temp, 0.95))
    #
    # plt.plot(rhos, author_lower, color='green')
    # plt.plot(rhos, author_upper, color='green')
    # plt.fill_between(rhos, author_lower05, author_lower95, color='green', alpha=0.5)
    # plt.fill_between(rhos, author_upper05, author_upper95, color='green', alpha=0.5)
    # plt.xlabel('ρ')
    # plt.ylabel('ATE')
    # plt.title('Bound from authors approach with a Random Forest model')
    # plt.show()

    for rho in tqdm(rhos):
        y_control, y_treated, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(data,
                                                                                                          sensitivity_model='f',
                                                                                                          sensitivity_measure=rho)
        f_upper.append(round(upper_treated - lower_control, 1))
        f_lower.append(round(lower_treated - upper_control, 1))
        # Closed f sensitivity
        lower = closed_form_f_sensitivity(data, rho, True)
        upper = closed_form_f_sensitivity(data, rho, False)
        closed_f_upper.append(round(upper, 2))
        closed_f_lower.append(round(lower, 2))
        # # Closed form Gaussian Mixture
        # lower = gaussian_mixture_model(data, rho, True, means, variances, k)
        # upper = gaussian_mixture_model(data, rho, False, means, variances, k)
        # gaussian_lower.append(round(lower, 2))
        # gaussian_upper.append(round(upper, 2))
    # # Show two graphs with the graphs
    plt.plot(rhos, f_upper, color='red')
    plt.plot(rhos, f_lower, color='red')
    plt.xlabel('ρ')
    plt.ylabel('ATE')
    plt.title('Bound from constraint programing approach')
    plt.show()
    #
    plt.plot(rhos, closed_f_upper, color='blue')
    plt.plot(rhos, closed_f_lower, color='blue')
    plt.xlabel('ρ')
    plt.ylabel('ATE')
    plt.title('Bound from EVaR approach')
    # plt.title('Bound from gradient descent approach with a Gaussian Mixture model')
    plt.show()
    # #
    # plt.plot(rhos, author_lower, color='green')
    # plt.plot(rhos, author_upper, color='green')
    # plt.xlabel('ρ')
    # plt.ylabel('ATE')
    # plt.title('Bound from authors approach with a Random Forest model')
    # plt.show()
    #
    # plt.plot(rhos, gaussian_upper, color='red', label="Closed Form")
    # plt.plot(rhos, gaussian_lower, color='red')
    # plt.plot(rhos, closed_f_upper, color='blue', label="EVaR approach")
    # plt.plot(rhos, closed_f_lower, color='blue')
    # plt.xlabel('ρ')
    # plt.ylabel('ATE')
    # plt.legend()
    # plt.title('Bound from closed form and EVaR approaches using random forest')
    # plt.show()
    # # # Print out results into a table
    # # # print(f"""
    # # # \\(\\rho\\)&  0 & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 1 & 1.1 & 1.2\\\\
    # # #          \hline
    # # #          CF Upper & {'&'.join(list(map(str,gaussian_upper)))} \\\\
    # # #          \hline
    # # #          CF Lower & {'&'.join(list(map(str,gaussian_lower)))} \\\\
    # # #          \hline
    # # #          GD Upper & {'&'.join(list(map(str,closed_f_upper)))}\\\\
    # # #          \hline
    # # #          GD Lower & {'&'.join(list(map(str,closed_f_lower)))}
    # # # """)


def main3():
    rhos = np.linspace(0, 1, 3)
    ps = [0.05, 0.15, 0.95]
    xs = [0.05, 0.15, 0.95]
    ts = [-0.25, 0.05, 0.15]
    ys = [-0.5, -0.25, 0.05, 0.15, 0.95]
    bar = tqdm(range(len(ps) * len(xs) * len(ts) * len(ys)))
    res = np.zeros((4, 5))
    for i_p, p in enumerate(ps):
        for i_x, x_effect in enumerate(xs):
            for i_t, t_effect in enumerate(ts):
                for i_y, y_effect in enumerate(ys):
                    data, _ = create_generator(p, x_effect, t_effect, y_effect, dim=1, base_x_prob=0, base_t_prob=0.5)
                    differences = []
                    for rho in rhos:
                        _, _, lower_control, lower_treated, upper_control, upper_treated = bounds_creator(
                            data,
                            sensitivity_model='f',
                            sensitivity_measure=rho)
                        cp_upper = upper_treated - lower_control
                        cp_lower = lower_treated - upper_control
                        # Gradient Descent f sensitivity
                        gd_lower = closed_form_f_sensitivity(data, rho, True)
                        gd_upper = closed_form_f_sensitivity(data, rho, False)
                        differences.append(abs(cp_upper - gd_upper))
                        differences.append(abs(cp_lower - gd_lower))
                    max_diff = max(differences)
                    # Probability of U
                    res[0, i_p+2] = round(max(max_diff, res[0, i_p+2]), 3)
                    # Probability of X
                    res[1, i_x+2] = round(max(max_diff, res[1, i_x+2]), 3)
                    # Probability of T
                    res[2, i_t+1] = round(max(max_diff, res[2, i_t+1]), 3)
                    # Outcome
                    res[3, i_y] = round(max(max_diff, res[3, i_y]), 3)
                    bar.update()
    print(f"""
    & -0.5 & -0.25 & 0.01 & 0.5 & 0.99 \\
     u_p & - & - & {res[0, 2]} & {res[0, 3]} & {res[0, 4]}\\
     x_u & - & - & {res[1, 2]} & {res[1, 3]} & {res[1, 4]}\\
     t_u & - & {res[2, 1]} & {res[2, 2]} & {res[2, 3]} & -\\
     y_u & {res[3, 0]} & {res[3, 1]} & {res[3, 2]} & {res[3, 3]} & {res[3, 4]}\\
    """)


if __name__ == '__main__':
    main2()
    # main3()
    # rosenbaum = RosenbaumSensitivityModel("Rosenbaum Sensitivity Model")
    # rosenbaum_metrics = {}
    # msm = MarginalSensitivityModel("Marginal Sensitivity Model")
    # msm_metrics = {}
    # fmsm = FMarginalSensitivityModel("MSM with f-divergence")
    # fmsm_metrics = {}
    # fdmsm = FdMarginalSensitivityModel("MSM with f-divergence in derivative")
    # fdmsm_metrics = {}
    # fm = FSensitivityModel("f-sensitivity Model", lambda t: t * np.log(t))
    # fm_metrics = {}
    # ps = np.linspace(0.05, 0.95, 5)
    # xs = np.linspace(-0.5, 0.5, 5)
    # ts = np.linspace(-0.5, 0.5, 5)
    # ys = np.linspace(-0.5, 0.5, 5)
    # bar = tqdm(range(len(ps) * len(xs) * len(ts) * len(ys)))
    # for p in ps:
    #     for x_effect in xs:
    #         for t_effect in ts:
    #             for y_effect in ys:
    #                 setting = (p, x_effect, t_effect, y_effect)
    #                 df, _ = create_generator(*setting)
    #                 rosenbaum_metric = rosenbaum.sensitivity_measure(df)
    #                 rosenbaum_metrics[setting] = rosenbaum_metric
    #                 msm_metric = msm.sensitivity_measure(df)
    #                 msm_metrics[setting] = msm_metric
    #                 fm_metric = fm.sensitivity_measure(df)
    #                 fm_metrics[setting] = fm_metric
    #                 fmsm_metric = fmsm.sensitivity_measure(df)
    #                 fmsm_metrics[setting] = fmsm_metric
    #                 fdmsm_metric = fdmsm.sensitivity_measure(df)
    #                 fdmsm_metrics[setting] = fdmsm_metric
    #                 bar.update()
    # fig, ax = plt.subplots()
    # r = 0.05
    # # p to X
    # for p in ps:
    #     for x in xs:
    #         circle = plt.Circle((p, x), r, facecolor=(c(p, x), p, p), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on P(U=1) and effect of U on X")
    # ax.set_xlabel("P(U=1)")
    # ax.set_ylabel("E[X|U=1] - E[X|U=0]")
    # plt.show()
    # # p to T
    # fig, ax = plt.subplots()
    # for p in ps:
    #     for t in ts:
    #         circle = plt.Circle((p, t), r, facecolor=(p, c(p, t), p), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on P(U=1) and effect of U on T")
    # ax.set_xlabel("P(U=1)")
    # ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    # plt.show()
    # # p to Y
    # fig, ax = plt.subplots()
    # for p in ps:
    #     for y in ys:
    #         circle = plt.Circle((p, y), r, facecolor=(p, p, c(p, y)), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on P(U=1) and effect of U on Y")
    # ax.set_xlabel("P(U=1)")
    # ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    # plt.show()
    # # X to T
    # fig, ax = plt.subplots()
    # for x in xs:
    #     for t in ts:
    #         circle = plt.Circle((x, t), r, facecolor=(0.5 - x, 0.5 - t, 0), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on effect of U on X and T")
    # ax.set_xlabel("E[X|U=1] - E[X|U=0]")
    # ax.set_ylabel("E[T|U=1, X] - E[T|U=0, X]")
    # plt.show()
    # # T to Y
    # fig, ax = plt.subplots()
    # for t in ts:
    #     for y in ys:
    #         circle = plt.Circle((t, y), r, facecolor=(0, 0.5 - t, 0.5 - y), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on effect of U on T and Y")
    # ax.set_xlabel("E[T|U=1, X] - E[T|U=0, X]")
    # ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    # plt.show()
    # # X to Y
    # fig, ax = plt.subplots()
    # for x in xs:
    #     for y in ys:
    #         circle = plt.Circle((x, y), r, facecolor=(0.5 - x, 0, 0.5 - y), linewidth=1, edgecolor='black')
    #         ax.add_patch(circle)
    # ax.autoscale_view()
    # ax.set_title("Simple settings colored based on effect of U on X and Y")
    # ax.set_xlabel("E[X|U=1] - E[X|U=0]")
    # ax.set_ylabel("E[Y|U=1, X, T] - E[Y|U=0, X, T]")
    # plt.show()
    #
    # # Plot rosenbaum sensitivity distribution
    # # plot_setting_distribution(rosenbaum_metrics, "Γ")
    # plot_setting_distribution(rosenbaum_metrics, "Γ", logscale=True)
    # # plot_setting_distribution(msm_metrics, "Λ")
    # plot_setting_distribution(msm_metrics, "Λ", logscale=True)
    # # plot_setting_distribution(fm_metrics, "ρ")
    # plot_setting_distribution(fm_metrics, "ρ", logscale=True)
    # # plot_setting_distribution(fmsm_metrics, "Γ")
    # plot_setting_distribution(fmsm_metrics, "Γ", logscale=True)
    # # plot_setting_distribution(fdmsm_metrics, "Γ")
    # plot_setting_distribution(fdmsm_metrics, "Γ", logscale=True)

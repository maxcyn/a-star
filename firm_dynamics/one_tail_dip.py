from firm_dynamics.hill import hill_hazard, model_survival_curve_hill
import numpy as np
from scipy.integrate import trapezoid, quad
from firm_dynamics.survival_analysis import obtain_survival_fractions, obtain_total_alive_count
from scipy.optimize import minimize


def epsilon(s, a, eps0, tau, lam, t_e):
    '''
    Time-dependent perturbation function
    Inputs:
    - s: integration variable (time)
    - a: age
    - eps0: amplitude of the perturbation
    - tau: decay rate (against age of firm) of the effect of external event
    - lam: recovery rate (against time since event)
    - t_e: time of event causing the perturbation
    '''
    def g(a, eps0, tau, t_e):
        return eps0 * np.exp(-abs(a - t_e) * tau)
    if s < a - t_e:
        return 0
    else:
        return g(a, eps0, tau, t_e)*np.exp(-lam*(t_e-(a-s)))


def hazard_with_perturbation(s, a, mu_ub, mu_lb, K, m,
                             eps0, tau, lam, t_e):
    """
    μ_pert(a,s) = (1 + ε(a,s)) μ_Hill(s)
    """
    base = hill_hazard(s, mu_ub, mu_lb, K, m)
    eps = epsilon(s, a, eps0, tau, lam, t_e)
    return (1.0 + eps) * base


# 5. Survival function f(a) with perturbation
def survival_hill_with_perturbation(ages,
                                    mu_ub, mu_lb, K, m,
                                    eps0, tau, lam, t_e):
    """
    Compute survival f(a) for an array of ages, under Hill hazard with
    a time-dependent perturbation.

    ages: 1D array of cohort ages (years)
    use_quad: if True, use scipy.integrate.quad per age
              if False, use simple trapezoidal rule on fixed grid.
    """
    ages = np.asarray(ages, dtype=float)
    f_vals = np.empty_like(ages)

    # For each age, integrate μ_pert(a,s) from s=0 to s=a
    for i, a in enumerate(ages):
        if a <= 0:
            f_vals[i] = 1.0
            continue

        def integrand(s):
            return hazard_with_perturbation(
                s, a, mu_ub, mu_lb, K, m,
                eps0, tau, lam, t_e
            )

        integral, _ = quad(integrand, 0.0, a)
        f_vals[i] = np.exp(-integral)

    return f_vals


def find_dip(df_analysis, sector):
    '''
    Find the dip location for a given sector by finding points of minimum log-likelihood
    '''
    sector_list = ['G', 'M', 'F', 'J', 'K', 'C', 'H', 'S', 'N', 'I', 'P', 'L', 'Q', 'R']
    parameters = [
        [0.13772780796801345, 0.06040065635354765, 9.678611031814139, 38.28653412450151],
        [0.10853412588782649, 0.060405453614713986, 10.324137068120953, 20.272105749873536],
        [0.08284945935354733, 0.07532165486572806, 7.246079688090822, 99.99992722770091],
        [0.13138896943819373, 0.06387240987874851, 9.026894200245238, 64.59003275843841],
        [0.07012013735407156, 0.011068176577001085, 17.600965883019793, 11.716328585887581],
        [0.09997560159043367, 0.049269924679968866, 8.699286997697827, 100.0],
        [0.19014381450690954, 0.028015942761638372, 6.937678823494837, 100.0],
        [0.14057226596633066, 1e-10, 12.954249620403116, 5.199152676121435],
        [0.12396183075343398, 1e-10, 16.432802053332697, 3.6763982546287552],
        [0.12639621277376695, 1.0000001467199392e-10, 20.851484353072767, 3.8694421065642923],
        [0.12121307544489934, 0.06868412785046317, 9.445279125853938, 100.0],
        [0.07256783090508868, 0.0725678283579821, 14.249169553345428, 0.8209674189944658],
        [0.07830172410553711, 0.04719751513515326, 7.792008550188214, 100.0],
        [0.13228882837926284, 0.08548611352494333, 8.857641818720507, 100.0]
        ]
    sector_params_MLE = dict(zip(sector_list, parameters))

    _, ages = obtain_survival_fractions(df_analysis, 'Sector', sector)
    totals, survivors = obtain_total_alive_count(df_analysis, 'Sector', sector)

    mu_ub, mu_lb, K, m = sector_params_MLE[sector]
    S_vals = model_survival_curve_hill(ages, mu_ub, mu_lb, K, m)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)  # avoid log(0)

    deaths = totals - survivors
    logL = survivors * np.log(S_vals) + deaths * np.log(1 - S_vals)

    minlogL_ages = ages[np.argsort(logL)[:9]]
    max_count = 0
    best_cluster = []

    for i in range(len(minlogL_ages)):
        # Find all points within window of minlogL_ages[i]
        cluster = minlogL_ages[(minlogL_ages >= minlogL_ages[i]) & (minlogL_ages <= minlogL_ages[i] + 0.5)]
        if len(cluster) > max_count:
            max_count = len(cluster)
            best_cluster = cluster

    return float(np.mean(best_cluster)) if len(best_cluster) > 0 else None


def log_likelihood_perturbed(params, ages, survivors, totals):
    """
    Binomial log-likelihood for the perturbed Hill survival model.

    params: array-like of length 8:
        [mu_ub, mu_lb, K, m, eps0, tau, lam, t_e]
    ages:      1D array of ages (years)
    survivors: number of survivors at each age
    totals:    cohort size at each age
    """
    (mu_ub, mu_lb, K, m,
     eps0, tau, lam, t_e) = params

    # Basic parameter validity checks (adapt bounds as needed)
    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0:
        return -np.inf
    if eps0 < 0 or tau < 0 or lam < 0:
        return -np.inf

    # Survival probabilities at each age
    S_vals = survival_hill_with_perturbation(
        ages, mu_ub, mu_lb, K, m,
        eps0, tau, lam, t_e
    )

    # Numerical safety: keep inside (0,1)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)

    deaths = totals - survivors

    # Extra consistency check
    if (np.any(deaths < 0) or
        np.any(survivors < 0) or
        np.any(survivors > totals)):
        return -np.inf

    logL = np.sum(
        survivors * np.log(S_vals) +
        deaths * np.log(1.0 - S_vals)
    )
    return logL


def neg_log_likelihood_perturbed(params, ages, survivors, totals):
    """
    Wrapper returning -logL for use with scipy.optimize.minimize.
    """
    ll = log_likelihood_perturbed(params, ages, survivors, totals)
    if not np.isfinite(ll):
        return 1e10  # large penalty
    return -ll


def mle_sector_perturbed(sector, df_analysis,
                         initial_guess=[0.1, 0.05, 10, 5, 1, 1, 1, 7]):
    """
    MLE of perturbed Hill model for a single sector.

    sector: sector label
    df_analysis: your DataFrame with firm data
    ages_data: global age grid (1D array)
    initial_hill_params: optional initial guess for [mu_ub, mu_lb, K, m].
                         If None, you can plug in the Hill-only MLE.

    Returns:
        result (scipy OptimizeResult), plus the data used.
    """
    # 1) Get counts for this sector
    totals, survivors = obtain_total_alive_count(df_analysis, 'Sector', sector)

    valid_mask = totals > 0
    totals = totals[valid_mask]
    survivors = survivors[valid_mask]
    ages = obtain_survival_fractions(df_analysis, 'Sector', sector)[1][valid_mask]

    if len(survivors) == 0:
        raise ValueError(f"No valid data for sector {sector}")

    # 2) Bounds for parameters
    bounds = [
        (1e-6, 0.3),    # mu_ub
        (1e-6, 0.15),   # mu_lb
        (0.1, 50.0),    # K
        (0.1, 100.0),   # m
        (0.0, 5.0),     # eps0 (>=0; adjust upper bound as needed)
        (0.0, 5.0),     # tau
        (0.0, 5.0),     # lam
        (0.0, ages.max())  # t_e within age range
    ]

    # 3) Optimisation
    result = minimize(
        neg_log_likelihood_perturbed,
        initial_guess,
        args=(ages, survivors, totals),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )

    return result


# def lsq_hill_with_dip(params, ages, survival_fractions):
#     mu_ub, mu_lb, K, m, t_e, eps0, tau, lam = params
#     model = model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam)
#     model = np.clip(model, 1e-12, 1 - 1e-12)
#     return np.sum((survival_fractions - model) ** 2)


# def lsqfit_hill_with_dip(ages, survival_fractions, initial_guess=[0.1, 0.05, 10, 5, 7, 1, 1, 1]):
#     '''
#     Least squares fit for the Hill model with dip.
#     '''
#     bounds = [
#         (0.01, 0.3),   # mu_ub
#         (0.001, 0.15),   # mu_lb
#         (0.1, 30),     # K
#         (0.5, 50),    # m
#         (3, 10),      # t_e
#         (0, 20),    # eps0
#         (0.01, 10),      # tau
#         (10e-6, 10)      # lam
#     ]

#     constraints = [
#         {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # mu_ub - mu_lb > 0
#         {'type': 'ineq', 'fun': lambda x: x[6] - x[7]},  # tau - lam > 0
#     ]
#     minimize(lsq_hill_with_dip, initial_guess, args=(ages, survival_fractions), bounds=bounds, constraints=constraints)


def epsilon_simple_vec(s_grid, a, eps0, lam, t_e):
    s = np.asarray(s_grid, dtype=float)
    eps = np.zeros_like(s)
    mask = s >= (a - t_e)
    eps[mask] = eps0 * np.exp(-lam * (t_e - (a - s[mask])))
    return eps


def survival_hill_with_perturbation_fast(ages,
                                         mu_ub, mu_lb, K, m,
                                         eps0, lam, t_e):
    ages = np.asarray(ages, dtype=float)
    f_vals = np.empty_like(ages)
    for i, a in enumerate(ages):
        if a <= 0:
            f_vals[i] = 1.0
            continue

        s_grid = np.linspace(0.0, a, n_grid)
        integral = (quad(hill_hazard(s_grid, mu_ub, mu_lb, K, m) *
                              (1.0 + epsilon_simple_vec(s_grid, a, eps0, lam, t_e)), s_grid)
                    )
        f_vals[i] = np.exp(-integral)

    return f_vals


def log_likelihood_perturbed_fast(params, ages, survivors, totals,
                                  n_grid=300):
    mu_ub, mu_lb, K, m, eps0, lam, t_e = params

    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0:
        return -np.inf
    if eps0 < 0 or lam < 0 or t_e < 0 or t_e > ages.max():
        return -np.inf

    S_vals = survival_hill_with_perturbation_fast(
        ages, mu_ub, mu_lb, K, m, eps0, lam, t_e,
        n_grid=n_grid
    )
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)

    deaths = totals - survivors
    if (np.any(deaths < 0) or
        np.any(survivors < 0) or
        np.any(survivors > totals)):
        return -np.inf

    logL = np.sum(
        survivors * np.log(S_vals) +
        deaths    * np.log(1.0 - S_vals)
    )
    return logL


def neg_log_likelihood_perturbed_fast(params, ages, survivors, totals,
                                      n_grid=300):
    ll = log_likelihood_perturbed_fast(params, ages, survivors, totals,
                                       n_grid=n_grid)
    if not np.isfinite(ll):
        return 1e10
    return -ll


def mle_sector_perturbed_fast(sector, df_analysis,
                              initial_guess=[0.1, 0.05, 10, 5, 1, 1, 7],
                              n_grid=300):
    totals, survivors = obtain_total_alive_count(df_analysis, 'Sector', sector)
    valid_mask = totals > 0
    totals = totals[valid_mask]
    survivors = survivors[valid_mask]
    ages = obtain_survival_fractions(df_analysis, 'Sector', sector)[1][valid_mask]

    if len(survivors) == 0:
        raise ValueError(f"No valid data for sector {sector}")

    bounds = [
        (1e-6, 0.3),              # mu_ub
        (1e-6, 0.15),             # mu_lb
        (0.1, 50.0),              # K
        (0.1, 100.0),             # m
        (0.0, 10.0),               # eps0
        (0.0, 30.0),               # lam
        (0.0, 10)  # t_e
    ]

    result = minimize(
        neg_log_likelihood_perturbed_fast,
        initial_guess,
        args=(ages, survivors, totals, n_grid),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )

    return result

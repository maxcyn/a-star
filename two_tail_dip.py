from hill import *
import numpy as np
from scipy.integrate import quad
from survival_analysis import obtain_survival_fractions, obtain_total_alive_count
from scipy.optimize import minimize

def epsilon(s, a, eps0, tau1, tau2, lam, t_e):
    '''
    Time-dependent perturbation function
    Inputs:
    - s: integration variable (time)
    - a: age
    - eps0: amplitude of the perturbation
    - tau1: decay rate (against age of firm) of the effect of external event on firms born after t_e
    - tau2: decay rate (against age of firm) of the effect of external event on firms born before t_e
    - lam: recovery rate (against time since event)
    - t_e: time of event causing the perturbation
    '''
    def g(a, eps0, tau1, tau2, t_e):
        if a < t_e:
            return eps0 * np.exp(-abs(a - t_e) * tau1)
        else:
            return eps0 * np.exp(-abs(a - t_e) * tau2)
    if s < a-t_e:
        return 0
    else:
        return g(a, eps0, tau1, tau2, t_e)*np.exp(-lam*(t_e-(a-s)))

def hill_survival_with_dip(a, mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2, lam):
    val, _ = quad(lambda s: hill_hazard(s, mu_ub, mu_lb, K, m) * epsilon(s, a, eps0, tau1, tau2, lam, t_e), 0, a)
    return hill_survival_function(a, mu_ub, mu_lb, K, m) * np.exp(-val)

def model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2, lam):
    return np.array([hill_survival_with_dip(a, mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2, lam) for a in ages])

def find_dip(df_analysis, sector):
    '''
    Find the dip location for a given sector by finding points of minimum log-likelihood
    '''
    sector_list = ['G', 'M', 'F', 'J', 'K', 'C', 'H', 'S', 'N', 'I', 'P', 'L', 'Q', 'R']
    parameters = [
        [0.13660027, 0.03574423, 12.39113424, 4.14356328],
        [0.10877533, 0.0418096, 12.55269306, 4.41391222],
        [0.079990154, 1.00E-10, 26.18237719, 79.99986416],
        [0.13090805, 0.03791174, 11.9429807, 4.05657508],
        [0.070120134, 0.011071032, 17.60063205, 11.71975389],
        [0.10301031, 0.04128293, 9.26045477, 8.13925264],
        [0.190143914, 0.028016019, 6.93767599, 100],
        [0.14058029, 1.00E-10, 12.9535533, 5.1898739],
        [0.12396223, 1.00E-10, 16.4327672, 3.67640026],
        [0.12568692, 0.03447114, 17.44283135, 5.60609428],
        [0.121213526, 0.068684245, 9.44518567, 100],
        [0.074121126, 1.00E-10, 25.77531849, 79.99860496],
        [0.078301599, 0.047197935, 7.79197632, 100],
        [0.132289514, 0.085485775, 8.85732298, 100]
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

def neg_ll_hill_with_dip(params, ages, survivors, totals):
    mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2, lam = params
    ll = 0
    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0 or t_e < 0 or eps0 < 0 or tau1 <= 0 or tau2 <= 0 or lam <= 0:
        return np.inf

    S_vals = model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2, lam)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)  # avoid log(0)

    deaths = totals - survivors
    logL = np.sum(survivors * np.log(S_vals) + deaths * np.log(1 - S_vals))
    return -logL  # minimize negative log-likelihood

def fit_hill_with_dip(ages, survivors, totals, initial_guess=[0.1, 0.05, 6, 10, 5, 1, 1, 1, 1]):
    '''
    MLE fitting for Hill model with dip
    '''
    bounds = [
        (0.01, 0.3),   # mu_ub
        (0.001, 0.15),   # mu_lb
        (0.1, 30),     # K
        (0.5, 50),    # m
        (3, 10),      # t_e
        (0, 20),    # eps0
        (0.01, 10),      # tau1
        (0.01, 10),      # tau2
        (10e-6, 10)      # lam
    ]

    # Constraint: tau1 > lam and tau2 > lam
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # mu_ub - mu_lb > 0
        {'type': 'ineq', 'fun': lambda x: x[6] - x[8]},  # tau1 - lam > 0
        {'type': 'ineq', 'fun': lambda x: x[7] - x[8]},  # tau2 - lam > 0
    ]

    result = minimize(
        neg_ll_hill_with_dip,
        initial_guess,
        args=(ages, survivors, totals),
        bounds=bounds,
        constraints=constraints
    )

    return result

def lsq_hill_with_dip(params, ages, survival_fractions):
    mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2 = params
    model = model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau1, tau2)
    model = np.clip(model, 1e-12, 1 - 1e-12)
    return np.sum((survival_fractions - model) ** 2)
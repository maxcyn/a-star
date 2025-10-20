from firm_dynamics.hill import hill_survival_function, hill_hazard, model_survival_curve_hill
import numpy as np
from scipy.integrate import quad
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
        if a >= t_e:
            return eps0 * np.exp((a-t_e) * tau)
        else:
            return eps0
    if s < a - t_e:
        return 0
    else:
        return g(a, eps0, tau, t_e)*np.exp(-lam*(t_e-(a-s)))


def hill_survival_with_dip(a, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam):
    val, _ = quad(lambda s: hill_hazard(s, mu_ub, mu_lb, K, m) * epsilon(s, a, eps0, tau, lam, t_e), 0, a)
    return hill_survival_function(a, mu_ub, mu_lb, K, m) * np.exp(-val)


def model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam):
    return np.array([hill_survival_with_dip(a, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam) for a in ages])


def find_dip(df_analysis, sector):
    '''
    Find the dip location for a given sector by finding points of minimum log-likelihood
    '''
    sector_list = ['G', 'M', 'F', 'J', 'K', 'C', 'H', 'S', 'N', 'I', 'P', 'L', 'Q', 'R']
    parameters = [
        [0.13771959172635478, 0.06036883390826407, 9.683527817633134, 37.16554776246212], 
        [0.10853423852842373, 0.06040203094166163, 10.324655252053779, 20.2431022135619], 
        [0.08284949629045338, 0.07532170024242305, 7.246072550255881, 99.9995683704219], 
        [0.13138419047680286, 0.06388417019228498, 9.027011437441391, 63.20559648903392], 
        [0.07012004488933721, 0.011080713775644365, 17.599335895302804, 11.726782536983189], 
        [0.09997597066599069, 0.049270691945483475, 8.699132957275232, 100.0], 
        [0.19014425302275023, 0.02801565370451772, 6.937673733970917, 100.0], 
        [0.14057209547880267, 1.0000000076278874e-10, 12.954253262292541, 5.19908018295204], 
        [0.12396232834152839, 1e-10, 16.432702180468965, 3.6763918799521744], 
        [0.12639658719104133, 1.0000000249470075e-10, 20.851108767628933, 3.8694205399833757], 
        [0.1212132957507885, 0.06868421106599219, 9.445203344625348, 100.0], 
        [0.07305173006148806, 0.07305173006148806, 4.686889369376459, 49.86003240187913], 
        [0.07830160011697058, 0.047197607271208426, 7.792050676547918, 100.0], 
        [0.13228889671445893, 0.0854858990969846, 8.857658026221204, 100.0]
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
    mu_ub, mu_lb, K, m, t_e, eps0, tau, lam = params
    ll = 0
    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0 or t_e < 0 or eps0 < 0 or tau <= 0 or lam <= 0:
        return np.inf

    S_vals = model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)  # avoid log(0)

    deaths = totals - survivors
    logL = np.sum(survivors * np.log(S_vals) + deaths * np.log(1 - S_vals))
    return -logL  # minimize negative log-likelihood


def mlefit_hill_with_dip(ages, survivors, totals, initial_guess=[0.1, 0.05, 10, 5, 7, 1, 1, 1]):
    '''
    Fit the Hill model with dip using MLE.
    '''
    bounds = [
        (0.01, 0.3),   # mu_ub
        (0.001, 0.15),   # mu_lb
        (0.1, 30),     # K
        (0.5, 50),    # m
        (3, 10),      # t_e
        (0, 20),    # eps0
        (0.01, 10),      # tau
        (10e-6, 10)      # lam
    ]

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # mu_ub - mu_lb > 0
        {'type': 'ineq', 'fun': lambda x: x[6] - x[7]},  # tau - lam > 0
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
    mu_ub, mu_lb, K, m, t_e, eps0, tau, lam = params
    model = model_hill_with_dip(ages, mu_ub, mu_lb, K, m, t_e, eps0, tau, lam)
    model = np.clip(model, 1e-12, 1 - 1e-12)
    return np.sum((survival_fractions - model) ** 2)


def lsqfit_hill_with_dip(ages, survival_fractions, initial_guess=[0.1, 0.05, 10, 5, 7, 1, 1, 1]):
    '''
    Least squares fit for the Hill model with dip.
    '''
    bounds = [
        (0.01, 0.3),   # mu_ub
        (0.001, 0.15),   # mu_lb
        (0.1, 30),     # K
        (0.5, 50),    # m
        (3, 10),      # t_e
        (0, 20),    # eps0
        (0.01, 10),      # tau
        (10e-6, 10)      # lam
    ]

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # mu_ub - mu_lb > 0
        {'type': 'ineq', 'fun': lambda x: x[6] - x[7]},  # tau - lam > 0
    ]
    minimize(lsq_hill_with_dip, initial_guess, args=(ages, survival_fractions), bounds=bounds, constraints=constraints)
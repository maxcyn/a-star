from firm_dynamics.hill import hill_hazard, model_survival_curve_hill
import numpy as np
from scipy.integrate import cumulative_trapezoid
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


def hazard_with_perturbation(a, s, mu_ub, mu_lb, K, m,
                             eps0, tau, lam, t_e):
    """
    μ_pert(a,s) = (1 + ε(a,s)) μ_Hill(s)
    """
    base = hill_hazard(s, mu_ub, mu_lb, K, m)
    eps = epsilon(a, s, eps0, tau, lam, t_e)
    return (1.0 + eps) * base


python
import numpy as np
from scipy.integrate import quad

# 1. Baseline Hill hazard
def hill_hazard(s, mu_ub, mu_lb, K, m):
    """
    Hill-type hazard function μ_Hill(s).
    s: age (can be scalar or array)
    """
    s = np.asarray(s)
    # Avoid division by zero when s=0
    s_m = np.power(s, m)
    denom = s_m + np.power(K, m)
    frac = np.zeros_like(s, dtype=float)
    mask = denom > 0
    frac[mask] = s_m[mask] / denom[mask]
    
    mu = mu_ub - (mu_ub - mu_lb) * frac
    return mu


# 2. g(a): event-age amplitude function
def g_of_a(a, eps0, tau, t_e):
    """
    g(a) governing initial amplitude of perturbation at cohort age a.
    """
    a = np.asarray(a)
    g = np.empty_like(a, dtype=float)
    
    # Before event age
    before = a < t_e
    g[before] = eps0
    
    # After event age
    after = ~before
    g[after] = eps0 * np.exp(-tau * (a[after] - t_e))
    
    return g


# 3. epsilon(a, s): perturbation term
def epsilon(a, s, eps0, tau, lam, t_e):
    """
    ε(a,s): perturbation multiplier for hazard at integration age s
    for a cohort observed at age a.
    
    For s < a - t_e: event has not occurred yet for this cohort age → ε=0.
    For s ≥ a - t_e: ε(a,s) = g(a) * exp(-λ * (t_e - (a - s))).
    """
    # scalar a, s (we treat them as floats here)
    if s < a - t_e:
        return 0.0
    
    g_a = g_of_a(np.array([a]), eps0, tau, t_e)[0]
    # t_e - (a - s) = (t_e + s - a)
    return g_a * np.exp(-lam * (t_e - (a - s)))


# 4. Full perturbed hazard
def hazard_with_perturbation(a, s, mu_ub, mu_lb, K, m,
                             eps0, tau, lam, t_e):
    """
    μ_pert(a,s) = (1 + ε(a,s)) μ_Hill(s)
    """
    base = hill_hazard(s, mu_ub, mu_lb, K, m)
    eps = epsilon(a, s, eps0, tau, lam, t_e)
    return (1.0 + eps) * base


# 5. Survival function f(a) with perturbation
def survival_hill_with_perturbation(ages,
                                    mu_ub, mu_lb, K, m,
                                    eps0, tau, lam, t_e,
                                    use_quad=True, n_grid=200):
    """
    Compute survival f(a) for an array of ages, under Hill hazard with
    a time-dependent perturbation.

    ages: 1D array of cohort ages (years)
    use_quad: if True, use scipy.integrate.quad per age
              if False, use simple trapezoidal rule on fixed grid.
    """
    ages = np.asarray(ages, dtype=float)
    f_vals = np.empty_like(ages)

    if use_quad:
        # For each age, integrate μ_pert(a,s) from s=0 to s=a
        for i, a in enumerate(ages):
            if a <= 0:
                f_vals[i] = 1.0
                continue

            def integrand(s):
                return hazard_with_perturbation(
                    a, s, mu_ub, mu_lb, K, m,
                    eps0, tau, lam, t_e
                )

            integral, _ = quad(integrand, 0.0, a, limit=200)
            f_vals[i] = np.exp(-integral)
    else:
        # Fixed grid trapezoidal rule (faster, approximate)
        for i, a in enumerate(ages):
            if a <= 0:
                f_vals[i] = 1.0
                continue

            s_grid = np.linspace(0.0, a, n_grid)
            mu_vals = hazard_with_perturbation(
                a, s_grid, mu_ub, mu_lb, K, m,
                eps0, tau, lam, t_e
            )
            integral = np.trapz(mu_vals, s_grid)
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
        (10e-6, 0.15),   # mu_lb
        (0.1, 30),     # K
        (0.5, 50),    # m
        (3, 10),      # t_e
        (0, 20),    # eps0
        (0.01, 15),      # tau
        (10e-6, 15)      # lam
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
def hill_hazard(a, mu_ub, mu_lb, K, m):
    '''
    Hill function
    Inputs:
    - a: age
    - mu_ub: upper bound of the hazard rate
    - mu_lb: lower bound of the hazard rate
    - K: mid-point
    - m: Hill coefficient
    '''
    return mu_ub - (mu_ub - mu_lb) * (a**m) / (a**m + K**m + 1e-10)

def hill_survival_function(a, mu_ub, mu_lb, K, m):
    '''
    e^(-integrate u)
    '''
    import numpy as np
    from scipy.integrate import quad

    result, _ = quad(lambda s: hill_hazard(s, mu_ub, mu_lb, K, m), 0, a)
    return np.exp(-result)

def model_survival_curve_hill(ages, mu_ub, mu_lb, K, m):
    '''
    Model the survival curve using the Hill function.
    '''
    import numpy as np

    return np.array([hill_survival_function(a, mu_ub, mu_lb, K, m) for a in ages])

def neg_log_likelihood_hill(params, ages, survivors, totals):
    '''
    Negative log-likelihood function for the Hill model.
    '''
    import numpy as np
    
    mu_ub, mu_lb, K, m = params
    if mu_lb < 0 or mu_ub < mu_lb or K <= 0 or m <= 0:
        return np.inf

    S_vals = model_survival_curve_hill(ages, mu_ub, mu_lb, K, m)
    S_vals = np.clip(S_vals, 1e-12, 1 - 1e-12)  # avoid log(0)

    deaths = totals - survivors
    logL = np.sum(survivors * np.log(S_vals) + deaths * np.log(1 - S_vals))
    return -logL  # minimize negative log-likelihood
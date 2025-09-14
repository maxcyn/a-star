def find_dip(sector):
    from survival_analysis import obtain_survival_fractions, obtain_total_alive_count

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
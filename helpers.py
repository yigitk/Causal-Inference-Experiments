from random import random


def print_run_ratio(bad_dict, good_dict):
    good_runs = 0
    bad_runs = 0
    for index in bad_dict:
        if bad_dict[index] == good_dict[index]:
            good_runs += 1
        else:
            bad_runs += 1
    print("Bad Runs:", bad_runs)
    print("Good Runs:", good_runs)


def get_run_ratio(bad_dict, good_dict):
    good_runs = 0
    bad_runs = 0
    for index in bad_dict:
        if bad_dict[index] == good_dict[index]:
            good_runs += 1
        else:
            bad_runs += 1
    return bad_runs, good_runs


def fuzzy(good_expression, gen_bad):
    if gen_bad:
        return good_expression*2*random()
    else:
        return good_expression


version_bug_dict = {
    "clausenLongFL": "r_0",
    "clausenLongFL.1": "d_1",
    "deriv_centralFL": "dy_0",
    "deriv_centralFL.1": "error_1",
    "frexpFL": "ex_0",
    "frexpFL.1": "f_0",
    "gsl_atanhFL": "to_return_0",
    "gsl_atanhFL.1": "to_return_2",
    "J0LongFL": "seps_0",
    "J0LongFL.1": "y_0",
    "J0LongFL.2": "ampl_0",
    "J0LongFL.3": "d_4",
    "J0LongFL.4": "ceps_0",
    "lngammaLongFL": "Ag_1",
    "lngammaLongFL.1": "x_9",
    "lngammaLongFL.2": "result_val_41",
    "lngammaLongFL.3": "Ag_0",
    "lngammaLongFL.4": "term2_0",
    "matric_transposeLongFL": "m_data_e2_1",
    "matric_transposeLongFL.1": "tmp_0",
    "skewnessFL": "mean_1",
    "skewnessFL.1": "variance_1",
    "synchrotron_1LongFL": "t_1",
    "synchrotron_1LongFL.1": "px_0",
    "synchrotron_1LongFL.2": "d_4",
    "synchrotron_1LongFL.3": "x_4",
    "synchrotron_1LongFL.4": "d_1",
    "transport_2LongFL": "result_val_5",
    "transport_2LongFL.1": "t_2"
}
version_bug_dict_multi = {
    "J0LongFL2Bug": ["d_1", "seps_0"],
    "J0LongFL3Bug": ["d_1", "seps_0", "d_4"],
    "lngammaLongFL2Bug": ["x_9", "result_val_41"],
    "lngammaLongFL3Bug": ["x_9", "result_val_41", "result_err_61"],
    "matric_transposeLongFL2Bug": ["m_data_e1_1", "m_data_e2_1"],
    "matric_transposeLongFL3Bug": ["m_data_e1_1", "m_data_e2_1", "tmp_0"],
    "synchrotron_1LongFL2Bug": ["x_4", "t_1"],
    "synchrotron_1LongFL3Bug": ["x_4", "t_1", "y_0"]
}
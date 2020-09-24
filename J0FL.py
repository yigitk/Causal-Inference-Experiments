import numpy as np
import pandas as pd
import numbers
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from phi import *
import random

import os

from J0 import good_dict
os.system('python J0.py')

dd_0 = None;
dd_2 = None
dd_1 = None;
dd_3 = None;
reason_0 = None;
stat_ca_0 = None;
stat_ca_1 = None;
GSL_NAN_0 = None;
ampl_0 = None;
ampl_1 = None;
abs_sum_0 = None;
fn_0 = None;
fn_1 = None;
ceps_0 = None;
ceps_1 = None;
ceps_2 = None;
result_0 = None;
result_1 = None;
result_2 = None;
result_3 = None;
result_4 = None;
sqrty_0 = None;
sqrty_1 = None;
stat_cp_0 = None;
stat_cp_1 = None;
gsl_errno_0 = None;
stat_ct_0 = None;
stat_ct_1 = None;
y2_0 = None;
ca_0 = None;
ca_1 = None;
order_0 = None;
val_0 = None;
seps_0 = None;
seps_1 = None;
seps_2 = None;
a_0 = None;
a_1 = None;
a_2 = None;
a_3 = None;
a_4 = None;
b_0 = None;
b_1 = None;
b_2 = None;
b_3 = None;
b_4 = None;
temp_1 = None;
temp_0 = None;
temp_2 = None;
c_0 = None;
c_1 = None;
c_2 = None;
c_3 = None;
err_0 = None;
d_0 = None;
d_2 = None;
d_1 = None;
d_3 = None;
d_4 = None;
d_5 = None;
d_6 = None;
e_0 = None;
e_2 = None;
e_1 = None;
e_3 = None;
e_4 = None;
sy_0 = None;
eps_0 = None;
j_0 = None;
j_2 = None;
j_1 = None;
j_3 = None;
e2_0 = None;
e2_1 = None;
cp_0 = None;
cp_1 = None;
cs_0 = None;
ct_0 = None;
ct_1 = None;
s_0 = None;
cy_0 = None;
x_0 = None;
x_1 = None;
x_2 = None;
y_0 = None;
y_1 = None;
y_2 = None;
z_0 = None;
z_1 = None;
order_sp_0 = None;
status_0 = None

from collections import namedtuple
from math import fabs, sin, cos, sqrt

GSL_SUCCESS = 0
GSL_FAILURE = -1
GSL_CONTINUE = -2
GSL_EDOM = 1
GSL_ERANGE = 2
GSL_EFAULT = 3
GSL_EINVAL = 4
GSL_EFAILED = 5
GSL_EFACTOR = 6
GSL_ESANITY = 7
GSL_ENOMEM = 8
GSL_EBADFUNC = 9
GSL_ERUNAWAY = 1
GSL_EMAXITER = 11
GSL_EZERODIV = 12
GSL_EBADTOL = 13
GSL_ETOL = 14
GSL_EUNDRFLW = 15
GSL_EOVRFLW = 16
GSL_ELOSS = 17
GSL_EROUND = 18
GSL_EBADLEN = 19
GSL_ENOTSQR = 20
GSL_ESING = 21
GSL_EDIVERGE = 22
GSL_EUNSUP = 23
GSL_EUNIMPL = 24
GSL_ECACHE = 25
GSL_ETABLE = 26
GSL_ENOPROG = 27
GSL_ENOPROGJ = 28
GSL_ETOLF = 29
GSL_ETOLX = 30
GSL_ETOLG = 31
GSL_EOF = 32
GSL_DBL_EPSILON = 2.2204460492503131e-16

class gsl_sf_result:
    def __init__(self, val, err):
        val_0 = val;
        err_0 = err;

        self.val = val_0
        self.err = err_0


bj0_data = [0.100254161968939137, -0.665223007764405132, 0.248983703498281314, -0.0332527231700357697,
            0.0023114179304694015, -0.0000991127741995080, 0.0000028916708643998, -0.0000000612108586630,
            0.0000000009838650793, -0.0000000000124235515, 0.0000000000001265433, -0.0000000000000010619,
            0.0000000000000000074]


class cheb_series:
    def __init__(self, c, order, a, b, order_sp):
        c_0 = c;
        order_0 = order;
        a_0 = a;
        b_0 = b;
        order_sp_0 = order_sp;

        self.c = c_0
        self.order = order_0
        self.a = a_0
        self.b = b_0
        self.order_sp = order_sp_0


bj0_cs = cheb_series(bj0_data, 12, -1, 1, 9)


def cheb_eval_e(cs, x, result):
    cs_0 = cs;
    x_0 = x;
    result_0 = result;
    dd_0 = None;
    dd_2 = None;
    dd_1 = None;
    dd_3 = None;
    temp_1 = None;
    temp_0 = None;
    temp_2 = None;
    d_0 = None;
    d_2 = None;
    d_1 = None;
    d_3 = None;
    e_0 = None;
    e_2 = None;
    e_1 = None;
    e_3 = None;
    y_0 = None;
    y2_0 = None;
    j_0 = None;
    j_2 = None;
    j_1 = None;
    j_3 = None;

    d_0 = 0.0
    dd_0 = 0.0
    y_0 = (2.0 * x_0 - cs_0.a - cs_0.b) / (cs_0.b - cs_0.a)
    y2_0 = 2.0 * y_0
    e_0 = 0.0
    j_0 = cs_0.order
    phi0 = Phi()
    while phi0.phiLoopTest(j_0, j_1) >= 1:
        phi0.set()
        dd_2 = phi0.phiEntry(dd_0, dd_1)
        temp_1 = phi0.phiEntry(None, temp_0)
        d_2 = phi0.phiEntry(d_0, d_1)
        e_2 = phi0.phiEntry(e_0, e_1)
        j_2 = phi0.phiEntry(j_0, j_1)

        temp_0 = d_2
        d_1 = y2_0 * d_2 - dd_2 + cs_0.c[j_2]
        e_1 = e_2 + fabs(y2_0 * temp_0) + fabs(dd_2) + fabs(cs_0.c[j_2])
        dd_1 = temp_0
        j_1 = j_2 - 1
    dd_3 = phi0.phiExit(dd_0, dd_1)
    temp_2 = phi0.phiExit(None, temp_0)
    d_3 = phi0.phiExit(d_0, d_1)
    e_3 = phi0.phiExit(e_0, e_1)
    j_3 = phi0.phiExit(j_0, j_1)
    result_0.val = d_3
    result_0.err = GSL_DBL_EPSILON * e_3 + fabs(cs_0.c[cs_0.order])
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS


GSL_SQRT_DBL_EPSILON = 1.4901161193847656e-08
GSL_ROOT5_DBL_EPSILON = 7.4009597974140505e-04
bm0_data = [0.09284961637381644, -0.00142987707403484, 0.00002830579271257, -0.00000143300611424, 0.00000012028628046,
            -0.00000001397113013, 0.00000000204076188, -0.00000000035399669, 0.00000000007024759, -0.00000000001554107,
            0.00000000000376226, -0.00000000000098282, 0.00000000000027408, -0.00000000000008091, 0.00000000000002511,
            -0.00000000000000814, 0.00000000000000275, -0.00000000000000096, 0.00000000000000034, -0.00000000000000012,
            0.00000000000000004]

_gsl_sf_bessel_amp_phase_bm0_cs = cheb_series(bm0_data, 20, -1, 1, 10)
bth0_data = [-0.24639163774300119, 0.001737098307508963, -0.000062183633402968, 0.000004368050165742,
             -0.000000456093019869, 0.000000062197400101, -0.000000010300442889, 0.000000001979526776,
             -0.000000000428198396, 0.000000000102035840, -0.000000000026363898, 0.000000000007297935,
             -0.000000000002144188, 0.000000000000663693, -0.000000000000215126, 0.000000000000072659,
             -0.000000000000025465, 0.000000000000009229, -0.000000000000003448, 0.000000000000001325,
             -0.000000000000000522, 0.000000000000000210, -0.000000000000000087, 0.000000000000000036]
_gsl_sf_bessel_amp_phase_bth0_cs = cheb_series(bth0_data, 23, -1, 1, 12)


def gsl_sf_bessel_cos_pi4_e(y, eps, result):
    y_1 = y;
    eps_0 = eps;
    result_1 = result;
    seps_0 = None;
    seps_1 = None;
    seps_2 = None;
    s_0 = None;
    d_4 = None;
    sy_0 = None;
    cy_0 = None;
    abs_sum_0 = None;
    ceps_0 = None;
    ceps_1 = None;
    ceps_2 = None;
    e2_0 = None;
    e2_1 = None;

    sy_0 = sin(y_1)
    cy_0 = cos(y_1)
    s_0 = sy_0 + cy_0
    d_4 = sy_0 - cy_0
    abs_sum_0 = fabs(cy_0) + fabs(sy_0)
    if fabs(eps_0) < GSL_ROOT5_DBL_EPSILON:
        e2_0 = eps_0 * eps_0
        seps_0 = eps_0 * (1.0 - e2_0 / 6.0 * (1.0 - e2_0 / 20.0))
        ceps_0 = 1.0 - e2_0 / 2.0 * (1.0 - e2_0 / 12.0)
    else:
        seps_1 = sin(eps_0)
        ceps_1 = cos(eps_0)
    phiPreds = [fabs(eps_0) < GSL_ROOT5_DBL_EPSILON]
    phiNames = [seps_0, seps_1]
    seps_2 = phiIf(phiPreds, phiNames)
    phiPreds = [fabs(eps_0) < GSL_ROOT5_DBL_EPSILON]
    phiNames = [ceps_0, ceps_1]
    ceps_2 = phiIf(phiPreds, phiNames)
    phiPreds = [fabs(eps_0) < GSL_ROOT5_DBL_EPSILON]
    phiNames = [e2_0, None]
    e2_1 = phiIf(phiPreds, phiNames)
    result_1.val = (ceps_2 * s_0 - seps_2 * d_4) / sqrt(2)
    result_1.err = 2.0 * GSL_DBL_EPSILON * (fabs(ceps_2) + fabs(seps_2)) * abs_sum_0 / sqrt(2)
    if y_1 > 1.0 / GSL_DBL_EPSILON:
        result.err = result.err * 0.5 * y_1
    elif y_1 > 1.0 / GSL_SQRT_DBL_EPSILON:
        result.err = result.err * 256.0 * y_1 * GSL_SQRT_DBL_EPSILON
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS


def GSL_ERROR_SELECT_2(a, b):
    a_1 = a;
    b_1 = b;
    lo = locals()
    record_locals(lo, test_counter)
    return a_1 if a_1 != GSL_SUCCESS else (b_1 if b_1 != GSL_SUCCESS else GSL_SUCCESS)


def GSL_ERROR_SELECT_3(a, b, c):
    a_2 = a;
    b_2 = b;
    c_1 = c;
    lo = locals()
    record_locals(lo, test_counter)
    return a_2 if a_2 != GSL_SUCCESS else GSL_ERROR_SELECT_2(b_2, c_1)


def GSL_ERROR_SELECT_4(a, b, c, d):
    a_3 = a;
    b_3 = b;
    c_2 = c;
    d_5 = d;
    lo = locals()
    record_locals(lo, test_counter)
    return a_3 if a_3 != GSL_SUCCESS else GSL_ERROR_SELECT_3(b_3, c_2, d_5)


def GSL_ERROR_SELECT_5(a, b, c, d, e):
    a_4 = a;
    b_4 = b;
    c_3 = c;
    d_6 = d;
    e_4 = e;
    lo = locals()
    record_locals(lo, test_counter)
    return a_4 if a_4 != GSL_SUCCESS else GSL_ERROR_SELECT_4(b_4, c_3, d_6, e_4)


def gsl_sf_bessel_J0_e(x, result):
    x_1 = x;
    result_2 = result;
    sqrty_0 = None;
    sqrty_1 = None;
    ct_0 = None;
    ct_1 = None;
    stat_ca_0 = None;
    stat_ca_1 = None;
    stat_cp_0 = None;
    stat_cp_1 = None;
    ampl_0 = None;
    ampl_1 = None;
    stat_ct_0 = None;
    stat_ct_1 = None;
    y_2 = None;
    z_0 = None;
    z_1 = None;
    cp_0 = None;
    cp_1 = None;
    ca_0 = None;
    ca_1 = None;

    y_2 = fabs(x_1)
    if y_2 < 2.0 * GSL_SQRT_DBL_EPSILON:
        result_2.val = 1.0
        result_2.err = y_2 * y_2
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif y_2 <= 4.0:
        lo = locals()
        record_locals(lo, test_counter)
        return cheb_eval_e(bj0_cs, 0.125 * y_2 * y_2 - 1.0, result_2)
    else:
        ca_0 = gsl_sf_result(0.0, 0.0)
        ct_0 = gsl_sf_result(0.0, 0.0)
        cp_0 = gsl_sf_result(0.0, 0.0)
        z_0 = 32.0 / (y_2 * y_2) - 1.0
        stat_ca_0 = cheb_eval_e(_gsl_sf_bessel_amp_phase_bm0_cs, z_0, ca_0)
        stat_ct_0 = cheb_eval_e(_gsl_sf_bessel_amp_phase_bth0_cs, z_0, ct_0)
        stat_cp_0 = gsl_sf_bessel_cos_pi4_e(y_2, ct_0.val / y_2, cp_0)
        sqrty_0 = sqrt(y_2)
        #bug should be ampl_0 = (0.75 + ca_0.val) / sqrty_0
        ampl_0 = (0.75 + ca_0.val) / sqrty_0 + bug
        
        result_2.val = ampl_0 * cp_0.val
        result_2.err = fabs(cp_0.val) * ca_0.err / sqrty_0 + fabs(ampl_0) * cp_0.err
        result.err = result.err + GSL_DBL_EPSILON * fabs(result_2.val)
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_ERROR_SELECT_3(stat_ca_0, stat_ct_0, stat_cp_0)



def GSL_ERROR_VAL(reason, gsl_errno, GSL_NAN):
    reason_0 = reason;
    gsl_errno_0 = gsl_errno;
    GSL_NAN_0 = GSL_NAN;
    lo = locals()
    record_locals(lo, test_counter)
    return


def EVAL_RESULT(fn, result):
    fn_0 = fn;
    result_3 = result;
    status_0 = None;

    status_0 = fn_0
    if status_0 != GSL_SUCCESS:
        GSL_ERROR_VAL(fn_0, status_0, result_3.val)
    lo = locals()
    record_locals(lo, test_counter)
    return result_3.val


def gsl_sf_bessel_J0_bad(x):
    x_2 = x;
    result_4 = None;
    fn_1 = None;

    result_4 = gsl_sf_result(0.0, 0.0)
    fn_1 = gsl_sf_bessel_J0_e(x_2, result_4)
    lo = locals()
    record_locals(lo, test_counter)
    return EVAL_RESULT(fn_1, result_4)




#-----end of bad program----------


def record_locals(lo, i):
    for name in lo:

        if isinstance(lo[name], numbers.Number) and name in causal_map:
            if name not in global_value_dict:
                columns = causal_map[name].copy()
                columns.insert(0, name)
                global_value_dict[name] = pd.DataFrame(columns=columns)
            new_row = [np.float64(lo[name])]

            for pa in causal_map[name]:
                if isinstance(lo[pa], numbers.Number):
                    new_row.append(np.float64(lo[pa]))
                else:
                    new_row.append(lo[pa])
            global_value_dict[name].loc[i] = new_row



            



causal_map = dict(cy_0=['y_1'], d_0=[], d_2=['d_0', 'd_1'], d_1=['y2_0', 'd_2', 'dd_2', 'cs_0', 'j_2'],
                  d_4=['sy_0', 'cy_0'], d_3=['d_0', 'd_1'], j_0=['cs_0'], dd_3=['dd_0', 'dd_1'],
                  abs_sum_0=['cy_0', 'sy_0'], j_2=['j_0', 'j_1'], dd_1=['temp_0'], dd_2=['dd_0', 'dd_1'], j_1=['j_2'],
                  dd_0=[], j_3=['j_0', 'j_1'], sy_0=['y_1'], ca_0=[], z_0=['y_2', 'y_2'], z_1=['z_0'],
                  stat_ct_1=['stat_ct_0'], temp_0=['d_2'], temp_1=[ 'temp_0'], temp_2=['temp_0'],
                  stat_ct_0=['z_0', 'ct_0'], stat_cp_1=['stat_cp_0'], stat_cp_0=['y_2', 'ct_0', 'y_2', 'cp_0'],
                  status_0=['fn_0'], ca_1=['ca_0'], seps_1=['eps_0'], stat_ca_1=['stat_ca_0'],
                  seps_0=['eps_0', 'e2_0', 'e2_0'], stat_ca_0=['z_0', 'ca_0'], seps_2=['seps_0', 'seps_1'],
                  fn_1=['x_2', 'result_4'], ct_0=[], ct_1=['ct_0'],
                  e_1=['e_2', 'y2_0', 'temp_0', 'dd_2', 'cs_0', 'j_2'], e_0=[], e_3=['e_0', 'e_1'], e_2=['e_0', 'e_1'],
                  ampl_1=['ampl_0'], ampl_0=['ca_0', 'sqrty_0'], ceps_1=['eps_0'], s_0=['sy_0', 'cy_0'],
                  ceps_0=['e2_0', 'e2_0'], y_0=['x_0', 'cs_0', 'cs_0', 'cs_0', 'cs_0'], ceps_2=['ceps_0', 'ceps_1'],
                  y_2=['x_1'], result_4=[], e2_0=['eps_0', 'eps_0'], cp_0=[], e2_1=['e2_0'], cp_1=['cp_0'],
                  sqrty_1=['sqrty_0'], sqrty_0=['y_2'], y2_0=['y_0'], )

global_value_dict = {}

arg1s = np.arange(0, 1000)
#arg2s = np.arange(2, 12)
outcome_list = []
bad_outcome_list = []
dist_list = []
bad_dict = {}

def fluky(good_val, bad_val, p):
        r = random.random()
        if r <= p:
            return bad_val
        else:
            return good_val


test_counter = 0
for arg1 in arg1s:
    bug = fluky(0,5,0.9)
    #outcome = gsl_sf_bessel_J0(arg1)
    bad_outcome = gsl_sf_bessel_J0_bad(arg1)
    #outcome_list.append(outcome)
    bad_dict[test_counter] = bad_outcome
    #dist_list.append(0 if abs(bad_outcome - outcome) == 0 else 1)
    test_counter += 1

diff_dict = {index : 0.0 if abs(bad_dict[index] - good_dict[index]) == 0 else 1.0 for index in bad_dict }

for key in global_value_dict:
    rows = global_value_dict[key].index
    outcome_list = [diff_dict[i] for i in rows]
    global_value_dict[key]['outcome'] = outcome_list


def get_quantiled_tr(W):
    # 10 quantiles from 0.05 to 0.95
    quantile_list = []
    for i in np.arange(0.05, 1.05, 0.1):
        quantile_list.append(W.quantile(i))
    return quantile_list


def predict_causal_risk_list(train_set_X, quantiles, model):

    risk_list = []
    print(train_set_X.columns[0] + " being treatment...")
    X_with_quantile = train_set_X.drop(train_set_X.columns[0], axis=1)

    for quantile in quantiles:
        X_with_quantile.insert(loc=0, column=train_set_X.columns[0],
                               value=np.full((len(X_with_quantile), 1), quantile))
        # X_with_quantile[train_set_X.columns[col_index_todrop]] = np.full((len(X_with_quantile), 1), quantile)
        # print(X_with_quantile.describe())
        risk_list.append(model.predict(X_with_quantile).mean())
        X_with_quantile = X_with_quantile.drop(train_set_X.columns[0], axis=1)
    return risk_list


def suspicious_ranking(global_value_dict, model_to_use):

    suspicious_df = pd.DataFrame(columns=['variable_name', 'max_risk_diff', 'quantile1', 'quantile2'])
    for name in global_value_dict:

        #df cleaning
        df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        train_set = df
        #train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set_X = train_set.drop(['outcome'], axis=1)
        train_set_Y = train_set['outcome']
        if model_to_use == 0:
            model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        if model_to_use == 1:
            model = Lasso(alpha=0.1)

        
        model.fit(train_set_X, train_set_Y)

        W = df.iloc[:, 0].to_frame()
        quantiles = get_quantiled_tr(W)
        risk_list = predict_causal_risk_list(train_set_X, quantiles, model)
        max_risk = max(risk_list)
        min_risk = min(risk_list)
        row = [df.columns[0], max_risk - min_risk, risk_list.index(max_risk), risk_list.index(min_risk)]
        suspicious_df.loc[len(suspicious_df)] = row
    return suspicious_df.sort_values(by='max_risk_diff', ascending=False)

# 0-> random forest  1 -> lasso
result = suspicious_ranking(global_value_dict, 0)
pd.set_option("display.precision", 8)
print(result)


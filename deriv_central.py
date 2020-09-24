from math import *
import numpy as np
GSL_SUCCESS = 1

def GSL_MAX(a,b):
    return max(a,b)

class gsl_function:
    def __init__(self, function, params):
        self.function = function
        self.params = params

GSL_DBL_EPSILON    =    2.2204460492503131e-16

def GSL_FN_EVAL(F, x):
    return F.function(x, F.params)

def central_deriv(f, x, h, result, abserr_round, abserr_trunc):
    fm1 = GSL_FN_EVAL(f, x - h)
    fp1 = GSL_FN_EVAL(f, x + h)
 
    fmh = GSL_FN_EVAL(f, x - h / 2)
    fph = GSL_FN_EVAL(f, x + h / 2)

    r3 = 0.5 * (fp1 - fm1)
    r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3

    e3 = (fabs(fp1) + fabs(fm1)) * GSL_DBL_EPSILON
    e5 = 2.0 * (fabs(fph) + fabs(fmh)) * GSL_DBL_EPSILON + e3

    dy = GSL_MAX(fabs(r3 / h), fabs(r5 / h)) *(fabs(x) / h) * GSL_DBL_EPSILON
    result = r5 / h
    abserr_trunc = fabs((r5 - r3) / h)
    abserr_round = fabs(e5 / h) + dy
    return result, abserr_trunc, abserr_round

def gsl_deriv_central(f, x, h, result, abseer):
    r_0 = 0.0
    round = 0.0
    trunc = 0.0
    error = 0.0
    r_0, round, trunc, = central_deriv (f, x, h, r_0, round, trunc)
    error = round + trunc

    if round < trunc and (round > 0 and trunc > 0):
        r_opt = 0.0
        round_opt = 0.0 
        trunc_opt = 0.0
        error_opt = 0.0 
        h_opt = h * pow (round / (2.0 * trunc), 1.0 / 3.0)
        central_deriv (f, x, h_opt, r_opt, round_opt, trunc_opt)
        error_opt = round_opt + trunc_opt
        if error_opt < error and fabs (r_opt - r_0) < 4.0 * error:
            r_0 = r_opt
            error = error_opt

    result = r_0
    abserr = error

    return result, abserr

def f(x, params):
    return pow(x, 1.5)



F = gsl_function(f, 0)
result = 0.0
abserr = 0.0

good_dict = {}
args0 = np.arange(0.1,100.1,0.1)
count = 0
for arg0 in args0:
    result = 0.0
    abserr = 0.0
    deriv_and_error = gsl_deriv_central(F, arg0, 1e-8, result, abserr) 
    good_dict[count] = deriv_and_error
    count+=1




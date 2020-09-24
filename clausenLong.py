from math import *
import numpy as np 

GSL_SUCCESS = 0
GSL_FAILURE  = -1
GSL_CONTINUE = -2
GSL_EDOM     = 1
GSL_ERANGE   = 2
GSL_EFAULT   = 3
GSL_EINVAL   = 4
GSL_EFAILED  = 5
GSL_EFACTOR  = 6
GSL_ESANITY  = 7
GSL_ENOMEM   = 8
GSL_EBADFUNC = 9
GSL_ERUNAWAY = 1
GSL_EMAXITER = 11
GSL_EZERODIV = 12
GSL_EBADTOL  = 13
GSL_ETOL     = 14
GSL_EUNDRFLW = 15
GSL_EOVRFLW  = 16
GSL_ELOSS    = 17
GSL_EROUND   = 18
GSL_EBADLEN  = 19
GSL_ENOTSQR  = 20
GSL_ESING    = 21
GSL_EDIVERGE = 22
GSL_EUNSUP   = 23
GSL_EUNIMPL  = 24
GSL_ECACHE   = 25
GSL_ETABLE   = 26
GSL_ENOPROG  = 27
GSL_ENOPROGJ = 28
GSL_ETOLF    = 29
GSL_ETOLX    = 30
GSL_ETOLG    = 31
GSL_EOF      = 32
GSL_DBL_EPSILON  =      2.2204460492503131e-16
GSL_SQRT_DBL_EPSILON  = 1.4901161193847656e-08

M_PI    =   3.14159265358979323846264338328 
class gsl_sf_result:
    def __init__(self, val, err):
        self.val = val
        self.err = err
def GSL_ERROR(reason, gsl_errno):
    return
def GSL_ERROR_VAL(reason, gsl_errno, value):
    return
def EVAL_RESULT(fn, result):

    status = fn
    if status != GSL_SUCCESS:
        GSL_ERROR_VAL(fn, status, result.val)
    return result.val

class cheb_series:
    def __init__(self, c, order, a, b, order_sp):
        self.c = c
        self.order = order
        self.a = a
        self.b = b
        self.order_sp = order_sp

aclaus_data = [2.142694363766688447e+00,
	0.723324281221257925e-01,
	0.101642475021151164e-02,
	0.3245250328531645e-04,
	0.133315187571472e-05,
	0.6213240591653e-07,
	0.313004135337e-08,
	0.16635723056e-09,
	0.919659293e-11,
	0.52400462e-12,
	0.3058040e-13,
	0.18197e-14,
	0.1100e-15,
	0.68e-17,
	0.4e-18]
aclaus_cs =  cheb_series(aclaus_data, 14, -1, 1, 8)



def cheb_eval_e(cs, x, result):
    d = 0.0
    dd = 0.0
    #added
    cs_a_IV = cs.a
    cs_b_IV = cs.b
    y = (2.0*x - cs_a_IV - cs_b_IV) / (cs_b_IV - cs_a_IV)
    y2 = 2.0 * y
    e = 0.0
    for j in range(cs.order, 0, -1):
        temp = d
        #add 
        cs_c_j_IV = cs.c[j]
        d = y2 * d - dd + cs_c_j_IV
        e += fabs(y2*temp) + fabs(dd) + fabs(cs_c_j_IV)
        dd = temp
    temp = d
    #add
    cs_c_0_IV = cs.c[0]
    d = y * d - dd + 0.5 * cs_c_0_IV
    e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs_c_0_IV)
    result_val_IV = d
    result.val = result_val_IV
    #add
    cs_c_cs_order_IV = cs.c[cs.order]
    result_err = GSL_DBL_EPSILON * e + fabs(cs_c_cs_order_IV)
    result.err = result_err
    return GSL_SUCCESS

def gsl_sf_angle_restrict_pos_err_e(theta, result):
    P1 = 4 * 7.85398125648498535156e-01
    P2 = 4 * 3.77489470793079817668e-08
    P3 = 4 * 2.69515142907905952645e-15
    TwoPi = 2 * (P1 + P2 + P3)
    y = 2 * floor(theta / TwoPi)
    r = ((theta - y * P1) - y * P2) - y * P3
    if r > TwoPi:
        r = (((r - 2 * P1) - 2 * P2) - 2 * P3)
    elif r < 0:
        r = (((r + 2 * P1) + 2 * P2) + 2 * P3)
    result.val = r
    
    if fabs(theta) > 0.0625 / GSL_DBL_EPSILON:
        result.val = float('nan')
        result.err = fabs(result.val)
        GSL_ERROR("error", GSL_ELOSS)
    elif fabs(theta) > 0.0625 / GSL_SQRT_DBL_EPSILON:
        result_val_IV = result.val
        result_err = GSL_DBL_EPSILON * fabs(result_val_IV - theta)
        result.err = result_err
        return GSL_SUCCESS
    else:
        result_val_IV = result.val
        delta = fabs(result_val_IV - theta)

        result_err = 2.0 * GSL_DBL_EPSILON * (delta if delta < M_PI else M_PI)
        result.err = result_err
        return GSL_SUCCESS

def gsl_sf_angle_restrict_pos_e(theta):
    r = gsl_sf_result(0.0, 0.0)
    stat = gsl_sf_angle_restrict_pos_err_e(theta, r)
    r_val_IV = r.val
    theta = r_val_IV
    return stat, theta

def gsl_sf_clausen_e(x, result):
    x_cut = M_PI * GSL_SQRT_DBL_EPSILON
    sgn = 1.0
    if x < 0.0:
        x = -x
        sgn = -1.0
    status_red,x = gsl_sf_angle_restrict_pos_e(x)
    if x > M_PI:
        p0 = 6.28125
        p1 = 0.19353071795864769253e-02
        x = (p0 - x) + p1
        sgn = -sgn
    if x == 0.0:
        result_val = 0.0
        result.val = result_val

        result_err = 0.0
        result.err = result_err
    elif x < x_cut:
        result_val = x * (1.0 - log(x))
        result.val = result_val
        result_err = x * GSL_DBL_EPSILON
        result.err = result_err
    else:
        t = 2.0*(x*x / (M_PI*M_PI) - 0.5)
        result_c = gsl_sf_result(0.0, 0.0)
        cheb_eval_e(aclaus_cs, t, result_c)
        result_c_val_IV = result_c.val
        result_val = x * (result_c_val_IV - log(x))
        result.val = result_val

        result_c_err_IV = result_c.err
        result_err = x * (result_c_err_IV + GSL_DBL_EPSILON)
        result.err = result_err

    result_val = result.val
    result_val *= sgn
    result.val = result_val
    return status_red

def gsl_sf_clausen(x):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_clausen_e(x, result), result)

good_dict = {}

args0 = np.arange(0, 10, 0.01)
cnt = 0
for arg0 in args0:
    good_dict[cnt] = gsl_sf_clausen(arg0)
    cnt += 1
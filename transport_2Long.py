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
GSL_LOG_DBL_EPSILON = (-3.6043653389117154e+01)

class cheb_series:
    def __init__(self, c, order, a, b, order_sp):
        self.c = c
        self.order = order
        self.a = a
        self.b = b
        self.order_sp = order_sp

class gsl_sf_result:
    def __init__(self, val, err):
        self.val = val
        self.err = err
def GSL_ERROR_VAL(reason, gsl_errno, value):
    return
def EVAL_RESULT(fn, result):
    status = fn
    if status != GSL_SUCCESS:
        GSL_ERROR_VAL(fn, status, result.val)
    return result.val

transport2_data = [1.671760446434538503,
	-0.147735359946794490,
	0.148213819946936338e-01,
	-0.14195330326305613e-02,
	0.1306541324415708e-03,
	-0.117155795867579e-04,
	0.10333498445756e-05,
	-0.901911304223e-07,
	0.78177169833e-08,
	-0.6744565684e-09,
	0.579946394e-10,
	-0.49747619e-11,
	0.425961e-12,
	-0.36422e-13,
	0.3111e-14,
	-0.265e-15,
	0.23e-16,
	-0.19e-17]

transport2_cs = cheb_series(transport2_data,
	17,
	-1, 1,
	9)

def cheb_eval_e(cs, x, result):
    d = 0.0
    dd = 0.0
    #added
    cs_a = cs.a
    cs_b = cs.b
    y = (2.0*x - cs_a - cs_b) / (cs_b - cs_a)
    y2 = 2.0 * y
    e = 0.0
    for j in range(cs.order, 0, -1):
        temp = d
        #add 
        cs_c_j = cs.c[j]
        d = y2 * d - dd + cs_c_j
        e += fabs(y2*temp) + fabs(dd) + fabs(cs_c_j)
        dd = temp
    temp = d
    #add
    cs_c_0 = cs.c[0]
    d = y * d - dd + 0.5 * cs_c_0
    e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs_c_0)
    result_val = d
    result.val = result_val
    #add
    cs_c_cs_order = cs.c[cs.order]
    result_err = GSL_DBL_EPSILON * e + fabs(cs_c_cs_order)
    result.err = result_err
    return GSL_SUCCESS

def transport_sumexp(numexp, order, t, x):
    rk = numexp
    sumexp = 0.0
    for k in range(1, numexp + 1):
        sum2 =  1.0
        xk = 1.0 / (rk * x)
        xk1 = 1.0
        for j in range(1, order + 1):
            sum2 = sum2 * xk1*xk + 1.0
            xk1 += 1.0
        sumexp *= t
        sumexp += sum2
        rk -= 1.0
    return sumexp

def gsl_sf_transport_2_e(x, result):
    val_infinity = 3.289868133696452873
    if x < 0.0:
        print("domian error")
    elif x < 3.0*GSL_SQRT_DBL_EPSILON:
        result_val = x
        result.val = result_val

        result_err = GSL_DBL_EPSILON * fabs(x) + x * x / 2.0
        result.err = result_err
        return GSL_SUCCESS
    elif x <= 4.0:
        t = (x*x / 8.0 - 0.5) - 0.5
        result_c = gsl_sf_result(0.0, 0.0)
        cheb_eval_e(transport2_cs, t, result_c)
        result_c_val = result_c.val
        result_val = x * result_c_val
        result.val = result_val
        result_c_err = result_c.err
        result_err = x * result_c_err
        result.err = result_err
        result_val = result.val
        result_err = result.err
        result_err += 2.0 * GSL_DBL_EPSILON * fabs(result.val)
        result.err = result_err
        return GSL_SUCCESS
    elif x < -GSL_LOG_DBL_EPSILON:
        numexp = (int)((-GSL_LOG_DBL_EPSILON) / x) + 1
        sumexp = transport_sumexp(numexp, 2, exp(-x), x)
        t = 2.0 * log(x) - x + log(sumexp)
        if t < GSL_LOG_DBL_EPSILON:

            result_val = val_infinity
            result.val = result_val

            result_err = 2.0 * GSL_DBL_EPSILON * val_infinity
            result.err = result_err
        else:
            et = exp(t)
            result_val = val_infinity - et
            result.val = result_val

            result_err = 2.0 * GSL_DBL_EPSILON * (val_infinity + fabs(t) * et)
            result.err = result_err
        return GSL_SUCCESS
    elif x < 2.0 / GSL_DBL_EPSILON:
        numexp = 1
        sumexp = transport_sumexp(numexp, 2, 1.0, x)
        t = 2.0 * log(x) - x + log(sumexp)
        if t < GSL_LOG_DBL_EPSILON:
            result_val = val_infinity
            result.val = result_val
            result_err = 2.0 * GSL_DBL_EPSILON * val_infinity
            result.err = result_err
        else:
            et = exp(t)
            result_val = val_infinity - et
            result.val = result_val
            result_err = 2.0 * GSL_DBL_EPSILON * (val_infinity + (fabs(t) + 1.0) * et)
            result.err = result_err
        return GSL_SUCCESS
    else:
        t = 2.0 * log(x) - x
        if t < GSL_LOG_DBL_EPSILON:
            result_val = val_infinity
            result.val = result_val
            result_err = 2.0 * GSL_DBL_EPSILON * val_infinity
            result.err = result_err
        else:
            et = exp(t)
            result_val = val_infinity - et
            result.val = result_val
            result_err = 2.0 * GSL_DBL_EPSILON * (val_infinity + (fabs(t) + 1.0) * et)
            result.err = result_err
        return GSL_SUCCESS

def gsl_sf_transport_2(x):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_transport_2_e(x, result), result)

cnt = 0
good_dict = {}
args0 = np.arange(0.0, 10, 0.01)
for arg0 in args0:
    good_dict[cnt] = gsl_sf_transport_2(arg0)
    cnt+=1





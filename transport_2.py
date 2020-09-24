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

    y = (2.0*x - cs.a - cs.b) / (cs.b - cs.a)
    y2 = 2.0 * y
    e = 0.0
    for j in range(cs.order, 0, -1):
        temp = d
        d = y2 * d - dd + cs.c[j]
        e += fabs(y2*temp) + fabs(dd) + fabs(cs.c[j])
        dd = temp
    temp = d
    d = y * d - dd + 0.5 * cs.c[0]
    e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs.c[0])
    result.val = d
    result.err = GSL_DBL_EPSILON * e + fabs(cs.c[cs.order])
    return GSL_SUCCESS

def transport_sumexp(numexp, order, t, x):
    rk = numexp
    sumexp = 0.0
    for k in range(1, numexp):
        sum2 =  1.0
        xk = 1.0 / (rk * x)
        xk1 = 1.0
        for j in range(j, order):
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
        result.val = x
        result.err = GSL_DBL_EPSILON * fabs(x) + x * x / 2.0

        return GSL_SUCCESS

    elif x <= 4.0:
        t = (x*x / 8.0 - 0.5) - 0.5
        result_c = gsl_sf_result(0.0, 0.0)
        cheb_eval_e(transport2_cs, t, result_c)
        result.val = x * result_c.val
        result.err = x * result_c.err
        result.err += 2.0 * GSL_DBL_EPSILON * fabs(result.val)

        return GSL_SUCCESS

    elif x < -GSL_LOG_DBL_EPSILON:
        numexp = (int)((-GSL_LOG_DBL_EPSILON) / x) + 1
        sumexp = transport_sumexp(numexp, 2, exp(-x), x)
        t = 2.0 * log(x) - x + log(sumexp)
        if t < GSL_LOG_DBL_EPSILON:
            result.val = val_infinity
            result.err = 2.0 * GSL_DBL_EPSILON * val_infinity
        else:
            et = exp(t)
            result.val = val_infinity - et
            result.err = 2.0 * GSL_DBL_EPSILON * (val_infinity + fabs(t) * et)

        return GSL_SUCCESS
    elif x < 2.0 / GSL_DBL_EPSILON:

        numexp = 1
        sumexp = transport_sumexp(numexp, 2, 1.0, x)
        t = 2.0 * log(x) - x + log(sumexp)
        if t < GSL_LOG_DBL_EPSILON:
            result.val = val_infinity
            result.err = 2.0 * GSL_DBL_EPSILON * val_infinity
        else:
            et = exp(t)
            result.val = val_infinity - et
            result.err = 2.0 * GSL_DBL_EPSILON * (val_infinity + (fabs(t) + 1.0) * et)
            
        return GSL_SUCCESS
    else:
        t = 2.0 * log(x) - x
        if t < GSL_LOG_DBL_EPSILON:
            result.val = val_infinity
            result.err = 2.0 * GSL_DBL_EPSILON * val_infinity
        else:
            et = exp(t)
            result.val = val_infinity - et
            result.err = 2.0 * GSL_DBL_EPSILON * (val_infinity + (fabs(t) + 1.0) * et)

        return GSL_SUCCESS

def gsl_sf_transport_2(x):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_transport_2_e(x, result), result)

print(gsl_sf_transport_2(4))





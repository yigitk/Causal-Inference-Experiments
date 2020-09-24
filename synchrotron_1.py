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
GSL_LOG_DBL_MIN  = (-7.0839641853226408e+02)
M_SQRT2  =  1.41421356237309504880168872421
M_SQRT3  =  1.73205080756887729352744634151

def GSL_IS_ODD(n):
    return (n % 2) == 1
def GSL_ERROR_VAL(reason, gsl_errno, value):
    return
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

def GSL_ERROR(reason, gsl_errno):
    return

def EVAL_RESULT(fn, result):
    status = fn
    if status != GSL_SUCCESS:
        GSL_ERROR_VAL(fn, status, result.val)
    return result.val

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

def gsl_sf_pow_int_e(x, n, result):
    value = 1.0
    count = 0
    
    if n < 0:
        n = -n

        if x == 0.0:
            u = 1.0 / x
            result.val = u if n % 2 == 1 else u * u
            result.err = inf
            print("overflow err")
        x = 1.0 / x
    
    while True:
        
        if GSL_IS_ODD(n):
            value *= x
        n >>= 1
        
        x *= x
        
        count += 1 
        
        if n == 0:
            break
    print(value)
    result.val = value
    result.err = 2.0 * GSL_DBL_EPSILON * (count + 1.0) * fabs(value)
    return GSL_SUCCESS

def gsl_sf_pow_int(x, n):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_pow_int_e(x,n, result), result)

synchrotron1a_data = [2.1329305161355000985,
	0.741352864954200240e-01,
	0.86968099909964198e-02,
	0.11703826248775692e-02,
	0.1645105798619192e-03,
	0.240201021420640e-04,
	0.35827756389389e-05,
	0.5447747626984e-06,
	0.838802856196e-07,
	0.13069882684e-07,
	0.2053099071e-08,
	0.325187537e-09,
	0.517914041e-10,
	0.83002988e-11,
	0.13352728e-11,
	0.2159150e-12,
	0.349967e-13,
	0.56994e-14,
	0.9291e-15,
	0.152e-15,
	0.249e-16,
	0.41e-17,
	0.7e-18]

synchrotron1_data = [30.364682982501076273,
	17.079395277408394574,
	4.560132133545072889,
	0.549281246730419979,
	0.372976075069301172e-01,
	0.161362430201041242e-02,
	0.481916772120371e-04,
	0.10512425288938e-05,
	0.174638504670e-07,
	0.22815486544e-09,
	0.240443082e-11,
	0.2086588e-13,
	0.15167e-15]

synchrotron2_data = [0.4490721623532660844,
	0.898353677994187218e-01,
	0.81044573772151290e-02,
	0.4261716991089162e-03,
	0.147609631270746e-04,
	0.3628633615300e-06,
	0.66634807498e-08,
	0.949077166e-10,
	0.1079125e-11,
	0.10022e-13,
	0.77e-16,
	0.5e-18]

synchrotron1_cs = cheb_series(synchrotron1_data,
	12,
	-1.0, 1.0,
	9)

synchrotron2_cs = cheb_series(synchrotron2_data,
	11,
	-1.0, 1.0,
	7)

synchrotron1a_cs = cheb_series(synchrotron1a_data,
	22,
	-1.0, 1.0,
	11)

def gsl_sf_synchrotron_1_e(x, result):
    if x < 0.0:
        print("domain error")
    elif x < 2.0*M_SQRT2 * GSL_SQRT_DBL_EPSILON:
        
        z = pow(x, 1.0/3.0)
        cf = 1 - 8.43812762813205e-01 * z * z
        result.val = 2.14952824153447863671 * z * cf
        result.err = GSL_DBL_EPSILON * result.val
        return GSL_SUCCESS
    elif x <= 4.0:
        
        c0   = M_PI/M_SQRT3
        px   = pow(x,1.0/3.0)
        
        px11 = gsl_sf_pow_int(px,11)
        
        t = x*x/8.0 - 1.0
        result_c1 = gsl_sf_result(0, 0)
        result_c2 = gsl_sf_result(0, 0)
        cheb_eval_e(synchrotron1_cs, t, result_c1)
        cheb_eval_e(synchrotron2_cs, t, result_c2)
        
        result.val  = px * result_c1.val - px11 * result_c2.val - c0 * x
        result.err  = px * result_c1.err + px11 * result_c2.err + c0 * x * GSL_DBL_EPSILON
        result.err += 2.0 * GSL_DBL_EPSILON * fabs(result.val)
        
        return GSL_SUCCESS
    elif x < -8.0*GSL_LOG_DBL_MIN/7.0:
        c0 = 0.2257913526447274323630976
        t = (12.0 - x) / (x + 4.0)
        result_c1 = gsl_sf_result(0, 0)
        cheb_eval_e(synchrotron1a_cs, t, result_c1)
        result.val = sqrt(x) * result_c1.val * exp(c0 - x)
        result.err = 2.0 * GSL_DBL_EPSILON * result.val * (fabs(c0-x)+1.0)
        return GSL_SUCCESS
    else:
        print("underflow error")

def gsl_sf_synchrotron_1(x):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_synchrotron_1_e(x, result), result)

print(gsl_sf_synchrotron_1(2))


from collections import namedtuple
from math import fabs,sin, cos, sqrt
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
GSL_DBL_EPSILON    =    2.2204460492503131e-16
class gsl_sf_result:
    def __init__(self, val, err):
        self.val = val
        self.err = err
bj0_data = [
    0.100254161968939137,
    -0.665223007764405132,
    0.248983703498281314,
    -0.0332527231700357697,
    0.0023114179304694015,
    -0.0000991127741995080,
    0.0000028916708643998,
    -0.0000000612108586630,
    0.0000000009838650793,
    -0.0000000000124235515,
    0.0000000000001265433,
    -0.0000000000000010619,
    0.0000000000000000074]


class cheb_series:
    def __init__(self, c, order, a, b, order_sp):
        self.c = c
        self.order = order
        self.a = a
        self.b = b
        self.order_sp = order_sp

bj0_cs = cheb_series(bj0_data,12,-1, 1,9)


def cheb_eval_e(cs, x, result):
    d = 0.0
    dd = 0.0
    y = (2.0 * x - cs.a - cs.b) / (cs.b - cs.a)
    y2 = 2.0 * y
    e = 0.0
    j = cs.order
    while j >= 1:
        temp = d
        d = y2 * d -dd + cs.c[j]
        e += fabs(y2 * temp) + fabs(dd) + fabs(cs.c[j])
        dd = temp
        j -= 1
    result.val = d
    result.err = GSL_DBL_EPSILON * e + fabs(cs.c[cs.order])
    return GSL_SUCCESS

GSL_SQRT_DBL_EPSILON = 1.4901161193847656e-08
GSL_ROOT5_DBL_EPSILON = 7.4009597974140505e-04
bm0_data = [0.09284961637381644, -0.00142987707403484,
    0.00002830579271257,
    -0.00000143300611424,
    0.00000012028628046,
    -0.00000001397113013,
    0.00000000204076188,
    -0.00000000035399669,
    0.00000000007024759,
    -0.00000000001554107,
    0.00000000000376226,
    -0.00000000000098282,
    0.00000000000027408,
    -0.00000000000008091,
    0.00000000000002511,
    -0.00000000000000814,
    0.00000000000000275,
    -0.00000000000000096,
    0.00000000000000034,
    -0.00000000000000012,
    0.00000000000000004]

_gsl_sf_bessel_amp_phase_bm0_cs = cheb_series(bm0_data, 20, -1, 1, 10)

bth0_data = [-0.24639163774300119,
    0.001737098307508963,
    -0.000062183633402968,
    0.000004368050165742,
    -0.000000456093019869,
    0.000000062197400101,
    -0.000000010300442889,
    0.000000001979526776,
    -0.000000000428198396,
    0.000000000102035840,
    -0.000000000026363898,
    0.000000000007297935,
    -0.000000000002144188,
    0.000000000000663693,
    -0.000000000000215126,
    0.000000000000072659,
    -0.000000000000025465,
    0.000000000000009229,
    -0.000000000000003448,
    0.000000000000001325,
    -0.000000000000000522,
    0.000000000000000210,
    -0.000000000000000087,
    0.000000000000000036]
_gsl_sf_bessel_amp_phase_bth0_cs = cheb_series(bth0_data, 23, -1, 1, 12)

def gsl_sf_bessel_cos_pi4_e(y, eps, result):
    sy = sin(y)
    cy = cos(y)
    s = sy + cy
    d = sy - cy
    abs_sum = fabs(cy) + fabs(sy)
    if fabs(eps) < GSL_ROOT5_DBL_EPSILON:
        e2 = eps * eps
        seps = eps * (1.0 - e2 / 6.0 * (1.0 - e2 / 20.0))
        ceps = 1.0 - e2 / 2.0 * (1.0 - e2 / 12.0)
    else:
        seps = sin(eps)
        ceps = cos(eps)
    result.val = (ceps * s - seps * d) / sqrt(2)
    result.err = 2.0 * GSL_DBL_EPSILON * (fabs(ceps) + fabs(seps)) * abs_sum / sqrt(2)
    if y > 1.0 / GSL_DBL_EPSILON:
        result.err *= 0.5 * y
    elif y > 1.0 / GSL_SQRT_DBL_EPSILON:
        result.err *= 256.0 * y * GSL_SQRT_DBL_EPSILON
    return GSL_SUCCESS


def GSL_ERROR_SELECT_2(a,b):
    return a if a != GSL_SUCCESS else (b if b != GSL_SUCCESS else GSL_SUCCESS)


def GSL_ERROR_SELECT_3(a,b,c):
    return a if a != GSL_SUCCESS else GSL_ERROR_SELECT_2(b,c)


def GSL_ERROR_SELECT_4(a,b,c,d):
    return a if a != GSL_SUCCESS else GSL_ERROR_SELECT_3(b,c,d)


def GSL_ERROR_SELECT_5(a,b,c,d,e):
    return a if a != GSL_SUCCESS else GSL_ERROR_SELECT_4(b,c,d,e)


def gsl_sf_bessel_J0_e(x, result):
    y = fabs(x)
    if y < 2.0 * GSL_SQRT_DBL_EPSILON:
        result.val = 1.0
        result.err = y * y
        return GSL_SUCCESS
    elif y <= 4.0:
        return cheb_eval_e(bj0_cs,0.125*y*y - 1.0, result)
    else:
        ca = gsl_sf_result(0.0, 0.0)
        ct = gsl_sf_result(0.0, 0.0)
        cp = gsl_sf_result(0.0, 0.0)
        z = 32.0/(y*y) - 1.0
        stat_ca = cheb_eval_e(_gsl_sf_bessel_amp_phase_bm0_cs, z, ca)
        stat_ct = cheb_eval_e(_gsl_sf_bessel_amp_phase_bth0_cs, z ,ct)
        stat_cp = gsl_sf_bessel_cos_pi4_e(y, ct.val / y,  cp)
        sqrty = sqrt(y)
        ampl = (0.75 + ca.val) / sqrty
        result.val = ampl * cp.val

        result.err = fabs(cp.val) * ca.err / sqrty + fabs(ampl) * cp.err
        result.err += GSL_DBL_EPSILON * fabs(result.val)
        return GSL_ERROR_SELECT_3(stat_ca, stat_ct, stat_cp)


def GSL_ERROR_VAL(reason, gsl_errno, GSL_NAN):
    return


def EVAL_RESULT(fn, result):

    status = fn
    if status != GSL_SUCCESS:
        GSL_ERROR_VAL(fn, status, result.val)

    return result.val


def gsl_sf_bessel_J0(x):
    result = gsl_sf_result(0.0, 0.0)
    fn = gsl_sf_bessel_J0_e(x, result)
    return EVAL_RESULT(fn, result)

good_dict = {}

args0 = np.arange(0, 1000)
for arg0 in args0:
    good_dict[arg0] = gsl_sf_bessel_J0(arg0)


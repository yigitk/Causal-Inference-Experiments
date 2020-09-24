import math
import numpy as np

M_LN2 = 0.69314718055994530941723212146
DBL_MIN_EXP = -1021

def gsl_finite(x):
    return np.isfinite(x)

def gsl_frexp(x, e):
    if x == 0.0:
        e = 0

        return 0.0, e
    elif not gsl_finite(x):
        e = 0

        return x, e
    elif abs(x) >= 0.5 and abs(x) < 1:
        e = 0
        
        return x, e
    else:
        ex = math.ceil(math.log(abs(x)) / M_LN2)
        ei = ex
        
        if ei < DBL_MIN_EXP:
            ei  = DBL_MIN_EXP
        if ei > -DBL_MIN_EXP:
            ei = -DBL_MIN_EXP
        f = x * pow(2.0, -ei)

        if not gsl_finite(f):
            e = 0
            return f, e
        
        while abs(f) >= 1.0:
            ei += 1
            f /= 2.0 
        
        while abs(f) > 0 and abs(f) < 0.5:
            ei -= 1
            f *= 2.0
        
        e = ei
        return f, e

good_dict = {}

args0 = np.arange(0, 1000)
for arg0 in args0:
    e = 0.0
    good_dict[arg0] = gsl_frexp(arg0, e)


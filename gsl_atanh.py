import math
import numpy as np
GSL_DBL_EPSILON = 2.2204460492503131e-16
GSL_POSINF = math.inf
GSL_NEGINF = -math.inf
GSL_NAN = math.nan

def gsl_log1p(x):
    y = 1 + x
    z = y - 1
    to_return = math.log(y) - (z - x) / y
    return to_return

def gsl_atanh(x):
    a = abs(x)
    s = -1 if x < 0 else 1
    if a > 1:
        return GSL_NAN
    elif a == 1:
        to_return = GSL_NEGINF if x < 0 else GSL_POSINF
        return to_return
    elif a >= 0.5:
        to_return = s * 0.5 * gsl_log1p(2 * a / (1 - a))
        return to_return
    elif a > GSL_DBL_EPSILON:
        to_return = s * 0.5 * gsl_log1p(2 * a + 2 * a * a / (1 - a))
        return to_return
    else:
        return x

good_dict = {}
args0 = np.arange(0, 1, 0.001)
test_counter = 0
for arg0 in args0:
    good_dict[test_counter] = gsl_atanh(arg0)
    test_counter += 1
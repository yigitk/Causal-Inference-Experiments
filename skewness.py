import math
import numpy as np

def gsl_stats_mean(data, stride, size):

    mean = 0
    for i in range(0, size):
        mean += (data[i * stride] - mean) / (i + 1)
        
    return mean

def compute_variance(data, stride, n, mean):

    variance = 0
    for i in range(0, n):
        delta = (data[i * stride] - mean);
        variance += (delta * delta - variance) / (i + 1)

    return variance

def gsl_stats_sd_m(data, stride, n, mean):

    variance = compute_variance(data, stride, n, mean)
    sd = math.sqrt(variance * (n / (n - 1)))

    return sd

def gsl_stats_skew_m_sd(data, stride, n, mean, sd):

    skew = 0
    for i in range(0, n):
        x = (data[i * stride] - mean) / sd
        skew += (x * x * x - skew) / (i + 1)

    return skew

def gsl_stats_skew(data, stride, n):

    mean = gsl_stats_mean(data, stride, n)
    sd = gsl_stats_sd_m(data, stride, n, mean)
    skewness = gsl_stats_skew_m_sd(data, stride, n, mean, sd)
    return skewness


good_dict = {}
args0 = []
for i in range(0, 1000):
    size = np.random.randint(low = 10, high = 100)
    data = np.random.normal(size = size)
    args0.append(data)
    skewness = gsl_stats_skew(data, 1, size)
    good_dict[i] = skewness
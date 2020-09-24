import numpy as np
import pandas as pd
import numbers
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from phi import *
import random
import os
from math import *
import sys
from helpers import *

insertion_count = 0

from synchrotron_1Long import good_dict
os.system('python synchrotron_1Long.py') 
from phi import *

from math import *
import numpy as np
GSL_SUCCESS=0 
GSL_FAILURE=-1 
GSL_CONTINUE=-2 
GSL_EDOM=1 
GSL_ERANGE=2 
GSL_EFAULT=3 
GSL_EINVAL=4 
GSL_EFAILED=5 
GSL_EFACTOR=6 
GSL_ESANITY=7 
GSL_ENOMEM=8 
GSL_EBADFUNC=9 
GSL_ERUNAWAY=1 
GSL_EMAXITER=11 
GSL_EZERODIV=12 
GSL_EBADTOL=13 
GSL_ETOL=14 
GSL_EUNDRFLW=15 
GSL_EOVRFLW=16 
GSL_ELOSS=17 
GSL_EROUND=18 
GSL_EBADLEN=19 
GSL_ENOTSQR=20 
GSL_ESING=21 
GSL_EDIVERGE=22 
GSL_EUNSUP=23 
GSL_EUNIMPL=24 
GSL_ECACHE=25 
GSL_ETABLE=26 
GSL_ENOPROG=27 
GSL_ENOPROGJ=28 
GSL_ETOLF=29 
GSL_ETOLX=30 
GSL_ETOLG=31 
GSL_EOF=32 
GSL_DBL_EPSILON=2.2204460492503131e-16 
GSL_SQRT_DBL_EPSILON=1.4901161193847656e-08 
M_PI=3.14159265358979323846264338328 
GSL_LOG_DBL_MIN=(-7.0839641853226408e+02) 
M_SQRT2=1.41421356237309504880168872421 
M_SQRT3=1.73205080756887729352744634151 
def GSL_IS_ODD(n):
    n_0 = n;
    

    return (n_0%2)==1

def GSL_ERROR_VAL(reason,gsl_errno,value):
    reason_0 = reason;gsl_errno_0 = gsl_errno;value_0 = value;
    

    return 

class cheb_series:
    def __init__(self,c,order,a,b,order_sp):
        c_0 = c;order_0 = order;a_0 = a;b_0 = b;order_sp_0 = order_sp;
        

        self.c=c_0 
        self.order=order_0 
        self.a=a_0 
        self.b=b_0 
        self.order_sp=order_sp_0 

class gsl_sf_result:
    def __init__(self,val,err):
        val_0 = val;err_0 = err;
        

        self.val=val_0 
        self.err=err_0 

def GSL_ERROR(reason,gsl_errno):
    reason_1 = reason;gsl_errno_1 = gsl_errno;
    

    return 

def EVAL_RESULT(fn,result):
    fn_0 = fn;result_0 = result;
    result_val_IV_0=None;status_0=None;

    status_0=fn_0 
    if status_0!=GSL_SUCCESS:
        GSL_ERROR_VAL(fn_0,status_0,result_0.val) 
    result_val_IV_0=result_0.val 
    lo = locals()
    record_locals(lo, test_counter)
    return result_val_IV_0

def cheb_eval_e(cs,x,result):
    cs_0 = cs;x_0 = x;result_1 = result;
    dd_0=None;dd_2=None;dd_1=None;dd_3=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;cs_c_j_IV_1=None;cs_c_j_IV_0=None;cs_c_j_IV_2=None;cs_c_cs_order_IV_0=None;cs_c_0_IV_0=None;result_err_0=None;result_val_0=None;cs_a_IV_0=None;cs_b_IV_0=None;y_0=None;y2_0=None;

    gen_bad = random() < probability
    global insertion_count
    if gen_bad:
        insertion_count += 1
        
    d_0=0.0 
    dd_0=0.0 
    cs_a_IV_0=cs_0.a 
    cs_b_IV_0=cs_0.b 
    y_0=(2.0*x_0-cs_a_IV_0-cs_b_IV_0)/(cs_b_IV_0-cs_a_IV_0) 
    y2_0=2.0*y_0 
    e_0=0.0 
    phi0 = Phi()
    for j_0 in range(cs_0.order,0,-1):
        phi0.set()
        dd_2 = phi0.phiEntry(dd_0,dd_1)
        temp_1 = phi0.phiEntry(None,temp_0)
        d_2 = phi0.phiEntry(d_0,d_1)
        e_2 = phi0.phiEntry(e_0,e_1)
        cs_c_j_IV_1 = phi0.phiEntry(None,cs_c_j_IV_0)

        temp_0=d_2 
        cs_c_j_IV_0=cs_0.c[j_0] 
        d_1=y2_0*d_2-dd_2+cs_c_j_IV_0 
        e_1 = e_2+fabs(y2_0*temp_0)+fabs(dd_2)+fabs(cs_c_j_IV_0)
        dd_1=temp_0 
    dd_3 = phi0.phiExit(dd_0,dd_1)
    temp_2 = phi0.phiExit(None,temp_0)
    d_3 = phi0.phiExit(d_0,d_1)
    e_3 = phi0.phiExit(e_0,e_1)
    cs_c_j_IV_2 = phi0.phiExit(None,cs_c_j_IV_0)
    temp_3=d_3 
    cs_c_0_IV_0=cs_0.c[0] 
    d_4=fuzzy(y_0*d_3-dd_3+0.5*cs_c_0_IV_0, gen_bad)
    e_4 = e_3+fabs(y_0*temp_3)+fabs(dd_3)+0.5*fabs(cs_c_0_IV_0)
    result_val_0=d_4 
    result_1.val=result_val_0 
    cs_c_cs_order_IV_0=cs_0.c[cs_0.order] 
    result_err_0=GSL_DBL_EPSILON*e_4+fabs(cs_c_cs_order_IV_0) 
    result_1.err=result_err_0 
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS

def gsl_sf_pow_int_e(x,n,result):
    x_1 = x;n_1 = n;result_2 = result;
    result_val_1=None;result_val_2=None;result_val_3=None;result_val_4=None;u_0=None;u_1=None;u_2=None;x_2=None;x_3=None;x_5=None;x_4=None;x_6=None;count_0=None;count_2=None;count_1=None;count_3=None;result_err_1=None;result_err_2=None;result_err_3=None;result_err_4=None;value_1=None;value_4=None;value_2=None;value_3=None;value_5=None;n_2=None;n_3=None;n_5=None;n_4=None;n_6=None;

    value_1=1.0 
    count_0=0 
    if n_1<0:
        n_2=-n_1 
        if x_1==0.0:
            u_0=1.0/x_1 
            result_val_1=u_0 if n_2%2==1 else u_0*u_0 
            result_err_1=inf 
            result_2.val=result_val_1 
            result_2.err=result_err_1 
            print("overflow err") 
        phiPreds = [x_1==0.0]
        phiNames = [result_val_1,None]
        result_val_2= phiIf(phiPreds, phiNames)
        phiPreds = [x_1==0.0]
        phiNames = [u_0,None]
        u_1= phiIf(phiPreds, phiNames)
        phiPreds = [x_1==0.0]
        phiNames = [result_err_1,None]
        result_err_2= phiIf(phiPreds, phiNames)
        x_2=1.0/x_1 
    phiPreds = [n_1<0]
    phiNames = [result_val_2,None]
    result_val_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_1<0]
    phiNames = [u_1,None]
    u_2= phiIf(phiPreds, phiNames)
    phiPreds = [n_1<0]
    phiNames = [x_2,x_1]
    x_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_1<0]
    phiNames = [result_err_2,None]
    result_err_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_1<0]
    phiNames = [n_2,n_1]
    n_3= phiIf(phiPreds, phiNames)
    phi0 = Phi()
    while True:
        phi0.set()
        x_5 = phi0.phiEntry(x_3,x_4)
        count_2 = phi0.phiEntry(count_0,count_1)
        value_4 = phi0.phiEntry(value_1,value_3)
        n_5 = phi0.phiEntry(n_3,n_4)

        if GSL_IS_ODD(n_5):
            value_2 = value_4*x_5
        phiPreds = [GSL_IS_ODD(n_5)]
        phiNames = [value_2,value_4]
        value_3= phiIf(phiPreds, phiNames)
        n_4 = n_5>>1
        x_4 = x_5*x_5
        count_1 = count_2+1
        if n_4==0:
            break
    x_6 = phi0.phiExit(x_3,x_4)
    count_3 = phi0.phiExit(count_0,count_1)
    value_5 = phi0.phiExit(value_1,value_3)
    n_6 = phi0.phiExit(n_3,n_4)
    result_val_4=value_5 
    result_err_4=2.0*GSL_DBL_EPSILON*(count_3+1.0)*fabs(value_5) 
    result_2.val=result_val_4 
    result_2.err=result_err_4 
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS

def gsl_sf_pow_int(x,n):
    x_7 = x;n_7 = n;
    result_3=None;return_val_0=None;

    result_3=gsl_sf_result(0.0,0.0) 
    return_val_0=EVAL_RESULT(gsl_sf_pow_int_e(x_7,n_7,result_3),result_3) 
    lo = locals()
    record_locals(lo, test_counter)
    return return_val_0

synchrotron1a_data=[2.1329305161355000985,0.741352864954200240e-01,0.86968099909964198e-02,0.11703826248775692e-02,0.1645105798619192e-03,0.240201021420640e-04,0.35827756389389e-05,0.5447747626984e-06,0.838802856196e-07,0.13069882684e-07,0.2053099071e-08,0.325187537e-09,0.517914041e-10,0.83002988e-11,0.13352728e-11,0.2159150e-12,0.349967e-13,0.56994e-14,0.9291e-15,0.152e-15,0.249e-16,0.41e-17,0.7e-18] 
synchrotron1_data=[30.364682982501076273,17.079395277408394574,4.560132133545072889,0.549281246730419979,0.372976075069301172e-01,0.161362430201041242e-02,0.481916772120371e-04,0.10512425288938e-05,0.174638504670e-07,0.22815486544e-09,0.240443082e-11,0.2086588e-13,0.15167e-15] 
synchrotron2_data=[0.4490721623532660844,0.898353677994187218e-01,0.81044573772151290e-02,0.4261716991089162e-03,0.147609631270746e-04,0.3628633615300e-06,0.66634807498e-08,0.949077166e-10,0.1079125e-11,0.10022e-13,0.77e-16,0.5e-18] 
synchrotron1_cs=cheb_series(synchrotron1_data,12,-1.0,1.0,9) 
synchrotron2_cs=cheb_series(synchrotron2_data,11,-1.0,1.0,7) 
synchrotron1a_cs=cheb_series(synchrotron1a_data,22,-1.0,1.0,11) 
def gsl_sf_synchrotron_1_e(x,result):
    x_8 = x;result_4 = result;
    cf_0=None;cf_1=None;result_c1_err_IV_0=None;result_c1_err_IV_1=None;result_c2_err_IV_0=None;result_c2_err_IV_1=None;px_0=None;px_1=None;result_err_5=None;result_err_6=None;result_err_7=None;result_err_8=None;result_err_9=None;result_err_10=None;result_err_11=None;c0_0=None;c0_1=None;c0_2=None;result_val_5=None;result_val_6=None;result_val_7=None;result_val_8=None;t_0=None;t_1=None;t_2=None;result_val_IV_1=None;result_val_IV_2=None;result_val_IV_3=None;result_val_IV_4=None;result_c1_val_IV_0=None;result_c1_val_IV_1=None;result_c1_val_IV_2=None;z_0=None;z_1=None;result_c2_0=None;result_c2_1=None;result_c2_val_IV_0=None;result_c2_val_IV_1=None;result_c1_0=None;result_c1_1=None;result_c1_2=None;px11_0=None;px11_1=None;

    if x_8<0.0:
        print("domain error") 
    elif x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON:
        z_0=pow(x_8,1.0/3.0) 
        cf_0=1-8.43812762813205e-01*z_0*z_0 
        result_val_5=2.14952824153447863671*z_0*cf_0 
        result_4.val=result_val_5 
        result_val_IV_1=result_4.val 
        result_err_5=GSL_DBL_EPSILON*result_val_IV_1 
        result_4.err=result_err_5 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif x_8<=4.0:
        c0_0=M_PI/M_SQRT3 
        px_0=pow(x_8,1.0/3.0) 
        px11_0=gsl_sf_pow_int(px_0,11) 
        t_0=x_8*x_8/8.0-1.0 
        result_c1_0=gsl_sf_result(0,0) 
        result_c2_0=gsl_sf_result(0,0) 
        cheb_eval_e(synchrotron1_cs,t_0,result_c1_0) 
        cheb_eval_e(synchrotron2_cs,t_0,result_c2_0) 
        result_c1_val_IV_0=result_c1_0.val 
        result_c2_val_IV_0=result_c2_0.val 
        result_val_6=px_0*result_c1_val_IV_0-px11_0*result_c2_val_IV_0-c0_0*x_8
        result_4.val=result_val_6 
        result_c1_err_IV_0=result_c1_0.err 
        result_c2_err_IV_0=result_c2_0.err 
        result_err_6=px_0*result_c1_err_IV_0+px11_0*result_c2_err_IV_0+c0_0*x_8*GSL_DBL_EPSILON 
        result_4.err=result_err_6 
        result_val_IV_2=result_4.val 
        result_err_7=result_4.err 
        result_err_8 = result_err_7+2.0*GSL_DBL_EPSILON*fabs(result_val_IV_2)
        result_4.err=result_err_8 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif x_8<-8.0*GSL_LOG_DBL_MIN/7.0:
        c0_1=0.2257913526447274323630976 
        t_1=(12.0-x_8)/(x_8+4.0)
        result_c1_1=gsl_sf_result(0,0) 
        cheb_eval_e(synchrotron1a_cs,t_1,result_c1_1) 
        result_c1_val_IV_1=result_c1_1.val 
        result_val_7=sqrt(x_8)*result_c1_val_IV_1*exp(c0_1-x_8) 
        result_4.val=result_val_7 
        result_val_IV_3=result_4.val 
        result_err_9=result_4.err 
        result_err_10=2.0*GSL_DBL_EPSILON*result_val_IV_3*(fabs(c0_1-x_8)+1.0) 
        result_4.err=result_err_10 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    else:
        print("underflow error") 
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,cf_0,None,None,None]
    cf_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c1_err_IV_0,None,None]
    result_c1_err_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c2_err_IV_0,None,None]
    result_c2_err_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,px_0,None,None]
    px_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,result_err_5,result_err_8,result_err_10,None]
    result_err_11= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,c0_0,c0_1,None]
    c0_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,result_val_5,result_val_6,result_val_7,None]
    result_val_8= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,t_0,t_1,None]
    t_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,result_val_IV_1,result_val_IV_2,result_val_IV_3,None]
    result_val_IV_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c1_val_IV_0,result_c1_val_IV_1,None]
    result_c1_val_IV_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,z_0,None,None,None]
    z_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c2_0,None,None]
    result_c2_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c2_val_IV_0,None,None]
    result_c2_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,result_c1_0,result_c1_1,None]
    result_c1_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_8<0.0,x_8<2.0*M_SQRT2*GSL_SQRT_DBL_EPSILON,x_8<=4.0,x_8<-8.0*GSL_LOG_DBL_MIN/7.0]
    phiNames = [None,None,px11_0,None,None]
    px11_1= phiIf(phiPreds, phiNames)

def gsl_sf_synchrotron_1(x):
    x_9 = x;
    result_5=None;

    result_5=gsl_sf_result(0.0,0.0) 
    return EVAL_RESULT(gsl_sf_synchrotron_1_e(x_9,result_5),result_5)



#generate python causal map
causal_map = {'cs_c_0_IV_0':['cs_0'],'cs_c_j_IV_1':['cs_c_j_IV_0'],'cs_c_j_IV_0':['cs_0','j_0'],'px11_0':['px_0'],'px11_1':['px11_0'],'d_0':[],'count_3':['count_0','count_1'],'count_2':['count_0','count_1'],'d_2':['d_0','d_1'],'count_1':['count_2'],'d_1':['y2_0','d_2','dd_2','cs_c_j_IV_0'],'count_0':[],'d_4':['y_0','d_3','dd_3','cs_c_0_IV_0'],'d_3':['d_0','d_1'],'dd_3':['dd_0','dd_1'],'dd_1':['temp_0'],'dd_2':['dd_0','dd_1'],'dd_0':[],'return_val_0':['x_7','n_7','result_3','result_3'],'t_0':['x_8','x_8'],'t_2':['t_0','t_1'],'t_1':['x_8','x_8'],'cs_c_cs_order_IV_0':['cs_0','cs_0'],'value_4':['value_1','value_3'],'value_5':['value_1','value_3'],'x_2':['x_1'],'value_2':['value_4','x_5'],'value_3':['value_2','value_4'],'x_4':['x_5','x_5'],'value_1':[],'x_3':['x_2','x_1'],'x_6':['x_3','x_4'],'x_5':['x_3','x_4'],'result_c2_err_IV_1':['result_c2_err_IV_0'],'temp_0':['d_2'],'result_c2_err_IV_0':['result_c2_0'],'temp_1':['temp_0'],'temp_2':['temp_0'],'cs_c_j_IV_2':['cs_c_j_IV_0'],'temp_3':['d_3'],'result_val_5':['z_0','cf_0'],'result_val_4':['value_5'],'result_val_7':['x_8','result_c1_val_IV_1','c0_1','x_8'],'result_val_6':['px_0','result_c1_val_IV_0','px11_0','result_c2_val_IV_0','c0_0','x_8'],'result_val_1':['u_0','n_2','u_0','u_0'],'result_val_0':['d_4'],'result_val_3':['result_val_2'],'result_val_2':['result_val_1'],'result_val_8':['result_val_5','result_val_6','result_val_7'],'result_3':[],'result_c2_val_IV_0':['result_c2_0'],'result_c2_val_IV_1':['result_c2_val_IV_0'],'result_5':[],'result_err_10':['result_val_IV_3','c0_1','x_8'],'result_err_11':['result_err_5','result_err_8','result_err_10'],'cf_0':['z_0','z_0'],'cf_1':['cf_0'],'y2_0':['y_0'],'px_0':['x_8'],'c0_2':['c0_0','c0_1'],'px_1':['px_0'],'cs_a_IV_0':['cs_0'],'n_2':['n_1'],'n_4':['n_5'],'n_3':['n_2','n_1'],'n_6':['n_3','n_4'],'n_5':['n_3','n_4'],'z_0':['x_8'],'cs_b_IV_0':['cs_0'],'z_1':['z_0'],'c0_1':[],'c0_0':[],'result_c1_0':[],'result_c1_1':[],'result_c1_2':['result_c1_0','result_c1_1'],'status_0':['fn_0'],'result_c1_err_IV_1':['result_c1_err_IV_0'],'result_val_IV_1':['result_4'],'result_val_IV_0':['result_0'],'result_c1_err_IV_0':['result_c1_0'],'e_1':['e_2','y2_0','temp_0','dd_2','cs_c_j_IV_0'],'e_0':[],'e_3':['e_0','e_1'],'e_2':['e_0','e_1'],'e_4':['e_3','y_0','temp_3','dd_3','cs_c_0_IV_0'],'u_1':['u_0'],'u_0':['x_1'],'u_2':['u_1'],'y_0':['x_0','cs_a_IV_0','cs_b_IV_0','cs_b_IV_0','cs_a_IV_0'],'result_err_9':['result_4'],'result_c1_val_IV_0':['result_c1_0'],'result_err_8':['result_err_7','result_val_IV_2'],'result_err_5':['result_val_IV_1'],'result_err_4':['count_3','value_5'],'result_c2_0':[],'result_err_7':['result_4'],'result_c1_val_IV_1':['result_c1_1'],'result_err_6':['px_0','result_c1_err_IV_0','px11_0','result_c2_err_IV_0','c0_0','x_8'],'result_c1_val_IV_2':['result_c1_val_IV_0','result_c1_val_IV_1'],'result_c2_1':['result_c2_0'],'result_err_1':[],'result_val_IV_3':['result_4'],'result_err_0':['e_4','cs_c_cs_order_IV_0'],'result_val_IV_2':['result_4'],'result_err_3':['result_err_2'],'result_err_2':['result_err_1'],'result_val_IV_4':['result_val_IV_1','result_val_IV_2','result_val_IV_3'],}

#added phi names
phi_names_set = {'dd_2','temp_1','d_2','e_2','cs_c_j_IV_1','dd_3','temp_2','d_3','e_3','cs_c_j_IV_2','result_val_2','u_1','result_err_2','result_val_3','u_2','x_3','result_err_3','n_3','x_5','count_2','value_4','n_5','value_3','x_6','count_3','value_5','n_6','cf_1','result_c1_err_IV_1','result_c2_err_IV_1','px_1','result_err_11','c0_2','result_val_8','t_2','result_val_IV_4','result_c1_val_IV_2','z_1','result_c2_1','result_c2_val_IV_1','result_c1_2','px11_1',}
#-------------end of program---------------------------

def record_locals(lo, i):
    for name in lo:
        if '_IV' in name:
            continue
        if isinstance(lo[name], numbers.Number) and name in causal_map:
            if name not in global_value_dict:
                columns = list(causal_map[name])
                columns.insert(0, name)
                global_value_dict[name] = pd.DataFrame(columns=columns)
            new_row = [np.float64(lo[name])]

            for pa in causal_map[name]:
                if isinstance(lo[pa], numbers.Number):
                    new_row.append(np.float64(lo[pa]))
                else:
                    new_row.append(lo[pa])
            global_value_dict[name].loc[i] = new_row

bad_dict = {}
global_value_dict = {}
arg1s = np.arange(0.1, 10.1, 0.01)

# add this for NUMFL
filename = './' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/'
os.makedirs(os.path.dirname(filename), exist_ok=True)
bugid = version_bug_dict[str(os.path.basename(sys.argv[0])[:-3])]
print("Bug is " + bugid)
test_counter = 0


probability = float(sys.argv[1])/100.0
for arg1 in arg1s:
    bad_outcome = gsl_sf_synchrotron_1(arg1)
    bad_dict[test_counter] = bad_outcome
    test_counter += 1

diff_dict = {index : 0.0 if bad_dict[index] == good_dict[index] else 1.0 for index in bad_dict }

print_run_ratio(bad_dict, good_dict)

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
    # add this for NUMFL/Coverage
    global bugindex
    counting = 1
    for name in global_value_dict:
        if name in phi_names_set:
            continue

        # df cleaning
        # df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='any')
        train_set = df
        # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set_X = train_set.drop(['outcome'], axis=1)
        train_set_Y = train_set['outcome']
        if model_to_use == 0:
            model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        if model_to_use == 1:
            model = Lasso(alpha=0.1)

        model.fit(train_set_X, train_set_Y)

        W = df.iloc[:, 0].to_frame()
        # add this for NUMFL
        if (name == bugid):
            bugindex = counting
        with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + str(counting) + ".txt",
                  "w") as f1:
            f1.write(str(train_set_X.to_csv(sep=' ', mode='a')))
            counting += 1
        quantiles = get_quantiled_tr(W)
        risk_list = predict_causal_risk_list(train_set_X, quantiles, model)
        max_risk = max(risk_list)
        min_risk = min(risk_list)
        row = [df.columns[0], max_risk - min_risk, risk_list.index(max_risk),
               risk_list.index(min_risk)]
        suspicious_df.loc[len(suspicious_df)] = row
    suspicious_df = suspicious_df.sort_values(by='max_risk_diff', ascending=False)
    return filter_phi_rows(suspicious_df, phi_names_set)


def filter_phi_rows(suspicious_df, phi_names_set):
    return suspicious_df[~suspicious_df['variable_name'].isin(phi_names_set)]


# 0-> random forest  1 -> lasso
result = suspicious_ranking(global_value_dict, 0)
pd.set_option("display.precision", 8)
print('*************Target variables in total: ', len(result),'*************')
print(result)
# add this to every program for NUMFL/Coverage
if type(bad_dict[0]) is tuple:
    difference_dict = {index: abs(bad_dict[index][0] - good_dict[index][0]) for index in bad_dict}
else:
    difference_dict = {index: abs(bad_dict[index] - good_dict[index]) for index in bad_dict}
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + "out.txt", "w") as f1:
    for k, v in diff_dict.items():
        f1.write(str(k) + ' ' + str(int(v)) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + "result.txt", "w") as f2:
    for k, v in bad_dict.items():
        f2.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + "truth.txt", "w") as f3:
    for k, v in good_dict.items():
        f3.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + "diff.txt", "w") as f4:
    for k, v in difference_dict.items():
        f4.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/' + "info.txt", "w") as f5:
    f5.write(str(len(result)) + '\n')
    f5.write(str(bugindex))
with open(os.path.basename(__file__)[:-3] + "-" + sys.argv[1] + "-Trial" + sys.argv[2] + ".txt", "w") as f:
    f.write('*************Target variables in total: ' + str(len(result)) + '*************\n')
    bad_runs, good_runs = get_run_ratio(bad_dict, good_dict)
    f.write("Number of Fault Insertions: " + str(insertion_count) + "\n")
    f.write("Number of Faulty Executions: " + str(bad_runs) + "\n")
    f.write(str(result.to_csv()))
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

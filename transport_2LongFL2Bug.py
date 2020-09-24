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
from helpers import *
import sys

from transport_2Long import good_dict
os.system('python transport_2Long.py')

from phi import *
dd_0=None;dd_2=None;dd_1=None;dd_3=None;reason_0=None;fn_0=None;result_c_val_0=None;result_c_val_1=None;cs_c_j_1=None;cs_c_j_0=None;cs_c_j_2=None;numexp_0=None;numexp_1=None;numexp_2=None;numexp_3=None;result_c_err_0=None;result_c_err_1=None;result_0=None;result_1=None;result_2=None;result_3=None;result_val_0=None;result_val_1=None;result_val_2=None;result_val_3=None;result_val_4=None;result_val_5=None;result_val_6=None;result_val_7=None;result_val_8=None;result_val_9=None;result_val_10=None;result_val_11=None;result_val_12=None;result_val_13=None;sumexp_0=None;sumexp_3=None;sumexp_1=None;sumexp_2=None;sumexp_4=None;sumexp_5=None;sumexp_6=None;sumexp_7=None;gsl_errno_0=None;y2_0=None;cs_c_cs_order_0=None;value_0=None;order_0=None;order_1=None;xk1_4=None;xk1_0=None;xk1_2=None;xk1_1=None;xk1_3=None;xk1_5=None;val_0=None;a_0=None;b_0=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;c_0=None;err_0=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;j_0=None;j_1=None;k_0=None;result_err_0=None;result_err_1=None;result_err_2=None;result_err_3=None;result_err_4=None;result_err_5=None;result_err_6=None;result_err_7=None;result_err_8=None;result_err_9=None;result_err_10=None;result_err_11=None;result_err_12=None;result_err_13=None;result_err_14=None;et_0=None;et_1=None;et_2=None;et_3=None;et_4=None;et_5=None;et_6=None;cs_0=None;sum2_4=None;sum2_0=None;sum2_2=None;sum2_1=None;sum2_3=None;sum2_5=None;xk_1=None;xk_0=None;xk_2=None;cs_c_0_0=None;t_0=None;t_1=None;t_2=None;t_3=None;t_4=None;t_5=None;x_0=None;x_1=None;x_2=None;x_3=None;y_0=None;rk_0=None;rk_2=None;rk_1=None;rk_3=None;val_infinity_0=None;order_sp_0=None;cs_a_0=None;cs_b_0=None;result_c_0=None;result_c_1=None;status_0=None

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
GSL_LOG_DBL_EPSILON=(-3.6043653389117154e+01) 
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

def GSL_ERROR_VAL(reason,gsl_errno,value):
    reason_0 = reason;gsl_errno_0 = gsl_errno;value_0 = value;
    
    lo = locals()
    record_locals(lo, test_counter)
    return 

def EVAL_RESULT(fn,result):
    fn_0 = fn;result_0 = result;
    status_0=None;

    status_0=fn_0 
    if status_0!=GSL_SUCCESS:
        GSL_ERROR_VAL(fn_0,status_0,result_0.val)
    lo = locals()
    record_locals(lo, test_counter) 
    return result_0.val

transport2_data=[1.671760446434538503,-0.147735359946794490,0.148213819946936338e-01,-0.14195330326305613e-02,0.1306541324415708e-03,-0.117155795867579e-04,0.10333498445756e-05,-0.901911304223e-07,0.78177169833e-08,-0.6744565684e-09,0.579946394e-10,-0.49747619e-11,0.425961e-12,-0.36422e-13,0.3111e-14,-0.265e-15,0.23e-16,-0.19e-17] 
transport2_cs=cheb_series(transport2_data,17,-1,1,9) 
def cheb_eval_e(cs,x,result):
    cs_0 = cs;x_0 = x;result_1 = result;
    dd_0=None;dd_2=None;dd_1=None;dd_3=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;result_err_0=None;cs_c_j_1=None;cs_c_j_0=None;cs_c_j_2=None;result_val_0=None;cs_c_0_0=None;y_0=None;y2_0=None;cs_c_cs_order_0=None;cs_a_0=None;cs_b_0=None;

    d_0=0.0 
    dd_0=0.0 
    cs_a_0=cs_0.a 
    cs_b_0=cs_0.b 
    y_0=(2.0*x_0-cs_a_0-cs_b_0)/(cs_b_0-cs_a_0) 
    y2_0=2.0*y_0 
    e_0=0.0 
    phi0 = Phi()
    for j_0 in range(cs_0.order,0,-1):
        phi0.set()
        dd_2 = phi0.phiEntry(dd_0,dd_1)
        temp_1 = phi0.phiEntry(None,temp_0)
        d_2 = phi0.phiEntry(d_0,d_1)
        e_2 = phi0.phiEntry(e_0,e_1)
        cs_c_j_1 = phi0.phiEntry(None,cs_c_j_0)

        temp_0=d_2 
        cs_c_j_0=cs_0.c[j_0] 
        d_1=y2_0*d_2-dd_2+cs_c_j_0 
        e_1 = e_2+fabs(y2_0*temp_0)+fabs(dd_2)+fabs(cs_c_j_0)
        dd_1=temp_0 
    dd_3 = phi0.phiExit(dd_0,dd_1)
    temp_2 = phi0.phiExit(None,temp_0)
    d_3 = phi0.phiExit(d_0,d_1)
    e_3 = phi0.phiExit(e_0,e_1)
    cs_c_j_2 = phi0.phiExit(None,cs_c_j_0)
    temp_3=d_3 
    cs_c_0_0=cs_0.c[0] 
    d_4=y_0*d_3-dd_3+0.5*cs_c_0_0 
    e_4 = e_3+fabs(y_0*temp_3)+fabs(dd_3)+0.5*fabs(cs_c_0_0)
    result_val_0=d_4 
    result_1.val=result_val_0 
    cs_c_cs_order_0=cs_0.c[cs_0.order] 
    result_err_0=GSL_DBL_EPSILON*e_4+fabs(cs_c_cs_order_0) 
    result_1.err=result_err_0 
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS

def transport_sumexp(numexp,order,t,x):
    numexp_0 = numexp;order_1 = order;t_0 = t;x_1 = x;
    sum2_4=None;sum2_0=None;sum2_2=None;sum2_1=None;sum2_3=None;sum2_5=None;sumexp_0=None;sumexp_3=None;sumexp_1=None;sumexp_2=None;sumexp_4=None;xk_1=None;xk_0=None;xk_2=None;rk_0=None;rk_2=None;rk_1=None;rk_3=None;xk1_4=None;xk1_0=None;xk1_2=None;xk1_1=None;xk1_3=None;xk1_5=None;

    rk_0=numexp_0 
    sumexp_0=0.0 
    phi0 = Phi()
    for k_0 in range(1,numexp_0+1):
        phi0.set()
        sum2_4 = phi0.phiEntry(None,sum2_3)
        xk_1 = phi0.phiEntry(None,xk_0)
        sumexp_3 = phi0.phiEntry(sumexp_0,sumexp_2)
        rk_2 = phi0.phiEntry(rk_0,rk_1)
        xk1_4 = phi0.phiEntry(None,xk1_3)

        sum2_0=1.0 
        xk_0=1.0/(rk_2*x_1) 
        xk1_0=1.0 
        phi1 = Phi()
        for j_1 in range(1,order_1+1):
            phi1.set()
            sum2_2 = phi1.phiEntry(sum2_0,sum2_1)
            xk1_2 = phi1.phiEntry(xk1_0,xk1_1)
            sum2_1=(sum2_2*xk1_2*xk_0+1.0 )
            xk1_1 = xk1_2+1.0
        sum2_3 = phi1.phiExit(sum2_0,sum2_1)
        xk1_3 = phi1.phiExit(xk1_0,xk1_1)
        sumexp_1 = sumexp_3*t_0
        sumexp_2 = sumexp_1+sum2_3 
        rk_1 = rk_2-1.0
    sum2_5 = phi0.phiExit(None,sum2_3)
    xk_2 = phi0.phiExit(None,xk_0)
    sumexp_4 = phi0.phiExit(sumexp_0,sumexp_2)
    rk_3 = phi0.phiExit(rk_0,rk_1)
    xk1_5 = phi0.phiExit(None,xk1_3)
    lo = locals()
    record_locals(lo, test_counter)
    return sumexp_4

def gsl_sf_transport_2_e(x,result):
    x_2 = x;result_2 = result;
    result_val_1=None;result_val_2=None;result_val_3=None;result_val_4=None;result_val_5=None;result_val_6=None;result_val_7=None;result_val_8=None;result_val_9=None;result_val_10=None;result_val_11=None;result_val_12=None;result_val_13=None;sumexp_5=None;sumexp_6=None;sumexp_7=None;t_1=None;t_2=None;t_3=None;t_4=None;t_5=None;val_infinity_0=None;result_c_val_0=None;result_c_val_1=None;result_err_1=None;result_err_2=None;result_err_3=None;result_err_4=None;result_err_5=None;result_err_6=None;result_err_7=None;result_err_8=None;result_err_9=None;result_err_10=None;result_err_11=None;result_err_12=None;result_err_13=None;result_err_14=None;result_c_0=None;result_c_1=None;result_c_err_0=None;result_c_err_1=None;numexp_1=None;numexp_2=None;numexp_3=None;et_0=None;et_1=None;et_2=None;et_3=None;et_4=None;et_5=None;et_6=None;

    val_infinity_0=3.289868133696452873 
    if x_2<0.0:
        print("domian error") 
    elif x_2<3.0*GSL_SQRT_DBL_EPSILON:
        result_val_1=x_2 
        result_2.val=result_val_1 
        result_err_1=GSL_DBL_EPSILON*fabs(x_2)+x_2*x_2/2.0 
        result_2.err=result_err_1 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif x_2<=4.0:
        t_1=(x_2*x_2/8.0-0.5)-0.5 
        result_c_0=gsl_sf_result(0.0,0.0) 
        cheb_eval_e(transport2_cs,t_1,result_c_0) 
        result_c_val_0=result_c_0.val 
        result_val_2=x_2*result_c_val_0
        result_2.val=result_val_2 
        result_c_err_0=result_c_0.err 
        result_err_2=x_2*result_c_err_0 
        result_2.err=result_err_2 
        result_val_3=result_2.val 
        result_err_3=result_2.err 
        result_err_4 = result_err_3+2.0*GSL_DBL_EPSILON*fabs(result_2.val)
        result_2.err=result_err_4 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif x_2<-GSL_LOG_DBL_EPSILON:
        numexp_1=(int)((-GSL_LOG_DBL_EPSILON)/x_2)+1 
        sumexp_5=transport_sumexp(numexp_1,2,exp(-x_2),x_2) 
        t_2=2.0*log(x_2)-x_2+log(sumexp_5) 
        if t_2<GSL_LOG_DBL_EPSILON:
            result_val_4=val_infinity_0 
            result_2.val=result_val_4 
            result_err_5=2.0*GSL_DBL_EPSILON*val_infinity_0 
            result_2.err=result_err_5 
        else:
            et_0=exp(t_2) 
            result_val_5=val_infinity_0-et_0 + bug
            result_2.val=result_val_5 
            result_err_6=2.0*GSL_DBL_EPSILON*(val_infinity_0+fabs(t_2)*et_0) 
            result_2.err=result_err_6 
        phiPreds = [t_2<GSL_LOG_DBL_EPSILON]
        phiNames = [result_val_4,result_val_5]
        result_val_6= phiIf(phiPreds, phiNames)
        phiPreds = [t_2<GSL_LOG_DBL_EPSILON]
        phiNames = [result_err_5,result_err_6]
        result_err_7= phiIf(phiPreds, phiNames)
        phiPreds = [t_2<GSL_LOG_DBL_EPSILON]
        phiNames = [None,et_0]
        et_1= phiIf(phiPreds, phiNames)
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    elif x_2<2.0/GSL_DBL_EPSILON:
        numexp_2=1 
        sumexp_6=transport_sumexp(numexp_2,2,1.0,x_2) 
        t_3=2.0*log(x_2)-x_2+log(sumexp_6) 
        if t_3<GSL_LOG_DBL_EPSILON:
            result_val_7=val_infinity_0 
            result_2.val=result_val_7 
            result_err_8=2.0*GSL_DBL_EPSILON*val_infinity_0 
            result_2.err=result_err_8 
        else:
            et_2=exp(t_3) 
            result_val_8=val_infinity_0-et_2 
            result_2.val=result_val_8 
            result_err_9=2.0*GSL_DBL_EPSILON*(val_infinity_0+(fabs(t_3)+1.0)*et_2) 
            result_2.err=result_err_9 
        phiPreds = [t_3<GSL_LOG_DBL_EPSILON]
        phiNames = [result_val_7,result_val_8]
        result_val_9= phiIf(phiPreds, phiNames)
        phiPreds = [t_3<GSL_LOG_DBL_EPSILON]
        phiNames = [result_err_8,result_err_9]
        result_err_10= phiIf(phiPreds, phiNames)
        phiPreds = [t_3<GSL_LOG_DBL_EPSILON]
        phiNames = [None,et_2]
        et_3= phiIf(phiPreds, phiNames)
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    else:
        t_4=2.0*log(x_2)-x_2 
        if t_4<GSL_LOG_DBL_EPSILON:
            result_val_10=val_infinity_0 
            result_2.val=result_val_10 
            result_err_11=2.0*GSL_DBL_EPSILON*val_infinity_0 
            result_2.err=result_err_11 
        else:
            et_4=exp(t_4) 
            result_val_11=val_infinity_0-et_4 
            result_2.val=result_val_11 
            result_err_12=2.0*GSL_DBL_EPSILON*(val_infinity_0+(fabs(t_4)+1.0)*et_4) 
            result_2.err=result_err_12 
        phiPreds = [t_4<GSL_LOG_DBL_EPSILON]
        phiNames = [result_val_10,result_val_11]
        result_val_12= phiIf(phiPreds, phiNames)
        phiPreds = [t_4<GSL_LOG_DBL_EPSILON]
        phiNames = [result_err_11,result_err_12]
        result_err_13= phiIf(phiPreds, phiNames)
        phiPreds = [t_4<GSL_LOG_DBL_EPSILON]
        phiNames = [None,et_4]
        et_5= phiIf(phiPreds, phiNames)
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,result_val_1,result_val_3,result_val_6,result_val_9,result_val_12]
    result_val_13= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,None,sumexp_5,sumexp_6,None]
    sumexp_7= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,t_1,t_2,t_3,t_4]
    t_5= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,result_c_val_0,None,None,None]
    result_c_val_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,result_err_1,result_err_4,result_err_7,result_err_10,result_err_13]
    result_err_14= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,result_c_0,None,None,None]
    result_c_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,result_c_err_0,None,None,None]
    result_c_err_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,None,numexp_1,numexp_2,None]
    numexp_3= phiIf(phiPreds, phiNames)
    phiPreds = [x_2<0.0,x_2<3.0*GSL_SQRT_DBL_EPSILON,x_2<=4.0,x_2<-GSL_LOG_DBL_EPSILON,x_2<2.0/GSL_DBL_EPSILON]
    phiNames = [None,None,None,et_1,et_3,et_5]
    et_6= phiIf(phiPreds, phiNames)

def gsl_sf_transport_2(x):
    x_3 = x;
    result_3=None;

    result_3=gsl_sf_result(0.0,0.0) 
    lo = locals()
    record_locals(lo, test_counter)
    return EVAL_RESULT(gsl_sf_transport_2_e(x_3,result_3),result_3)



#generate python causal map
causal_map = dict(sumexp_4=['sum2_3','xk_0','sumexp_0','sumexp_2','rk_0','rk_1','xk1_3'],sumexp_3=['sumexp_0','sumexp_2'],sumexp_2=['sumexp_1','sum2_3'],xk1_0=[],sumexp_1=['sumexp_3','t_0'],sumexp_0=[],numexp_2=[],numexp_3=['numexp_1','numexp_2'],numexp_1=['x_2'],cs_c_j_1=['cs_c_j_0'],cs_c_j_0=['cs_0','j_0'],sumexp_7=['sumexp_5','sumexp_6'],sumexp_6=['numexp_2','x_2'],cs_c_j_2=['dd_0','dd_1','temp_0','d_0','d_1','e_0','e_1','cs_c_j_0'],sumexp_5=['numexp_1','x_2','x_2'],d_0=[],d_2=['d_0','d_1'],cs_c_cs_order_0=['cs_0','cs_0'],d_1=['y2_0','d_2','dd_2','cs_c_j_0'],d_4=['y_0','d_3','dd_3','cs_c_0_0'],xk1_5=['sum2_3','xk_0','sumexp_0','sumexp_2','rk_0','rk_1','xk1_3'],d_3=['dd_0','dd_1','temp_0','d_0','d_1','e_0','e_1','cs_c_j_0'],dd_3=['dd_0','dd_1','temp_0','d_0','d_1','e_0','e_1','cs_c_j_0'],dd_1=['temp_0'],xk1_1=['xk1_2'],dd_2=['dd_0','dd_1'],xk1_2=['xk1_0','xk1_1'],xk1_3=['sum2_0','sum2_1','xk1_0','xk1_1'],dd_0=[],xk1_4=['xk1_3'],t_2=['x_2','x_2','sumexp_5'],t_1=['x_2','x_2'],t_4=['x_2','x_2'],t_3=['x_2','x_2','sumexp_6'],t_5=['t_1','t_2','t_3','t_4'],temp_0=['d_2'],temp_1=['temp_0'],temp_2=['dd_0','dd_1','temp_0','d_0','d_1','e_0','e_1','cs_c_j_0'],cs_b_0=['cs_0'],temp_3=['d_3'],result_val_5=['val_infinity_0','et_0'],result_val_4=['val_infinity_0'],result_val_7=['val_infinity_0'],result_val_6=['result_val_4','result_val_5'],result_val_1=['x_2'],result_val_0=['d_4'],result_val_3=['result_2'],result_val_2=['x_2','result_c_val_0'],cs_c_0_0=['cs_0'],result_val_9=['result_val_7','result_val_8'],result_val_8=['val_infinity_0','et_2'],result_3=[],result_err_10=['result_err_8','result_err_9'],result_err_11=['val_infinity_0'],result_err_12=['val_infinity_0','t_4','et_4'],result_err_13=['result_err_11','result_err_12'],result_err_14=['result_err_1','result_err_4','result_err_7','result_err_10','result_err_13'],y2_0=['y_0'],et_6=['et_1','et_3','et_5'],et_5=['et_4'],et_4=['t_4'],et_3=['et_2'],et_2=['t_3'],et_1=['et_0'],et_0=['t_2'],rk_3=['sum2_3','xk_0','sumexp_0','sumexp_2','rk_0','rk_1','xk1_3'],result_c_err_0=['result_c_0'],result_val_11=['val_infinity_0','et_4'],rk_2=['rk_0','rk_1'],result_val_12=['result_val_10','result_val_11'],result_c_err_1=['result_c_err_0'],result_val_13=['result_val_1','result_val_3','result_val_6','result_val_9','result_val_12'],rk_1=['rk_2'],rk_0=['numexp_0'],val_infinity_0=[],result_val_10=['val_infinity_0'],result_c_val_1=['result_c_val_0'],result_c_val_0=['result_c_0'],status_0=['fn_0'],e_1=['e_2','y2_0','temp_0','dd_2','cs_c_j_0'],e_0=[],e_3=['dd_0','dd_1','temp_0','d_0','d_1','e_0','e_1','cs_c_j_0'],sum2_0=[],e_2=['e_0','e_1'],sum2_1=['sum2_2','xk1_2','xk_0'],sum2_2=['sum2_0','sum2_1'],e_4=['e_3','y_0','temp_3','dd_3','cs_c_0_0'],sum2_3=['sum2_0','sum2_1','xk1_0','xk1_1'],sum2_4=['sum2_3'],xk_0=['rk_2','x_1'],result_c_1=['result_c_0'],xk_1=['xk_0'],sum2_5=['sum2_3','xk_0','sumexp_0','sumexp_2','rk_0','rk_1','xk1_3'],result_c_0=[],xk_2=['sum2_3','xk_0','sumexp_0','sumexp_2','rk_0','rk_1','xk1_3'],y_0=['x_0','cs_a_0','cs_b_0','cs_b_0','cs_a_0'],result_err_9=['val_infinity_0','t_3','et_2'],result_err_8=['val_infinity_0'],cs_a_0=['cs_0'],result_err_5=['val_infinity_0'],result_err_4=['result_err_3','result_2'],result_err_7=['result_err_5','result_err_6'],result_err_6=['val_infinity_0','t_2','et_0'],result_err_1=['x_2','x_2','x_2'],result_err_0=['e_4','cs_c_cs_order_0'],result_err_3=['result_2'],result_err_2=['x_2','result_c_err_0'],)

#added phi names
phi_names_set = {'dd_2','temp_1','d_2','e_2','cs_c_j_1','dd_3','temp_2','d_3','e_3','cs_c_j_2','sum2_4','xk_1','sumexp_3','rk_2','xk1_4','sum2_2','xk1_2','sum2_3','xk1_3','sum2_5','xk_2','sumexp_4','rk_3','xk1_5','result_val_6','result_err_7','et_1','result_val_9','result_err_10','et_3','result_val_12','result_err_13','et_5','result_val_13','sumexp_7','t_5','result_c_val_1','result_err_14','result_c_1','result_c_err_1','numexp_3','et_6',}
#------end of program---------------------------
def record_locals(lo, i):
    for name in lo:
        if '_IV' in name:
            continue
        if isinstance(lo[name], numbers.Number) and name in causal_map:
            if name not in global_value_dict:
                columns = causal_map[name].copy()
                columns.insert(0, name)
                global_value_dict[name] = pd.DataFrame(columns=columns)
            new_row = [np.float64(lo[name])]

            for pa in causal_map[name]:
                if isinstance(lo[pa], numbers.Number):
                    new_row.append(np.float64(lo[pa]))
                else:
                    new_row.append(lo[pa])
            global_value_dict[name].loc[i] = new_row

def fluky(good_val, bad_val, p):
        r = random.random()
        if r <= p:
            return bad_val
        else:
            return good_val


bad_dict = {}
global_value_dict = {}
arg1s = np.arange(0.0, 10, 0.01)
test_counter = 0

# add this for NUMFL
filename = './' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL-Multi/Data1/'
os.makedirs(os.path.dirname(filename), exist_ok=True)
bugid = version_bug_dict_multi[str(os.path.basename(sys.argv[0])[:-3])]
bugindex=[]
bug = 0

for arg1 in arg1s:
    bug = fluky(0,3.08 , 0.95)
    
    bad_outcome = gsl_sf_transport_2(arg1)
    bad_dict[test_counter] = bad_outcome
    test_counter += 1


diff_dict = {index : 0.0 if bad_dict[index] == good_dict[index] else 1.0 for index in bad_dict }


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
        if name in bugid:
            bugindex.append(counting)
        with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + str(counting) + ".txt",
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
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + "out.txt", "w") as f1:
    for k, v in diff_dict.items():
        f1.write(str(k) + ' ' + str(int(v)) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + "result.txt", "w") as f2:
    for k, v in bad_dict.items():
        f2.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + "truth.txt", "w") as f3:
    for k, v in good_dict.items():
        f3.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + "diff.txt", "w") as f4:
    for k, v in difference_dict.items():
        f4.write(str(v) + '\n')
with open('./' + os.path.basename(__file__)[:-3] + sys.argv[1] + '--NUMFL-Multi/Data1/' + "info.txt", "w") as f5:
    f5.write(str(len(result)) + '\n')
    f5.write(str(bugindex[0]) + '\n')
    f5.write(str(bugindex[1]) + '\n')
    f5.write(str(bugindex[2]) + '\n')
with open(os.path.basename(__file__)[:-3] + "-" + sys.argv[1] + "-Trial" + sys.argv[2] + ".txt", "w") as f:
    f.write('*************Target variables in total: ' + str(len(result)) + '*************\n')
    bad_runs, good_runs = get_run_ratio(bad_dict, good_dict)

    f.write("Number of Faulty Executions: " + str(bad_runs) + "\n")
    f.write(str(result.to_csv()))
f.close()
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

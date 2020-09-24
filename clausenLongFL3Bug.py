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
from collections import namedtuple


from clausenLong import good_dict
os.system('python clausenLong.py') 


from phi import *


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
class gsl_sf_result:
    def __init__(self,val,err):
        val_0 = val;err_0 = err;
        

        self.val=val_0 
        self.err=err_0 

def GSL_ERROR(reason,gsl_errno):
    reason_0 = reason;gsl_errno_0 = gsl_errno;
    

    return 

def GSL_ERROR_VAL(reason,gsl_errno,value):
    reason_1 = reason;gsl_errno_1 = gsl_errno;value_0 = value;
    

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

class cheb_series:
    def __init__(self,c,order,a,b,order_sp):
        c_0 = c;order_0 = order;a_0 = a;b_0 = b;order_sp_0 = order_sp;
        

        self.c=c_0 
        self.order=order_0 
        self.a=a_0 
        self.b=b_0 
        self.order_sp=order_sp_0 

aclaus_data=[2.142694363766688447e+00,0.723324281221257925e-01,0.101642475021151164e-02,0.3245250328531645e-04,0.133315187571472e-05,0.6213240591653e-07,0.313004135337e-08,0.16635723056e-09,0.919659293e-11,0.52400462e-12,0.3058040e-13,0.18197e-14,0.1100e-15,0.68e-17,0.4e-18] 
aclaus_cs=cheb_series(aclaus_data,14,-1,1,8) 
def cheb_eval_e(cs,x,result):
    cs_0 = cs;x_0 = x;result_1 = result;
    dd_0=None;dd_2=None;dd_1=None;dd_3=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;cs_c_j_IV_1=None;cs_c_j_IV_0=None;cs_c_j_IV_2=None;cs_c_cs_order_IV_0=None;cs_c_0_IV_0=None;result_err_0=None;cs_a_IV_0=None;result_val_IV_0=None;cs_b_IV_0=None;y_0=None;y2_0=None;

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
    d_4=y_0*d_3-dd_3+0.5*cs_c_0_IV_0 + bug3
    e_4 = e_3+fabs(y_0*temp_3)+fabs(dd_3)+0.5*fabs(cs_c_0_IV_0)
    result_val_IV_0=d_4 
    result_1.val=result_val_IV_0 
    cs_c_cs_order_IV_0=cs_0.c[cs_0.order] 
    result_err_0=GSL_DBL_EPSILON*e_4+fabs(cs_c_cs_order_IV_0) 
    result_1.err=result_err_0 
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS

def gsl_sf_angle_restrict_pos_err_e(theta,result):
    theta_0 = theta;result_2 = result;
    P1_0=None;P2_0=None;TwoPi_0=None;r_0=None;r_1=None;r_2=None;r_3=None;P3_0=None;result_val_IV_1=None;result_val_IV_2=None;result_val_IV_3=None;delta_0=None;delta_1=None;y_1=None;result_err_1=None;result_err_2=None;result_err_3=None;

    P1_0=4*7.85398125648498535156e-01 
    P2_0=4*3.77489470793079817668e-08 
    P3_0=4*2.69515142907905952645e-15 
    TwoPi_0=2*(P1_0+P2_0+P3_0) 
    y_1=2*floor(theta_0/TwoPi_0) + bug2
    r_0=((theta_0-y_1*P1_0)-y_1*P2_0)-y_1*P3_0 
    if r_0>TwoPi_0:
        r_1=(((r_0-2*P1_0)-2*P2_0)-2*P3_0) 
    elif r_0<0:
        r_2=(((r_0+2*P1_0)+2*P2_0)+2*P3_0) 
    phiPreds = [r_0>TwoPi_0,r_0<0]
    phiNames = [r_1,r_2,r_0]
    r_3= phiIf(phiPreds, phiNames)
    result_2.val=r_3 
    if fabs(theta_0)>0.0625/GSL_DBL_EPSILON:
        result_2.val=float('nan') 
        result_2.err=fabs(result_2.val) 
        GSL_ERROR("error",GSL_ELOSS) 
    elif fabs(theta_0)>0.0625/GSL_SQRT_DBL_EPSILON:
        result_val_IV_1=result_2.val 
        result_err_1=GSL_DBL_EPSILON*fabs(result_val_IV_1-theta_0) 
        result_2.err=result_err_1 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    else:
        result_val_IV_2=result_2.val 
        delta_0=fabs(result_val_IV_2-theta_0) 
        result_err_2=2.0*GSL_DBL_EPSILON*(delta_0 if delta_0<M_PI else M_PI) 
        result_2.err=result_err_2 
        lo = locals()
        record_locals(lo, test_counter)
        return GSL_SUCCESS
    phiPreds = [fabs(theta_0)>0.0625/GSL_DBL_EPSILON,fabs(theta_0)>0.0625/GSL_SQRT_DBL_EPSILON]
    phiNames = [None,result_val_IV_1,result_val_IV_2]
    result_val_IV_3= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(theta_0)>0.0625/GSL_DBL_EPSILON,fabs(theta_0)>0.0625/GSL_SQRT_DBL_EPSILON]
    phiNames = [None,None,delta_0]
    delta_1= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(theta_0)>0.0625/GSL_DBL_EPSILON,fabs(theta_0)>0.0625/GSL_SQRT_DBL_EPSILON]
    phiNames = [None,result_err_1,result_err_2]
    result_err_3= phiIf(phiPreds, phiNames)

def gsl_sf_angle_restrict_pos_e(theta):
    theta_1 = theta;
    r_4=None;stat_0=None;r_val_IV_0=None;theta_2=None;

    r_4=gsl_sf_result(0.0,0.0) 
    stat_0=gsl_sf_angle_restrict_pos_err_e(theta_1,r_4) 
    r_val_IV_0=r_4.val
    theta_2=r_val_IV_0 
    lo = locals()
    record_locals(lo, test_counter)
    return stat_0,theta_2

def gsl_sf_clausen_e(x,result):
    x_1 = x;result_3 = result;
    p0_0=None;p0_1=None;p1_0=None;p1_1=None;result_c_err_IV_0=None;result_c_err_IV_1=None;status_red_0=None;result_err_4=None;result_err_5=None;result_err_6=None;result_err_7=None;result_val_0=None;result_val_1=None;result_val_2=None;result_val_3=None;result_val_4=None;result_val_5=None;t_0=None;t_1=None;x_2=None;x_3=None;x_4=None;x_5=None;x_6=None;result_c_val_IV_0=None;result_c_val_IV_1=None;sgn_0=None;sgn_1=None;sgn_2=None;sgn_3=None;sgn_4=None;x_cut_0=None;result_c_0=None;result_c_1=None;

    x_cut_0=M_PI*GSL_SQRT_DBL_EPSILON 
    sgn_0=1.0 
    if x_1<0.0:
        x_2=-x_1 
        sgn_1=-1.0 
    phiPreds = [x_1<0.0]
    phiNames = [x_2,x_1]
    x_3= phiIf(phiPreds, phiNames)
    phiPreds = [x_1<0.0]
    phiNames = [sgn_1,sgn_0]
    sgn_2= phiIf(phiPreds, phiNames)
    status_red_0,x_4=gsl_sf_angle_restrict_pos_e(x_3) 
    if x_4>M_PI:
        p0_0=6.28125 
        p1_0=0.19353071795864769253e-02 
        x_5=(p0_0-x_4)+p1_0 
        sgn_3=-sgn_2 
    phiPreds = [x_4>M_PI]
    phiNames = [p0_0,None]
    p0_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_4>M_PI]
    phiNames = [p1_0,None]
    p1_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_4>M_PI]
    phiNames = [x_5,x_4]
    x_6= phiIf(phiPreds, phiNames)
    phiPreds = [x_4>M_PI]
    phiNames = [sgn_3,sgn_2]
    sgn_4= phiIf(phiPreds, phiNames)
    if x_6==0.0:
        result_val_0=0.0 
        result_3.val=result_val_0 
        result_err_4=0.0 
        result_3.err=result_err_4 
    elif x_6<x_cut_0:
        result_val_1=x_6*(1.0-log(x_6)) 
        result_3.val=result_val_1 
        result_err_5=x_6*GSL_DBL_EPSILON 
        result_3.err=result_err_5 
    else:
        t_0=2.0*(x_6*x_6/(M_PI*M_PI)-0.5) 
        result_c_0=gsl_sf_result(0.0,0.0) 
        cheb_eval_e(aclaus_cs,t_0,result_c_0) 
        result_c_val_IV_0=result_c_0.val 
        result_val_2=x_6*(result_c_val_IV_0-log(x_6)) + bug1
        result_3.val=result_val_2 
        result_c_err_IV_0=result_c_0.err 
        result_err_6=x_6*(result_c_err_IV_0+GSL_DBL_EPSILON) 
        result_3.err=result_err_6 
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [result_val_0,result_val_1,result_val_2]
    result_val_3= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [None,None,t_0]
    t_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [None,None,result_c_err_IV_0]
    result_c_err_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [None,None,result_c_val_IV_0]
    result_c_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [result_err_4,result_err_5,result_err_6]
    result_err_7= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0,x_6<x_cut_0]
    phiNames = [None,None,result_c_0]
    result_c_1= phiIf(phiPreds, phiNames)
    result_val_4=result_3.val 
    result_val_5 = result_val_4*sgn_4
    result_3.val=result_val_5 
    lo = locals()
    record_locals(lo, test_counter)
    return status_red_0

def gsl_sf_clausen(x):
    x_7 = x;
    result_4=None;

    result_4=gsl_sf_result(0.0,0.0) 
    lo = locals()
    record_locals(lo, test_counter)
    return EVAL_RESULT(gsl_sf_clausen_e(x_7,result_4),result_4)


#generate python causal map
causal_map = {'p1_1':['p1_0','x_4'],'p1_0':[],'d_0':[],'d_2':['d_0','d_1','j_0','cs_0'],'d_1':['y2_0','d_2','dd_2','cs_0','j_0'],'d_4':['y_0','d_3','dd_3','cs_0'],'d_3':['d_0','d_1','j_0','cs_0'],'delta_0':['result_2','theta_0'],'dd_3':['dd_0','dd_1','j_0','cs_0'],'delta_1':['delta_0','theta_0'],'dd_1':['temp_0'],'dd_2':['dd_0','dd_1','j_0','cs_0'],'dd_0':[],'r_0':['theta_0','y_1','P1_0','y_1','P2_0','y_1','P3_0'],'r_2':['r_0','P1_0','P2_0','P3_0'],'t_0':['x_6','x_6'],'r_1':['r_0','P1_0','P2_0','P3_0'],'r_4':[],'r_3':['r_1','r_2','r_0','r_0','TwoPi_0'],'t_1':['t_0','x_6','x_cut_0'],'x_2':['x_1'],'x_4':['x_3'],'x_3':['x_2','x_1','x_1'],'x_6':['x_5','x_4','x_4'],'x_5':['p0_0','x_4','p1_0'],'temp_0':['d_2'],'temp_1':['temp_0','j_0','cs_0'],'temp_2':['temp_0','j_0','cs_0'],'P2_0':[],'status_0':['fn_0'],'temp_3':['d_3'],'theta_2':['r_4'],'p0_1':['p0_0','x_4'],'p0_0':[],'stat_0':['theta_1','r_4'],'sgn_2':['sgn_1','sgn_0','x_1'],'sgn_1':[],'sgn_4':['sgn_3','sgn_2','x_4'],'sgn_3':['sgn_2'],'sgn_0':[],'e_1':['e_2','y2_0','temp_0','dd_2','cs_0','j_0'],'e_0':[],'e_3':['e_0','e_1','j_0','cs_0'],'e_2':['e_0','e_1','j_0','cs_0'],'status_red_0':['x_3'],'e_4':['e_3','y_0','temp_3','dd_3','cs_0'],'result_c_1':['result_c_0','x_6','x_cut_0'],'result_c_0':[],'x_cut_0':[],'TwoPi_0':['P1_0','P2_0','P3_0'],'y_1':['theta_0','TwoPi_0'],'y_0':['x_0','cs_0','cs_0','cs_0','cs_0'],'result_4':[],'P3_0':[],'P1_0':[],'y2_0':['y_0'],}

#added phi names
phi_names_set = {'dd_2','temp_1','d_2','e_2','cs_c_j_IV_1','dd_3','temp_2','d_3','e_3','cs_c_j_IV_2','r_3','result_val_IV_3','delta_1','result_err_3','x_3','sgn_2','p0_1','p1_1','x_6','sgn_4','result_val_3','t_1','result_c_err_IV_1','result_c_val_IV_1','result_err_7','result_c_1',}

#--------------------end of progarm-----------

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
arg1s = np.arange(0, 10, 0.01)
test_counter = 0


bug1 = 0
bug2 = 0
bug3 = 0
for arg1 in arg1s:
    bug1 = fluky(0, 1.05, 0.5)
    bug2 = fluky(0, 1.05, 0.5)
    bug3 = fluky(0, 1.05, 0.5)
    bad_outcome = gsl_sf_clausen(arg1)
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
    for name in global_value_dict:
        if name in phi_names_set:
            continue

        #df cleaning
        #df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='any')
        train_set = df
        #train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set_X = train_set.drop(['outcome'], axis=1)
        train_set_Y = train_set['outcome']
        if model_to_use == 0:
            model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        if model_to_use == 1:
            model = Lasso(alpha=0.1)

        
        model.fit(train_set_X, train_set_Y)

        W = df.iloc[:, 0].to_frame()
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
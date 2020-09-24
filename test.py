import numpy as np
import pandas as pd
import numbers
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from phi import *
import random
import os
from math import *

def acosh(x):
    GSL_SQRT_DBL_EPSILON  = 1.4901161193847656e-08
    M_LN2  =    0.69314718055994530941723212146
    a = abs (x)
    s = -1 if x < 0 else 1
    result = 0
    if a > 1 / GSL_SQRT_DBL_EPSILON:
        result = s * (log (a) + M_LN2)
    elif a > 2:
        result = s * log (2 * a + 1 / (a + sqrt (a * a + 1)))
    elif a > GSL_SQRT_DBL_EPSILON:
        a2 = a * a
        result = s * log1p (a + a2 / (1 + sqrt (1 + a2)))

    else:
        result =  x
    return result
def acosh_bad(x):
    x_0 = x;
    result_0=None;result_1=None;result_2=None;result_3=None;result_4=None;result_5=None;a_0=None;a2_0=None;a2_1=None;s_0=None;M_LN2_0=None;SQRT_DBL_EPSILON_0=None;

    GSL_SQRT_DBL_EPSILON=1.4901161193847656e-08 
    M_LN2=0.69314718055994530941723212146 
    a_0=abs(x_0) 
    s_0=-1 if x_0<0 else 1 
    result_0=0 
    if a_0>1/GSL_SQRT_DBL_EPSILON:
        result_1=s_0*(log(a_0)+M_LN2) 
    elif a_0>2:
        result_2=s_0*log(2*a_0+1/(a_0+sqrt(a_0*a_0+1))) 
    elif a_0>GSL_SQRT_DBL_EPSILON:
        print(a_0 , type(a_0))
        if a_0 == 2.0:
            print(a_0, a2_0 , result_3)
        a2_0=a_0 + a_0
        result_3=s_0*log1p(a_0+a2_0/(1+sqrt(1+a2_0))) 
        
    else:
        result_4=x_0 
    phiPreds = [a_0>1/GSL_SQRT_DBL_EPSILON,a_0>2,a_0>GSL_SQRT_DBL_EPSILON]
    phiNames = [result_1,result_2,result_3,result_4]
    result_5= phiIf(phiPreds, phiNames)
    phiPreds = [a_0>1/GSL_SQRT_DBL_EPSILON,a_0>2,a_0>GSL_SQRT_DBL_EPSILON]
    phiNames = [None,None,a2_0,None]
    a2_1= phiIf(phiPreds, phiNames)
 
    lo = locals()
    record_locals(lo, test_counter) 
    return result_5


#generate python causal map
causal_map = dict(M_LN2_0=[],result_0=[],a_0=['x_0'],result_1=['s_0','a_0','M_LN2_0'],s_0=['x_0'],SQRT_DBL_EPSILON_0=[],result_2=['s_0','a_0','a_0','a_0','a_0'],result_3=['s_0','a_0','a2_0','a2_0'],a2_0=['a_0','a_0'],result_4=['x_0'],result_5=['result_1','result_2','result_3','result_4'],a2_1=['a2_0'],)
#causal_map2 = {M_LN2_0:[],result_0:[],a_0:['x_0'],result_1:['s_0','a_0','M_LN2_0'],s_0:['x_0'],SQRT_DBL_EPSILON_0:[],result_2:['s_0','a_0','a_0','a_0','a_0'],result_3:['s_0','a_0','a2_0','a2_0'],a2_0:['a_0','a_0'],result_4:['x_0'],result_5:['result_1','result_2','result_3','result_4'],a2_1:['a2_0']}

#added phi names
phi_names_set = {'result_5','a2_1',}


def record_locals(lo, i):
    for name in lo:
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
good_dict = {}
global_value_dict = {}
arg0s = np.arange(0.1, 7, 0.01)
arg1s = np.arange(0.1, 7, 0.01)
test_counter = 0
for arg1 in arg1s:
    good_outcome = acosh(arg1)
    good_dict[test_counter] = good_outcome
    test_counter += 1


test_counter = 0
for arg1 in arg1s:
    bad_outcome = acosh_bad(arg1)
    bad_dict[test_counter] = bad_outcome
    test_counter += 1

diff_dict = {index : 0.0 if bad_dict[index] + 0.001 > good_dict[index] 
                            and bad_dict[index] - 0.001 < good_dict[index]
                        else 1.0 for index in bad_dict }


for key in global_value_dict:
    rows = global_value_dict[key].index
    outcome_list = [diff_dict[i] for i in rows]
    global_value_dict[key]['outcome'] = outcome_list


def get_quantiled_tr(W):
    # 10 quantiles from 0.05 to 0.95
    quantile_list = []
    
    for i in np.arange(0.05, 1.05, 0.05):
        quantile_list.append(W.quantile(i))
    return quantile_list


def predict_causal_risk_list(train_set_X, quantiles, model):

    risk_list = []
    
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
        print(name + " being treatment...")
        print(name,' df feature importance:')
        print(list(global_value_dict[name].columns.values)[:-1])
        print(model.feature_importances_ , '\n')
        
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
print(result)
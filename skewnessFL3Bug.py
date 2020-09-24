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
import math

from skewness import good_dict
from skewness import args0
os.system('python skewness.py')


from phi import *
sd_0=None;sd_1=None;sd_2=None;data_0=None;data_1=None;data_2=None;data_3=None;data_4=None;size_0=None;variance_0=None;variance_2=None;variance_1=None;variance_3=None;variance_4=None;mean_0=None;mean_2=None;mean_1=None;mean_3=None;mean_4=None;mean_5=None;mean_6=None;mean_7=None;delta_1=None;delta_0=None;delta_2=None;x_1=None;x_0=None;x_2=None;i_0=None;i_1=None;i_2=None;skewness_0=None;stride_0=None;stride_1=None;stride_2=None;stride_3=None;stride_4=None;n_0=None;n_1=None;n_2=None;n_3=None;skew_0=None;skew_2=None;skew_1=None;skew_3=None

import math
import numpy as np
def gsl_stats_mean(data,stride,size):
    data_0 = data;stride_0 = stride;size_0 = size;
    mean_0=None;mean_2=None;mean_1=None;mean_3=None;

    mean_0=0 
    phi0 = Phi()
    for i_0 in range(0,size_0):
        phi0.set()
        mean_2 = phi0.phiEntry(mean_0,mean_1)

        mean_1 = mean_2+(data_0[i_0*stride_0]-mean_2)/(i_0+1)
    mean_3 = phi0.phiExit(mean_0,mean_1)
    lo = locals()
    record_locals(lo, test_counter)
    return mean_3

def compute_variance(data,stride,n,mean):
    data_1 = data;stride_1 = stride;n_0 = n;mean_4 = mean;
    variance_0=None;variance_2=None;variance_1=None;variance_3=None;delta_1=None;delta_0=None;delta_2=None;

    variance_0=0 
    phi0 = Phi()
    for i_1 in range(0,n_0):
        phi0.set()
        variance_2 = phi0.phiEntry(variance_0,variance_1)
        delta_1 = phi0.phiEntry(None,delta_0)

        delta_0=(data_1[i_1*stride_1]-mean_4) + bug2
        variance_1 = variance_2+(delta_0*delta_0-variance_2)/(i_1+1)
    variance_3 = phi0.phiExit(variance_0,variance_1)
    delta_2 = phi0.phiExit(None,delta_0)
    lo = locals()
    record_locals(lo, test_counter)
    return variance_3

def gsl_stats_sd_m(data,stride,n,mean):
    data_2 = data;stride_2 = stride;n_1 = n;mean_5 = mean;
    sd_0=None;variance_4=None;

    variance_4=compute_variance(data_2,stride_2,n_1,mean_5) 
    sd_0=math.sqrt(variance_4*(n_1/(n_1-1)))
    lo = locals()
    record_locals(lo, test_counter)
    return sd_0

def gsl_stats_skew_m_sd(data,stride,n,mean,sd):
    data_3 = data;stride_3 = stride;n_2 = n;mean_6 = mean;sd_1 = sd;
    x_1=None;x_0=None;x_2=None;skew_0=None;skew_2=None;skew_1=None;skew_3=None;

    skew_0=0 
    phi0 = Phi()
    for i_2 in range(0,n_2):
        phi0.set()
        x_1 = phi0.phiEntry(None,x_0)
        skew_2 = phi0.phiEntry(skew_0,skew_1)

        x_0=(data_3[i_2*stride_3]-mean_6)/sd_1 + bug1
        skew_1 = skew_2+(x_0*x_0*x_0-skew_2)/(i_2+1) 
    x_2 = phi0.phiExit(None,x_0)
    skew_3 = phi0.phiExit(skew_0,skew_1)
    lo = locals()
    record_locals(lo, test_counter)
    return skew_3

def gsl_stats_skew(data,stride,n):
    data_4 = data;stride_4 = stride;n_3 = n;
    sd_2=None;mean_7=None;skewness_0=None;

    mean_7=gsl_stats_mean(data_4,stride_4,n_3) + bug3
    sd_2=gsl_stats_sd_m(data_4,stride_4,n_3,mean_7) 
    skewness_0 = gsl_stats_skew_m_sd(data_4,stride_4,n_3,mean_7,sd_2) 
    lo = locals()
    record_locals(lo, test_counter)
    return skewness_0



#generate python causal map
causal_map = dict(mean_7=[],mean_3=['mean_0','mean_1'],mean_2=['mean_0','mean_1'],x_0=['i_2','stride_3','mean_6','sd_1'],mean_1=['mean_2','data_0','i_0','stride_0','mean_2','i_0'],x_2=['x_0'],mean_0=[],skew_0=[],x_1=['x_0'],skew_1=['skew_2','x_0','x_0','x_0','skew_2','i_2'],skew_2=['skew_0','skew_1'],skew_3=['skew_0','skew_1'],sd_2=['data_4','stride_4','n_3','mean_7'],sd_0=['variance_4','n_1','n_1'],delta_0=['data_1','i_1','stride_1','mean_4'],variance_4=['data_2','stride_2','n_1','mean_5'],delta_1=['delta_0'],variance_3=['variance_0','variance_1'],variance_2=['variance_0','variance_1'],delta_2=['delta_0'],variance_1=['variance_2','delta_0','delta_0','variance_2','i_1'],skewness_0=['data_4','stride_4','n_3','mean_7','sd_2'],variance_0=[],)

#added phi names
phi_names_set = {'mean_2','mean_3','variance_2','delta_1','variance_3','delta_2','x_1','skew_2','x_2','skew_3',}

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
global_value_dict = {}
test_counter = 0
args1 = args0
bug1 = 0
bug2 = 0
bug3 = 0
for arg1 in args1:
    bug1 = fluky(0, -0.022, 0.05)
    bug2 = fluky(0, 1, 0.05)
    bug3 = fluky(0, -1, 0.05)
    sk = gsl_stats_skew(arg1, 1, len(arg1))
    bad_dict[test_counter] = sk
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
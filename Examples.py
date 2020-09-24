import sys

import numpy as np
import array
import random
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

from phi import *

import numpy as np
import array
import random

#
# def harmean(a, b):
#     x = a + b
#     y = a * b
#     pred = x == 0
#     if pred:
#         return 0
#     else:
#         r = 2 * y / x + x
#         return r
#
#
# def harmeanFault(a, b ,c):
#     x = a + b + c #BUG
#     y = a * b
#     pred = x == 0
#     if pred:
#         return 0
#     else:
#         r = 2 * y / x + x
#         return r
def harmean(a,b):
    a_0 = a;b_0 = b;
    r_0=None;r_1=None;pred_0=None;x_0=None;y_0=None;

    x_0=a_0+b_0
    y_0=a_0*b_0
    pred_0=x_0==0
    if pred_0:
        r_0=0
    else:
        r_0=2*y_0/x_0+x_0

    if r_0==0:
        return 1
    else:
         return 0
    phiPreds = [pred_0]
    phiNames = [None,r_0]
    r_1= phiIf(phiPreds, phiNames)

def harmeanFault(a,b,c):
    a_1 = a;b_1 = b;c_0 = c;
    r_2=None;r_3=None;pred_1=None;x_1=None;y_1=None;

    x_1 = a_1+b_1+c_0 #BUG
    y_1=a_1*b_1
    pred_1 = x_1==0

    if pred_1:
       r_2=0
    else:
        r_2=2*y_1/x_1+x_1
    lo = locals()
    record_locals(lo, test_counter)
    if r_2==0:
         return 1
    else:
        return 0
    lo = locals()
    record_locals(lo, test_counter)
    phiPreds = [pred_1]
    phiNames = [None,r_2]
    r_3= phiIf(phiPreds, phiNames)



#generate python causal map
causal_map = {'r_0':['y_0','x_0','x_0'],'r_2':['y_1','x_1','x_1'],'r_1':['r_0','pred_0'],'pred_0':['x_0'],'pred_1':['x_1'],'r_3':['r_2','pred_1'],'x_0':['a_0','b_0'],'y_1':['a_1','b_1'],'y_0':['a_0','b_0'],'x_1':['a_1','b_1','c_0'],}

#added phi names
phi_names_set = {'r_1','r_3',}

d = {}
f = {}

global_value_dict = {}
arg1s = np.arange(0, 10, 0.01)
test_counter = 0
diffCnt=0


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

# TESTING DONE HERE
for i in range(1, 1000):
    a = random.randint(-10, 10)
    b = random.randint(-10, 10)
    c = random.randint(0, 1)
    d[i] = harmean(a, b)
    f[i] = harmeanFault(a, b, c)
    test_counter += 1
    if d[i] != f[i]:
        diffCnt += 1
diff_dict = {index: 0.0 if f[index] == d[index] else 1.0 for index in d}
print(diff_dict)
print(diffCnt)

for key in global_value_dict:
    rows = global_value_dict[key].index + 1
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
        with open('.' + '/' + str(name) + ".txt",
                  "w") as f1:
            f1.write(str(train_set_X.to_csv(sep=' ', mode='a')))
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
print('*************Target variables in total: ', len(result), '*************')
print(result)

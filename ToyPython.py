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


#
def testNegFlt(a, b):
    """ All inputs should be negative; if this returns true, that means that one of the 2 is not negative"""
    result = 0
    p = a < 2
    if p:
        result += 1
    p1 = b < 0
    if p1:
        result += 1
    if result != 2:
        return 1
    else:
        return 0


def testNegFlt(a, b):
    a_0 = a;
    b_0 = b;
    result_0 = None;
    result_1 = None;
    result_2 = None;
    result_3 = None;
    result_4 = None;
    p_0 = None;
    p1_0 = None;

    """ All inputs should be negative; if this returns true, that means that one of the 2 is not negative"""
    result_0 = 0
    p_0 = a_0 < 2  # BUG
    if p_0:
        result_1 = result_0 + 1
    phiPreds = [p_0]
    phiNames = [result_1, result_0]
    result_2 = phiIf(phiPreds, phiNames)
    p1_0 = b_0 < 0
    if p1_0:
        result_3 = result_2 + 1
    phiPreds = [p1_0]
    phiNames = [result_3, result_2]
    result_4 = phiIf(phiPreds, phiNames)
    lo = locals()
    record_locals(lo, test_counter)
    if result_4 != 2:
        return 1
    else:
        return 0


#
# def testNeg(a, b):
#     """ All inputs should be negative; if this returns true, that means that one of the 2 is not negative"""
#     result = 0
#     p = a < 0
#     p1 = b < 0
#     if p:
#         result += 1
#     if p1:
#         result += 1
#     lo = locals()
#     record_locals(lo, test_counter)
#     if result!=2:
#         return 1
#     else:
#         return 0


def testNeg(a, b):
    a_1 = a;
    b_1 = b;
    result_5 = None;
    result_6 = None;
    result_7 = None;
    result_8 = None;
    result_9 = None;
    p_1 = None;
    p1_1 = None;

    """ All inputs should be negative; if this returns true, that means that one of the 2 is not negative"""
    result_5 = 0
    p_1 = a_1 < 0
    p1_1 = b_1 < 0
    if p_1:
        result_6 = result_5 + 1
    phiPreds = [p_1]
    phiNames = [result_6, result_5]
    result_7 = phiIf(phiPreds, phiNames)
    if p1_1:
        result_8 = result_7 + 1
    phiPreds = [p1_1]
    phiNames = [result_8, result_7]
    result_9 = phiIf(phiPreds, phiNames)

    if result_9 != 2:
        return 1
    else:
        return 0


# generate python causal map
causal_map = {'p_0': ['a_0'], 'p1_1': ['b_1'], 'p1_0': ['b_0'], 'p_1': ['a_1'], 'result_6': ['result_5'],
              'result_7': ['result_6', 'result_5', 'p_1'], 'result_8': ['result_7'],
              'result_9': ['result_8', 'result_7', 'p1_1'], 'result_2': ['result_1', 'result_0', 'p_0'],
              'result_3': ['result_2'], 'result_4': ['result_3', 'result_2', 'p1_0'], 'result_5': [], 'result_0': [],
              'result_1': ['result_0'], }

# added phi names
phi_names_set = {'result_2', 'result_4', 'result_7', 'result_9', }


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


globalres = pd.DataFrame(columns=['#Failing-Test', 'max_risk_diff'])
for i in range(1, 2):
    bad_dict = {}
    global_value_dict = {}
    test_counter = 0
    diffCnt = 0
    d = {}
    f = {}
    outcome_list = {}
    "testing happening here"
    for i in range(1, 400):
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        d[i] = testNeg(a, b)
        f[i] = testNegFlt(a, b)
        test_counter += 1
        if d[i] != f[i]:
            diffCnt += 1

    diff_dict = {index: 0.0 if f[index] == d[index] else 1.0 for index in d}
    print(diff_dict)
    print("How many failed?: " + str(diffCnt))
    for key in global_value_dict:
        rows = global_value_dict[key].index + 1
        outcome_list = [diff_dict[i] for i in rows]
        global_value_dict[key]['outcome'] = outcome_list


    def get_quantiled_tr(W):
        # 10 quantiles from 0.05 to 0.95
        quantile_list = []
        for i in np.arange(0.25, 1.05, 0.5):  # THIS IS WHERE QUANTITIES ARE MARKED
            quantile_list.append(W.quantile(i))
        return quantile_list


    def predict_causal_risk_list(train_set_X, quantiles, model):
        risk_list = []
        print(train_set_X.columns[0] + " being treated...")
        X_with_quantile = train_set_X.drop(train_set_X.columns[0], axis=1)

        for quantile in quantiles:
            X_with_quantile.insert(loc=0, column=train_set_X.columns[0],
                                   value=np.full((len(X_with_quantile), 1), quantile))
            # X_with_quantile[train_set_X.columns[col_index_todrop]] = np.full((len(X_with_quantile), 1), quantile)
            # print(X_with_quantile.describe())
            risk_list.append(model.predict(X_with_quantile).mean())
            modelsover = model.predict(X_with_quantile)
            print("model :", quantile)
            print(modelsover)
            print(len(modelsover))
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
            filename = "./" + os.path.basename(__file__)[:-3] + "/" + str(name) + ".txt"

            os.makedirs(os.path.dirname(filename), exist_ok=True)

            W = df.iloc[:, 0].to_frame()
            with open(filename,
                      "w") as f1:
                f1.write(str(train_set_X.to_csv(sep=' ', mode='a')))
            quantiles = get_quantiled_tr(W)
            risk_list = predict_causal_risk_list(train_set_X, quantiles, model)
            max_risk = max(risk_list)
            min_risk = min(risk_list)
            row = [df.columns[0], max_risk - min_risk, risk_list.index(max_risk),
                   risk_list.index(min_risk)]
            if name == "p_0":
                globalres.loc[len(globalres)] = [diffCnt, max_risk - min_risk]
            suspicious_df.loc[len(suspicious_df)] = row
        suspicious_df = suspicious_df.sort_values(by='max_risk_diff', ascending=False)
        return filter_phi_rows(suspicious_df, phi_names_set)


    def filter_phi_rows(suspicious_df, phi_names_set):
        return suspicious_df[~suspicious_df['variable_name'].isin(phi_names_set)]


    # 0-> random forest  1 -> lasso
    result = suspicious_ranking(global_value_dict, 0)
    pd.set_option("display.precision", 4)
    print('*************Target variables in total: ', len(result), '*************')
    print(result)

globalres.to_csv('Global.csv', sep='\t', mode='a', header=False)

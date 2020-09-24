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

from matric_transpose import good_dict
from matric_transpose import args0
os.system('python matric_transpose.py')

from phi import *
n1_0=None;n2_0=None;i_0=None;j_0=None;k_0=None;e1_5=None;e1_3=None;e1_1=None;e1_0=None;e1_2=None;e1_4=None;e1_6=None;m_0=None;m_1=None;e2_5=None;e2_3=None;e2_1=None;e2_0=None;e2_2=None;e2_4=None;e2_6=None;view_0=None;array_0=None;size1_0=None;tmp_5=None;tmp_3=None;tmp_1=None;tmp_0=None;tmp_2=None;tmp_4=None;tmp_6=None;size2_0=None

class gsl_block_struct(object):
    __slots__=['size','data'] 
gsl_block=gsl_block_struct 
NULL_MATRIX_VIEW=[[0,0,0,0,0,0]] 
NULL_MATRIX=[0,0,0,0,0,0] 
GSL_SUCCESS=1 
MULTIPLICITY=1 
class gsl_matrix(object):
    __slots__=['size1','size2','tda','data','block','owner'] 
class _gsl_matrix_view(object):
    __slots__=['matrix'] 
gsl_matrix_view=_gsl_matrix_view 
def gsl_matrix_view_array(array,n1,n2):
    array_0 = array;n1_0 = n1;n2_0 = n2;
    view_0=None;m_0=None;

    view_0=_gsl_matrix_view() 
    view_0.matrix=gsl_matrix() 
    view_0.matrix.size1=0 
    view_0.matrix.size2=0 
    view_0.matrix.tda=0 
    view_0.matrix.data=0 
    view_0.matrix.block=0 
    view_0.matrix.owner=0 
    m_0=gsl_matrix() 
    m_0.size1,m_0.size2,m_0.tda,m_0.data,m_0.block,m_0.owner=NULL_MATRIX 
    m_0.data=array_0 
    m_0.size1=n1_0 
    m_0.size2=n2_0 
    m_0.tda=n2_0 
    m_0.block=0 
    m_0.owner=0 
    view_0.matrix=m_0 
    lo = locals()
    record_locals(lo, test_counter)
    return view_0

def gsl_matrix_transpose(m):
    m_1 = m;
    size1_0=None;tmp_5=None;tmp_3=None;tmp_1=None;tmp_0=None;tmp_2=None;tmp_4=None;tmp_6=None;size2_0=None;e1_5=None;e1_3=None;e1_1=None;e1_0=None;e1_2=None;e1_4=None;e1_6=None;e2_5=None;e2_3=None;e2_1=None;e2_0=None;e2_2=None;e2_4=None;e2_6=None;
    size1_0=m_1.size1 
    size2_0=m_1.size2 
    if size1_0!=size2_0:
        print("matrix must be square to take transpose") 
    phi0 = Phi()
    for i_0 in range(0,size1_0):
        phi0.set()
        tmp_5 = phi0.phiEntry(None,tmp_4)
        e1_5 = phi0.phiEntry(None,e1_4)
        e2_5 = phi0.phiEntry(None,e2_4)
        phi1 = Phi()
        for j_0 in range(i_0+1,size2_0):
            phi1.set()
            tmp_3 = phi1.phiEntry(tmp_5,tmp_2)
            e1_3 = phi1.phiEntry(e1_5,e1_2)
            e2_3 = phi1.phiEntry(e2_5,e2_2)
            phi2 = Phi()
            for k_0 in range(0,MULTIPLICITY):
                phi2.set()
                tmp_1 = phi2.phiEntry(tmp_3,tmp_0)
                e1_1 = phi2.phiEntry(e1_3,e1_0)
                e2_1 = phi2.phiEntry(e2_3,e2_0)
                e1_0 = (i_0*m_1.tda+j_0) * MULTIPLICITY+k_0
                e2_0=(j_0*m_1.tda+i_0)*MULTIPLICITY+k_0 
                tmp_0=m_1.data[e1_0]
                m_1.data[e1_0]=m_1.data[e2_0]
                m_1.data[e2_0]=tmp_0
            tmp_2 = phi2.phiExit(tmp_3,tmp_0)
            e1_2 = phi2.phiExit(e1_3,e1_0)
            e2_2 = phi2.phiExit(e2_3,e2_0)
        tmp_4 = phi1.phiExit(tmp_5,tmp_2)
        e1_4 = phi1.phiExit(e1_5,e1_2)
        e2_4 = phi1.phiExit(e2_5,e2_2)
    tmp_6 = phi0.phiExit(None,tmp_4)
    e1_6 = phi0.phiExit(None,e1_4)
    e2_6 = phi0.phiExit(None,e2_4)
    lo = locals()
    record_locals(lo, test_counter)
    return GSL_SUCCESS


#generate python causal map
causal_map = dict(view_0=[],e2_6=['tmp_4','e1_4','e2_4'],e1_5=['e1_4'],
                  e2_4=['tmp_5','tmp_2','e1_5','e1_2','e2_5','e2_2'],
                  e2_5=['e2_4'],e1_6=['tmp_4','e1_4','e2_4'],
                  e1_3=['e1_5','e1_2'],e2_2=['tmp_3','tmp_0','e1_3','e1_0','e2_3','e2_0'],
                  e2_3=['e2_5','e2_2'],e1_4=['tmp_5','tmp_2','e1_5','e1_2','e2_5','e2_2'],
                  tmp_5=['tmp_4'],tmp_4=['tmp_5','tmp_2','e1_5','e1_2','e2_5','e2_2'],
                  tmp_6=['tmp_4','e1_4','e2_4'],tmp_1=['tmp_3','tmp_0'],tmp_0=['m_1','e1_0'],
                  tmp_3=['tmp_5','tmp_2'],m_0=[],tmp_2=['tmp_3','tmp_0','e1_3','e1_0','e2_3','e2_0'],
                  e1_1=['e1_3','e1_0'],e2_0=['j_0','m_1','i_0','k_0'],e2_1=['e2_3','e2_0'],
                  e1_2=['tmp_3','tmp_0','e1_3','e1_0','e2_3','e2_0'],e1_0=['i_0','m_1','j_0','k_0'],
                  size2_0=['m_1'],size1_0=['m_1'],)




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

def random_err_gen(a, b):
    ran = random.randint(1,2)
    if ran == 1:
        return a
    else:
        return b


bad_dict = {}
global_value_dict = {}
test_counter = 0
args1 = args0
for arg1 in args1:
    m = gsl_matrix_view_array(arg1, 8, 8)
    gsl_matrix_transpose(m.matrix)
    bad_dict[test_counter] = (m.matrix.data[0], m.matrix.data[1])
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
        row = [df.columns[0], max_risk - min_risk, risk_list.index(max_risk), risk_list.index(min_risk)]
        suspicious_df.loc[len(suspicious_df)] = row
    return suspicious_df.sort_values(by='max_risk_diff', ascending=False)

# 0-> random forest  1 -> lasso
result = suspicious_ranking(global_value_dict, 0)
pd.set_option("display.precision", 8)
print(result)
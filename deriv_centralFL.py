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

insertion_count = 0
from deriv_central import good_dict, args0
os.system('python deriv_central.py') 


from phi import *
fmh_0=None;abseer_0=None;r3_0=None;r5_0=None;F_0=None;fm1_0=None;error_0=None;error_1=None;error_2=None;error_3=None;error_4=None;result_0=None;result_1=None;result_2=None;result_3=None;abserr_0=None;dy_0=None;r_opt_0=None;r_opt_1=None;error_opt_0=None;error_opt_1=None;error_opt_2=None;function_0=None;fph_0=None;a_0=None;b_0=None;r_0_0=None;r_0_1=None;r_0_2=None;r_0_3=None;r_0_4=None;f_0=None;f_1=None;fp1_0=None;h_0=None;h_1=None;params_0=None;params_1=None;e3_0=None;e5_0=None;round_0=None;round_1=None;trunc_0=None;trunc_1=None;trunc_opt_0=None;trunc_opt_1=None;x_0=None;x_1=None;x_2=None;x_3=None;abserr_trunc_0=None;abserr_trunc_1=None;round_opt_0=None;round_opt_1=None;h_opt_0=None;h_opt_1=None;abserr_round_0=None;abserr_round_1=None

GSL_SUCCESS=1 
def GSL_MAX(a,b):
    a_0 = a;b_0 = b;
    return max(a_0,b_0)

class gsl_function:
    def __init__(self,function,params):
        function_0 = function;params_0 = params;
        

        self.function=function_0 
        self.params=params_0 

GSL_DBL_EPSILON=2.2204460492503131e-16 
def GSL_FN_EVAL(F,x):
    F_0 = F;x_0 = x;
    

    return F_0.function(x_0,F_0.params)

def central_deriv(f,x,h,result,abserr_round,abserr_trunc):
    f_0 = f;x_1 = x;h_0 = h;result_0 = result;abserr_round_0 = abserr_round;abserr_trunc_0 = abserr_trunc;
    fmh_0=None;r3_0=None;r5_0=None;fp1_0=None;fm1_0=None;e3_0=None;result_1=None;e5_0=None;dy_0=None;abserr_trunc_1=None;abserr_round_1=None;fph_0=None;

    gen_bad = random() < probability
    global insertion_count
    if gen_bad:
        insertion_count += 1

    fm1_0=GSL_FN_EVAL(f_0,x_1-h_0) 
    fp1_0=GSL_FN_EVAL(f_0,x_1+h_0) 
    fmh_0=GSL_FN_EVAL(f_0,x_1-h_0/2) 
    fph_0=GSL_FN_EVAL(f_0,x_1+h_0/2) 
    r3_0=0.5*(fp1_0-fm1_0) 
    r5_0=(4.0/3.0)*(fph_0-fmh_0)-(1.0/3.0)*r3_0 
    e3_0=(fabs(fp1_0)+fabs(fm1_0))*GSL_DBL_EPSILON 
    e5_0=2.0*(fabs(fph_0)+fabs(fmh_0))*GSL_DBL_EPSILON+e3_0 
    dy_0=fuzzy(GSL_MAX(fabs(r3_0/h_0),fabs(r5_0/h_0))*(fabs(x_1)/h_0)*GSL_DBL_EPSILON, gen_bad)
    result_1=r5_0/h_0 
    abserr_trunc_1=fabs((r5_0-r3_0)/h_0) 
    abserr_round_1=fabs(e5_0/h_0)+dy_0 
    lo = locals()
    record_locals(lo, test_counter)
    return result_1,abserr_trunc_1,abserr_round_1

def gsl_deriv_central(f,x,h,result,abseer):
    f_1 = f;x_2 = x;h_1 = h;result_2 = result;abseer_0 = abseer;
    r_0_0=None;r_0_1=None;r_0_2=None;r_0_3=None;r_0_4=None;error_0=None;error_1=None;error_2=None;error_3=None;error_4=None;result_3=None;abserr_0=None;round_0=None;round_1=None;trunc_0=None;trunc_1=None;r_opt_0=None;r_opt_1=None;error_opt_0=None;error_opt_1=None;error_opt_2=None;trunc_opt_0=None;trunc_opt_1=None;round_opt_0=None;round_opt_1=None;h_opt_0=None;h_opt_1=None;

    r_0_0=0.0 
    round_0=0.0 
    trunc_0=0.0 
    error_0=0.0 
    r_0_1,round_1,trunc_1,=central_deriv(f_1,x_2,h_1,r_0_0,round_0,trunc_0) 
    error_1=round_1+trunc_1 
    if round_1<trunc_1 and (round_1>0 and trunc_1>0):
        r_opt_0=0.0 
        round_opt_0=0.0 
        trunc_opt_0=0.0 
        error_opt_0=0.0 
        h_opt_0=h_1*pow(round_1/(2.0*trunc_1),1.0/3.0) 
        central_deriv(f_1,x_2,h_opt_0,r_opt_0,round_opt_0,trunc_opt_0) 
        error_opt_1=round_opt_0+trunc_opt_0 
        if error_opt_1<error_1 and fabs(r_opt_0-r_0_1)<4.0*error_1:
            r_0_2=r_opt_0 
            error_2=error_opt_1 
        phiPreds = [error_opt_1<error_1 and fabs(r_opt_0-r_0_1)<4.0*error_1]
        phiNames = [r_0_2,r_0_1]
        r_0_3= phiIf(phiPreds, phiNames)
        phiPreds = [error_opt_1<error_1 and fabs(r_opt_0-r_0_1)<4.0*error_1]
        phiNames = [error_2,error_1]
        error_3= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [r_0_3,r_0_1]
    r_0_4= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [r_opt_0,None]
    r_opt_1= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [error_opt_1,None]
    error_opt_2= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [trunc_opt_0,None]
    trunc_opt_1= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [round_opt_0,None]
    round_opt_1= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [h_opt_0,None]
    h_opt_1= phiIf(phiPreds, phiNames)
    phiPreds = [round_1<trunc_1 and (round_1>0 and trunc_1>0)]
    phiNames = [error_3,error_1]
    error_4= phiIf(phiPreds, phiNames)
    result_3=r_0_4 
    abserr_0=error_4 
    lo = locals()
    record_locals(lo, test_counter)
    
    return result_3,abserr_0


def f(x,params):
    x_3 = x;params_1 = params;


    return pow(x_3,1.5)


#generate python causal map
causal_map = {'r5_0':['fph_0','fmh_0','r3_0'],'trunc_opt_1':['trunc_opt_0'],'trunc_opt_0':[],'r3_0':['fp1_0','fm1_0'],'fp1_0':['f_0','x_1','h_0'],'e5_0':['fph_0','fmh_0','e3_0'],'error_opt_1':['round_opt_0','trunc_opt_0'],'error_opt_2':['error_opt_1'],'abserr_0':['error_4'],'error_opt_0':[],'round_1':['f_1','x_2','h_1','r_0_0','round_0','trunc_0'],'fph_0':['f_0','x_1','h_0'],'round_opt_1':['round_opt_0'],'error_0':[],'trunc_1':['f_1','x_2','h_1','r_0_0','round_0','trunc_0'],'fmh_0':['f_0','x_1','h_0'],'trunc_0':[],'error_1':['round_1','trunc_1'],'error_2':['error_opt_1'],'abserr_trunc_1':['r5_0','r3_0','h_0'],'r_opt_0':[],'r_0_2':['r_opt_0'],'error_3':['error_2','error_1'],'r_0_3':['r_0_2','r_0_1'],'error_4':['error_3','error_1'],'r_0_0':[],'r_0_1':['f_1','x_2','h_1','r_0_0','round_0','trunc_0'],'round_opt_0':[],'dy_0':['r3_0','h_0','r5_0','h_0','x_1','h_0'],'r_0_4':['r_0_3','r_0_1'],'round_0':[],'r_opt_1':['r_opt_0'],'result_3':['r_0_4'],'e3_0':['fp1_0','fm1_0'],'result_1':['r5_0','h_0'],'fm1_0':['f_0','x_1','h_0'],'h_opt_0':['h_1','round_1','trunc_1'],'h_opt_1':['h_opt_0'],'abserr_round_1':['e5_0','h_0','dy_0'],}

#added phi names
phi_names_set = {'r_0_3','error_3','r_0_4','r_opt_1','error_opt_2','trunc_opt_1','round_opt_1','h_opt_1','error_4',}

F = gsl_function(f, 0)



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
args1 = args0
test_counter = 0
# add this for NUMFL
filename = './' + os.path.basename(__file__)[:-3] + sys.argv[1] + '-NUMFL/Data1/'
os.makedirs(os.path.dirname(filename), exist_ok=True)
bugid = version_bug_dict[str(os.path.basename(sys.argv[0])[:-3])]
print("Bug is " + bugid)

probability = float(sys.argv[1])/100.0
for arg1 in args1:
    result = 0.0
    abserr = 0.0
    bad_outcome = gsl_deriv_central(F, arg1, 1e-8, result, abserr)
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
    difference_dict = {index: abs(bad_dict[index][1] - good_dict[index][1]) for index in bad_dict}
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
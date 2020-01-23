from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from ABCD_ML.pipeline.Nevergrad import NevergradSearchCV
import nevergrad as ng
from sklearn.metrics import make_scorer, r2_score

import pickle
import numpy as np

from ABCD_ML import ABCD_ML

def get_models(n_jobs):

    scaler = RobustScaler(quantile_range=(5, 95))

    base_lgbm_model = LGBMRegressor(n_jobs=n_jobs)
    lgbm_pipeline = Pipeline([('scaler', scaler), ('base_model', base_lgbm_model)])

    base_xgb_model = XGBRegressor(n_jobs = n_jobs, verbosity=0)
    xgb_pipeline = Pipeline([('scaler', scaler), ('base_model', base_xgb_model)])

    models = [[lgbm_pipeline], [xgb_pipeline, xgb_pipeline]]

    return models


def get_s_data(d_type):

    with open('data/' + d_type + '_lh_sst.pkl', 'rb') as f:
        lh = pickle.load(f)
    with open('data/' + d_type + '_rh_sst.pkl', 'rb') as f:
        rh = pickle.load(f)

    data_sst = [lh, rh]  

    with open('data/' + d_type + '_targets_sst.pkl', 'rb') as f:
        targets_sst = pickle.load(f)
        
    # Load Data needed
    with open('data/' + d_type + '_lh_nback.pkl', 'rb') as f:
        lh = pickle.load(f)
    with open('data/' + d_type + '_rh_nback.pkl', 'rb') as f:
        rh = pickle.load(f)

    data_nback = [lh, rh]  

    with open('data/' + d_type + '_targets_nback.pkl', 'rb') as f:
        targets_nback = pickle.load(f)
        
    data = [data_sst, data_nback]
    targets = [targets_sst, targets_nback]

    return data, targets


def get_data():

    data, targets = get_s_data('train')
        
    with open('data/geo.pkl', 'rb') as f:
        geo = pickle.load(f)
        
    medial_wall_inds = np.load('data/fs5_medial_wall_inds.npy')

    return data, targets, geo, medial_wall_inds


def get_ml(df, test_subjects, n_targets):


    ML = ABCD_ML(log_dr = None,
                 verbose = False,
                 random_state = 1)

    ML.Set_Default_Load_Params(dataset_type='custom')
    
    ML.Load_Data(df=df,
                 drop_keys=['target'])

    ML.Load_Targets(df=df,
                    col_name=['target_' + str(i) for i in range(n_targets)],
                    data_type=['f' for i in range(n_targets)])

    ML.Train_Test_Split(test_subjects=test_subjects)

    return ML

def run_on_vacc(ML, VE, results, ppn, mem, vmem):

    if VE is None:
            
        result = ML.Evaluate()
        results.append(result['summary score'][0][0])
    
    else:

        VE.run(ML=ML, ML_name='ML',
               cell='ML.Evaluate(run_name = "0")',
               ppn=ppn, mem=mem, vmem=vmem)

    return results

def level_two_eval_parcel(choice, ML, target=0, n_jobs=2, VE=None,
                         ppn='2', mem='16gb', vmem='18gb'):

    models = ['svm', 'ridge', 'light gbm', 'xgb', 'svm']
    model = models[choice]

    if choice == 4:
        feat_selector = 'univariate selection'
        feat_selector_params = 1
    else:
        feat_selector = None
        feat_selector_params = 0
    
    ML.Set_Default_ML_Params(problem_type = 'regression',
                             metric = 'r2',
                             splits = 3,
                             n_repeats = 1,
                             n_jobs = n_jobs,
                             target = target,
                             model_params = 1,
                             search_type = 'RandomSearch',
                             search_n_iter = 100,
                             ensemble = None,
                             feat_selector = feat_selector,
                             feat_selector_params = feat_selector_params,
                             model = model)

    search_types = ['RandomSearch', 'TwoPointsDE', 'DiscreteOnePlusOne']
    scalers = ['robust', 'standard']

    for st in search_types:
        for s in scalers:

            if s == 'robust':
                sp = 1
            else:
                sp = 0

            ML.Set_Default_ML_Params(search_type=st,
                                     scaler = s,
                                     scaler_params = sp)
            results = run_on_vacc(ML, VE, results, ppn, mem, vmem)

    return results

def eval_parcel(ML, target=0, n_jobs=2, VE=None,
                ppn='2', mem='16gb', vmem='18gb'):

    ML.Set_Default_ML_Params(problem_type = 'regression',
                             metric = 'r2',
                             scaler = 'robust',
                             splits = 3,
                             n_repeats = 1,
                             n_jobs = n_jobs,
                             target = target,
                             model_params = 1,
                             search_type = 'RandomSearch',
                             search_n_iter = 60,
                             feat_selector = None,
                             feat_selector_params = 0,
                             ensemble = None)

    results = []
    models = ['svm', 'ridge', 'light gbm', 'xgb']

    # Run base models
    for model in models:
        ML.Set_Default_ML_Params(model = model)
        results = run_on_vacc(ML, VE, results, ppn, mem, vmem)

    ML.Set_Default_ML_Params(model = 'svm',
                             search_type = 'RandomSearch',
                             feat_selector = 'univariate selection',
                             feat_selector_params = 1)
    results = run_on_vacc(ML, VE, results, ppn, mem, vmem)

    return results


def eval_full_parcel(train_data, val_data, train_targets,
                     val_targets, parcel, level_two=None,
                     n_jobs=4, VE=None, ppn='4',
                     mem='18gb', vmem='20gb'):

    print('Eval SST')
    df, test_subjects = parcel.get_df(train_data[0], val_data[0], train_targets[0], val_targets[0])
    ML = get_ml(df, test_subjects, 1)

    if level_two is None:
        sst_results = eval_parcel(ML, target=0, n_jobs=n_jobs, VE=VE,
                                  ppn=ppn, mem=mem, vmem=vmem)
    else:
        sst_results = level_two_eval_parcel(level_two[0], ML, target=0,
                                            n_jobs=n_jobs, VE=VE, ppn=ppn,
                                            mem=mem, vmem=vmem)

    print('Eval nBack 0')
    df, test_subjects = parcel.get_df(train_data[1], val_data[1], train_targets[1], val_targets[1])
    ML = get_ml(df, test_subjects, 2)

    if level_two is None:
        n0_results = eval_parcel(ML, target=0, n_jobs=n_jobs, VE=VE,
                                 ppn=ppn, mem=mem, vmem=vmem)
    else:
        n0_results = level_two_eval_parcel(level_two[1], ML, target=0,
                                           n_jobs=n_jobs, VE=VE, ppn=ppn,
                                           mem=mem, vmem=vmem)

    print('Eval nBack 2')
    if level_two is None:
        n2_results = eval_parcel(ML, target=1, n_jobs=n_jobs, VE=VE,
                                 ppn=ppn, mem=mem, vmem=vmem)
    else:
        n2_results = level_two_eval_parcel(level_two[2], ML, target=1,
                                           n_jobs=n_jobs, VE=VE, ppn=ppn,
                                           mem=mem, vmem=vmem)

    return sst_results, n0_results, n2_results

    









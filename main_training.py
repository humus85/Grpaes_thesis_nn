
##############################depreceated

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import webbrowser
import time
import skopt
from skopt import load
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence
# from email.mime.base import MIMEBase
# from email import encoders
import logging
from io import StringIO
# import statsmodels.api as sm
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
# import Object
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# simplefilter(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import optuna
from utils import get_data, calculate_err, predict_per_field, remove_field_col, models_with_fields_separation, df_to_view
from config import ERROR_NAME, CHROME_PATH, RESULTS_PATH, PRINT_HTML, CHECKPOINT_NAME, HYPER_PARAM_FILE_NAME, COLUMN_LABELS
from VariationalBayesModel import VariationalBayes
from FullyConnected import FullyConnectedModel
from CrossValidate import CrossValidation
from skopt.space import Real, Integer
from mailme import send_email_crash_notification
from skopt.utils import use_named_args
from MixtureModel import MixtureModel



space = [
    Real(0,30, name='a1'),
    Real(0,30, name='a2'),
    Real(0,30, name='a3'),
    Real(0,30, name='a4'),
    Real(0,30, name='a5'),
    Real(0,30, name='a6'),
    Real(0,30, name='a7'),
    Real(0,30, name='a8'),
    Real(0,30, name='a9'),
    Real(0,30, name='a10'),
    Real(0,30, name='a11'),
    # Real(10** -4, 10**-1, 'log-uniform', name='lr'),
    # Integer(95,120, name='n_hidden'),
    # Integer(0,2, name='n_h_layers'),
    # Integer(10,30, name='n_enc'),
    # Integer(0,1, name='act_func')
]


def optuna












@skopt.utils.use_named_args(space)
def objective(**params):
    global RUN
    RUN += 1
    print('####### CALL number {} #######'.format(RUN))
    # CV = CrossValidation()
    # a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = [100, 0, 100, 0, 0, 100, 100, 100, 0, 0, 0]
    lr = 0.001
    n_hidden = 102
    n_h_layers = 0
    n_enc =  18
    act_func = 1

    # params['a1'] = a1
    # params['a2'] = a2
    # params['a3'] = a3
    # params['a4'] = a4
    # params['a5'] = a5
    # params['a6'] = a6
    # params['a7'] = a7
    # params['a8'] = a8
    # params['a9'] = a9
    # params['a10'] = a10
    # params['a11'] = a11
    params['lr'] = lr
    params['n_hidden'] = n_hidden
    params['n_h_layers'] = n_h_layers
    params['n_enc'] = n_enc
    params['act_func'] = act_func
    # params['drop_out'] = drop_out


    FC_model = FullyConnectedModel(limit=3000
                                   , label_to_predict=[0,1,10], equal_weights=False)
    FC_model.set_params(**params)
    print({i: np.round(v, 7) for i, v in FC_model.__dict__.items() if i in params.keys()})

    ### new way to search ###
    result = CV.CrossValidate(X, y, idx_to_leave_out, model=FC_model, plot=False, non_aggregated=True)
    normalize_result = np.divide(result - np.array([0.032, 0.026, 0.007]), result) # ann single layer
    sum_exp_result = np.sum(np.exp(normalize_result))
    print('modified result: ',sum_exp_result)
    return sum_exp_result
    ###

    # return CV.CrossValidate(X, y, idx_to_leave_out, model=FC_model, plot=False)

def run(train_mode = True):
    if train_mode:
        global RUN
        RUN = 0
        # res = load(r"HyperParamResults.pkl")
        # x = res.x_iters
        # y = res.func_vals
        checkpoint_saver = CheckpointSaver(CHECKPOINT_NAME)
        res_gp = gp_minimize(objective,
                             space,
                             n_calls=200 ,
                             n_initial_points=40,
                             # random_state= res.random_state,
                             random_state = 123,
                             callback=[checkpoint_saver]
                             # x0=x,
                             # y0=y
                             )
        skopt.dump(res_gp, HYPER_PARAM_FILE_NAME, store_objective=False)
        print(res_gp.x)
        print(res_gp.fun)
    else:
        X, y = get_data()
        params = {'a1': 100, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0, 'a6': 0, 'a7': 0, 'a8': 0, 'a9': 0,
                  'a10': 0, 'a11': 0, 'lr': 0.00014766308602833455, 'n_hidden': 46, 'n_enc': 22}
        FC_model = FullyConnectedModel(limit=2000, equal_weights=False)
        FC_model.set_params(**params)
        print(FC_model.fit(X_train, y_train, X_test, y_test))



def optuna(direction = 'minimize',sampler= optuna.samplers.TPESampler(seed=1803),trials=30):
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(opt_objective, n_trials=trials)


def opt_objective(trial):

    params = {
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-1), # Real(10** -4, 10**-1, 'log-uniform', name='lr'),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD","RMSprop"]),
        'n_hidden': trial.suggest_int("n_hidden",95,120), # Integer(95,120, name='n_hidden'), --num of neorons in each hidden layer.
        'n_h_layers': trial.suggest_int("n_h_layers",0,2), # Integer(95,120, name='n_h_layers'),
        'n_enc': trial.suggest_int("n_enc",10,30),
        'drop_out' : trial.suggest_float("drop_out",0.2,0.5)
        'act_func': trial.suggest_categorical("act_func",["torch.nn.Tanh", "torch.nn.ReLU"])
          }

        model = build_model(params)

        accuracy = train_and_evaluate(params, model)

        return accuracy

    return accuracy


if __name__ ==  '__main__':
    MAILME = False
    train_mode = True
    X, y, idx_to_leave_out = get_data(eval=True,  cols_labels=COLUMN_LABELS)
    CV = CrossValidation(N = X.size(0), folds = 5)
    try:
        metric = ERROR_NAME
        run(train_mode)
    except Exception as e:
        log_stream = StringIO()
        if MAILME:
            logging.basicConfig(stream=log_stream, level=logging.INFO)
            logging.error("Exception occurred", exc_info=True)
            send_email_crash_notification(log_stream.getvalue())
        else:
            raise(e)


import warnings
# import Object
from warnings import simplefilter
from datetime import datetime
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
simplefilter(action='ignore', category=FutureWarning)
# simplefilter(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import pickle
import optuna
import shap
from utils import get_data
from config import COLUMN_LABELS_2020,COLUMN_LABELS_2022
from FullyConnected_v2 import FullyConnectedModel
import pandas as pd
from CrossValidate import CrossValidation
from utils import get_labels_position_to_predict_by_task,shared_cols_2020_2022,shared_cols_2020_2021_no_interuptions
import torch
from openpyxl import load_workbook


def run_optuna(X, y,direction='minimize',sampler= optuna.samplers.TPESampler(seed=1803),trials=30, pruner=optuna.pruners.MedianPruner()):
    study = optuna.create_study(direction=direction, sampler=sampler)
    og = Objective_Generator(X, y)
    study.optimize(og.opt_objective, n_trials=trials)
    return study

class Objective_Generator:
    def __init__ (self, X, y, num_folds=5):
        self.X = X
        self.y = y
        self.num_folds = num_folds

    def opt_objective(self, trial):

        params = {
            'lr': trial.suggest_categorical('lr', [1e-4, 1e-3 ,1e-2 ,1e-1]), # Real(10** -4, 10**-1, 'log-uniform', name='lr'),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD","RMSprop"]),
            'n_hidden': trial.suggest_int("n_hidden",95,120), # Integer(95,120, name='n_hidden'),
            'n_h_layers': trial.suggest_int("n_h_layers",0,2), # Integer(95,120, name='n_h_layers'),
            'n_enc': trial.suggest_int("n_enc",10,30),
            'drop_out' : trial.suggest_categorical("drop_out",[0.2,0.3,0.4,0.5]),
            'act_func': trial.suggest_categorical("act_func",[0,1]) #0 for tanh, 1 for relu
        }
        for i in range (y.size(1)):
            key = 'a' + str(i+1)
            params[key] = trial.suggest_float(key,0,30)

        model = FullyConnectedModel(limit=3000, y_size=y.size(1), label_to_predict=labels_position_to_predict, max_epoch=TRAIN_EPOCHS, equal_weights=False)
        model.set_params(**params)
        #calculate CV per trail
        err=0
        kf = KFold(n_splits=self.num_folds)
        i=1
        for train_index, test_index in kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[test_index]
            y_train, y_val = self.y[train_index], self.y[test_index]
            curr_err, single_losses = model.fit(X_train, y_train, X_val, y_val, trial=trial)
            err += curr_err/self.num_folds
            single_losses += [x / self.num_folds for x in single_losses]
            print(single_losses)
            print(err)
            print('end fold number:', i)
            i+=1
        return err


def prep_df_for_results(model_names_for_index,model_names_for_index_advanced):
    results_df = pd.DataFrame(columns=['Acceptance Score', 'Rachis Score', 'Decay'],
                              index=model_names_for_index)
                              # ['Linear Regression_Storage','Linear Regression','SVM','Gradient Boosting','Random Forest','ANN','MTPS-ANN'
                              #        ])
    results_df_2 = pd.DataFrame({'method' : ['PCA'] * 4 + ['AE'] * 4 +['MTPS-ANN-Encoder'] * 4,
                                 'algo' : model_names_for_index_advanced*3,
                                  'Acceptence Score': np.nan*12,
                                  'Rachis Score': np.nan*12,
                                  'Decay': np.nan*12
                                 })
    results_df_2_final = results_df_2.set_index(['method','algo'])
    return results_df,results_df_2_final


def get_data_for_baselines(X_base,y_base,labels_positions):
    seed=42
    CV_func = CrossValidation(N = X_base.size(0), folds = 5, SEED = seed,label_to_predict=labels_positions)
    X = X_base.data.numpy()
    y = y_base.data.numpy()
    y = y[:,labels_positions]
    return(X,y,CV_func)



def run_compare_basic_baselines(models,model_names,results_df,CV_sklearn,labels_positions,X,y,ann_res, mtps_ann_res):
    for j in range(len(labels_positions)):
        y_predicted = y[:,j]
        x_for_model = X
        for i,model_name in enumerate(models):
            if i==0:
                x_for_model= X[:,[0]]
                # x_for_model = x_for_model.reshape(-1, 1)
            else:
                x_for_model = X
            results_df.loc[model_names[i], results_df.columns[j]] = CV_sklearn.CrossValidate_sklearn(x_for_model, y_predicted, model= model_name)
    sqrt_df = results_df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    sqrt_df = np.sqrt(sqrt_df)
    sqrt_df.loc['ANN']=ann_res
    sqrt_df.loc['MTPS-ANN']=mtps_ann_res
    results_df.loc['ANN']=np.power(ann_res,2)
    results_df.loc['MTPS-ANN']=np.power(mtps_ann_res,2)
    return results_df, sqrt_df

def prep_net_for_ae(X_tensor):
    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_h_layers , n_enc, n_output ,act_func = 0):
            super(Net, self).__init__()
            torch.manual_seed(1)
            self.hidden_layers = []

            self.hidden = torch.nn.Linear(n_feature, n_hidden)
            for layer in range(n_h_layers):
                self.hidden_layers.append(torch.nn.Linear(n_hidden, n_hidden))  # hidden layer*
            self.enc = torch.nn.Linear(n_hidden, n_enc)  # encoder layer
            self.predict = torch.nn.Linear(n_enc, n_output)   # output layer
            func_list = [torch.nn.Tanh, torch.nn.ReLU]
            self.activation_func = func_list[act_func]()
            self.dropout1 = torch.nn.Dropout(p=0.5)
            self.dropout2 = torch.nn.Dropout(p=0.2)


        def forward(self, x):
            x = self.activation_func(self.hidden(x))      # activation function for hidden layer
            for h_layer in self.hidden_layers:
                x = self.activation_func(h_layer(x))
            x = self.activation_func(self.enc(x))              # encoder layer
            x = self.predict(x)           # linear output
            return x

        def extract_enc_layer(self, x):
            x = self.activation_func(self.hidden(x))  # activation function for hidden layer
            for h_layer in self.hidden_layers:
                x = self.activation_func(h_layer(x))
            x = self.activation_func(self.enc(x))
            return x

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
    n_features = X_tensor.shape[1]
    n_labels = X_tensor.shape[1]
    net = Net(n_feature=n_features, n_hidden=102, n_h_layers=0, n_enc=18, act_func=0, n_output=n_labels)
    net = net.float()
    net.apply(init_weights)
    lr = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr= lr)
    loss_func = torch.nn.MSELoss()
    epochs = 5000
    for i in range(epochs):
        prediction = net(X_tensor.float())  # input x and predict based on x
        optimizer.zero_grad()
        loss = loss_func(prediction, X_tensor.float())
        loss.backward()
        optimizer.step()  # apply gradients


    enc_vecs_AE = net.extract_enc_layer(X_tensor.float())
    enc_vecs_AE = enc_vecs_AE.detach().numpy()
    return enc_vecs_AE


def run_compare_advanced_baselines(models_advanced,model_names_advanced,results_df,CV_sklearn,labels_positions,X,y,x_enc_vec_model,X_for_ae,n_comp_pca=6):
    methods =['PCA','AE','MTPS-ANN-Encoder']
    pca = PCA(n_components=n_comp_pca)
    X_pca = pca.fit_transform(X)
    X_enc = prep_net_for_ae(X_for_ae)
    X_MTPS_ANN = x_enc_vec_model
    x_list_predict = [X_pca,X_enc,X_MTPS_ANN]
    for k,x_to_predict in enumerate(x_list_predict):
        for j in range(len(labels_positions)):
            y_predicted = y[:,j]
            x_for_model = x_to_predict
            for i,model_name in enumerate(models_advanced):
                results_df.loc[methods[k],model_names_advanced[i]][results_df.columns[j]] = CV_sklearn.CrossValidate_sklearn(x_for_model, y_predicted, model= model_name)
    sqrt_df = results_df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
    sqrt_df = np.sqrt(sqrt_df)
    print('end_advanced')
    return results_df,sqrt_df



def test_model(labels_position_to_predict,best_trial_params,X_train, y_train, X_test, y_test,X_set,y_set):
    FC_model = FullyConnectedModel(limit=2000,y_size=y_train.size(1), equal_weights=False,label_to_predict=labels_position_to_predict,max_epoch=TEST_EPOCHS)
    FC_model.set_params(**best_trial_params)
    # print("fit model for the test")
    FC_model.fit(X_train, y_train, X_test, y_test)
    # testing
    # print("testing the model on a given input")
    pred = FC_model.predict(X_set)
    if len(labels_position_to_predict) == 3:
        print('pred:' , pred)
        print('y_set:', y_set[:,[0,6,12]])
    # err = np.sum(np.power(pred - y_set, 2)) / pred.shape[0]
    enc_vecs_AE = FC_model.get_enc_vec(X_set.float())
    enc_vecs_AE = enc_vecs_AE.detach().numpy()
    # for i,label_pos in enumerate(labels_position_to_predict):
    #     x = np.sum(np.power(pred.detach().numpy()[:,i] - (y_set.detach().numpy())[:,label_pos],2))/ pred.detach().numpy().shape[0]
    #     print(x)
    #     print(np.sqrt(x))
    # err = np.sum(np.power(pred.detach().numpy() - y_set.detach().numpy()[:,[labels_position_to_predict]], 2)) / pred.detach().numpy().shape[0]
    CV = CrossValidation(N = X_set.shape[0], folds = 5,label_to_predict=labels_position_to_predict)
    cv_results,cv_results_mtps_ann = CV.CrossValidate(X_set, y_set, [], model=FC_model, plot=False, non_aggregated=True)
    return cv_results_mtps_ann,enc_vecs_AE,FC_model


def calc_mse_train_data_dummy_alg(y,labels_to_predict):
    mses=[]
    for i in labels_to_predict:
        mean_y = np.mean(y[:,i])
        err = np.sum(np.power(mean_y - y[:,i], 2)) / y[:,i].shape[0]
        mses.append(err)
    return(mses)

def print_and_save_outputs(fn,df_basics,r2_basics,df_dim_reduction,r2_dim_reduction,use_case,df_real_mse_yael
                           ,df_real_mse_yael_advanced):
    # fn = r'/Users/matanl/IdeaProjects/Yotam_Thesis/results_models.xlsx'
    writer = pd.ExcelWriter(fn)
    if use_case == 1:
        pos=1
    elif use_case == 2:
        pos=5
    else:
        pos=10

    df_basics.to_excel(writer, startcol=2*pos,startrow=4, header=None, index=False)
    r2_basics.to_excel(writer, startcol=2*pos,startrow=15, header=None, index=False)
    df_dim_reduction.to_excel(writer, startcol=2*pos + 1,startrow=26, header=None, index=False)
    r2_dim_reduction.to_excel(writer, startcol=2*pos + 1,startrow=41, header=None, index=False)
    df_real_mse_yael.to_excel(writer, startcol=10*pos + 1,startrow=26, header=None, index=False)
    df_real_mse_yael_advanced.to_excel(writer, startcol=10*pos + 1,startrow=26, header=None, index=False)
    writer.save()
    print(df_basics,r2_basics,df_dim_reduction,r2_dim_reduction,df_real_mse_yael,df_real_mse_yael_advanced)


def do_pickle(use_case: object, optuna: object, run_hyper_param_tuning: object, pkl_path='') -> object:
    if run_hyper_param_tuning:
        curr_date = datetime.now()
        file = open('important_202' + str(use_case) + '_' + str(curr_date.strftime("%d_%m_%Y_%H_%M_%S")), 'wb')
        pickle.dump(optuna, file)
        # close the file
        file.close()
        infile = open('important_202'+ str(use_case)  + '_' + str(curr_date.strftime("%d_%m_%Y_%H_%M_%S")),'rb')
        best_trial = pickle.load(infile)
        infile.close()
        return best_trial
    else:
        infile = open(pkl_path,'rb')
        best_trial = pickle.load(infile)
        infile.close()
    return best_trial


def get_relevant_data(use_case):
    if use_case == 1: ## (learn from 2020 to 2021)
        X, y, _,scaler_x,scaler_y = get_data(eval=True,  cols_labels=COLUMN_LABELS_2020,data_year=2020,use_case=use_case)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
        X_set, y_set,_,scaler_x,scaler_y = get_data(eval=True,  cols_labels=COLUMN_LABELS_2020,data_year=2021,
                                                    use_case=use_case,mm_scaler_x=scaler_x,
                                                    mm_scaler_y=scaler_y)
        labels_position_to_predict = get_labels_position_to_predict_by_task(COLUMN_LABELS_2020)
    elif use_case == 2: ## (all_data_tgtr)
        X, y, _,scaler_x,scaler_y= get_data(eval=True,X_cols=shared_cols_2020_2022, cols_labels=COLUMN_LABELS_2022,data_year=2022,use_case=use_case)
        X_train, X_test_t, y_train, y_test_t = train_test_split(X, y,test_size=0.3, random_state=42)
        X_test, X_set, y_test, y_set = train_test_split(X_test_t, y_test_t,test_size=0.5, random_state=42)
        labels_position_to_predict = get_labels_position_to_predict_by_task(COLUMN_LABELS_2022)

    elif use_case ==3: ## only 2020
        X, y, _,scaler_x,scaler_y= get_data(eval=True, cols_labels=COLUMN_LABELS_2020,data_year=2020,use_case=use_case)
        X_train, X_test_t, y_train, y_test_t = train_test_split(X, y,test_size=0.3, random_state=42)
        X_test, X_set, y_test, y_set = train_test_split(X_test_t, y_test_t,test_size=0.5, random_state=42)
        labels_position_to_predict = get_labels_position_to_predict_by_task(COLUMN_LABELS_2020)

    elif use_case ==4: ## only 2021
        X, y, _,scaler_x,scaler_y= get_data(eval=True, X_cols=shared_cols_2020_2022, cols_labels=COLUMN_LABELS_2022,data_year=2021,use_case=use_case)
        X_train, X_test_t, y_train, y_test_t = train_test_split(X, y,test_size=0.3, random_state=42)
        X_test, X_set, y_test, y_set = train_test_split(X_test_t, y_test_t,test_size=0.5, random_state=42)
        labels_position_to_predict = get_labels_position_to_predict_by_task(COLUMN_LABELS_2020)

    return(X,y,X_train,y_train,X_test, X_set, y_test, y_set,labels_position_to_predict,scaler_x,scaler_y)


if __name__ ==  '__main__':
    #vars
    ### use_case dict
        # 1 - learn from 2020, predict on 2021
        # 2 - combine 2 years of data
        # 3 - 2020 only
        # 4 - 2021 only
    use_case = 1
    run_hyper_param_tuning= False
    PKL_PATH = '/Users/matanl/IdeaProjects/Yotam_Thesis/important_2021'
    TRAIN_EPOCHS = 30000
    TEST_EPOCHS = 3
    N_TRIALS = 30
    models = [LinearRegression(),LinearRegression(),svm.SVR(),GradientBoostingRegressor(random_state=1241),RandomForestRegressor(random_state=1)]
    model_names= ['Linear Regression_Storage','Linear Regression','SVM','Gradient Boosting','Random Forest','ANN','MTPS-ANN']
    models_advanced = [LinearRegression(),svm.SVR(),GradientBoostingRegressor(random_state=1241),RandomForestRegressor(random_state=1)]
    model_names_advanced = ['Linear Regression','SVM','Gradient Boosting','Random Forest']
    seed=42
    RESULTS_PATH = r'/Users/matanl/IdeaProjects/Yotam_Thesis/results_models.xlsx'
    test_ANN = []

    X,y,X_train,y_train,X_test, X_set, y_test, y_set,labels_position_to_predict,scaler_x,scaler_y=get_relevant_data(use_case=use_case)

    print("Hyper params are: use_case:", use_case, ". Optuna will run:" ,run_hyper_param_tuning, ".  N_optuna:" ,N_TRIALS, "path_topickle:" ,PKL_PATH )
    user_input = input('you sure you want to run this code(yes/no)')
    if user_input.lower() == 'yes':
        print('yalla balagan')
    elif user_input.lower() == 'no':
        exit(1)
    if run_hyper_param_tuning:
        print('optuna will learn')
        hyper_param_optuna = run_optuna(X_train, y_train, trials=N_TRIALS)
    else:
        print('optuna will not learn today')
        hyper_param_optuna = np.NAN #do_pickle
    best_trial_data = do_pickle(pkl_path= PKL_PATH
                                ,use_case=use_case,run_hyper_param_tuning=run_hyper_param_tuning,optuna=hyper_param_optuna)
    best_trial_params = best_trial_data.best_params
    #

    test_results_mtps_ann,enc_vec_model,fc_model = test_model(labels_position_to_predict=labels_position_to_predict,
                                                     best_trial_params=best_trial_params,X_train=X_train, y_train=y_train,
                                                     X_test= X_test, y_test=y_test,X_set=X_set,y_set=y_set)

    for i in range(len(labels_position_to_predict)):
        test_results_ann,_,fc_model_2 = test_model(labels_position_to_predict=[labels_position_to_predict[i]],best_trial_params=best_trial_params,
                                       X_train=X_train, y_train=y_train, X_test= X_test, y_test=y_test,
                                       X_set=X_set,y_set=y_set)
        test_ANN.append(test_results_ann)

    # SHAP
    f = lambda x: fc_model.predict( Variable( torch.from_numpy(x) ) ).detach().numpy()
    data = X_train.numpy()
    explainer = shap.KernelExplainer(f,shap.sample(data,nsamples=100))
    shap_values = explainer.shap_values(data)
    shap_values = explainer.shap_values(X_set,nsamples=100)
    if use_case ==1 or use_case ==3:
        features = shared_cols_2020_2021_no_interuptions
    else: # combined data or 2021 only:
        features = shared_cols_2020_2022
    shap.summary_plot(shap_values, features=data, feature_names=features, plot_type='bar')
    plt.show()

    # shap.summary_plot(shap_values, features=X_scaled, feature_names=f_lst, plot_type='bar') # giovani
    #



    df_results_basic, df_results_advanced = prep_df_for_results(model_names_for_index=model_names,
                                                                model_names_for_index_advanced=model_names_advanced)
    X_np,y_np,CV = get_data_for_baselines(X_base=X_set,y_base=y_set,labels_positions=labels_position_to_predict)
    # test_ANN=[]
    # test_results_mtps_ann=[]
    # enc_vec_model = []
    df_results_basic, df_results_sqrt = run_compare_basic_baselines(models,model_names,X=X_np,y=y_np,CV_sklearn=CV
                                                                    ,results_df=df_results_basic,labels_positions=labels_position_to_predict,
                                                                    ann_res = test_ANN, mtps_ann_res = test_results_mtps_ann)

    # print('Start Run baselines advanced')
    # print('dasdasdasddas')
    df_results_advanced, df_results_sqrt_advanced = run_compare_advanced_baselines(models_advanced=models_advanced,
                                                                                   model_names_advanced=model_names_advanced,
                                                                                   results_df=df_results_advanced,
                                                                                   CV_sklearn=CV,labels_positions=labels_position_to_predict,
                                                                                   X=X_np,x_enc_vec_model=enc_vec_model,X_for_ae=X_set,
                                                                                   y=y_np,n_comp_pca=6)

    mse_train_data_for_r2 = calc_mse_train_data_dummy_alg(y_train.detach().numpy(),labels_to_predict=labels_position_to_predict)
    # print('create r2 metrics')
    r2_basics = 1 - (df_results_basic / mse_train_data_for_r2)
    r2_advanced = 1 - (df_results_advanced / mse_train_data_for_r2)

    # convert this section into function
    real_mse_vectors = []
    real_mse_vectors_advances = []
    for i,label_pos in enumerate(labels_position_to_predict):
        real_mse_vectors.append(scaler_y.data_range_[label_pos] * df_results_basic.iloc[:,i] + scaler_y.data_min_[label_pos])
    df_real_mse = pd.concat(real_mse_vectors,axis=1)

    for j,label_pos in enumerate(labels_position_to_predict):
        real_mse_vectors_advances.append(scaler_y.data_range_[label_pos] * df_results_advanced.iloc[:,j] + scaler_y.data_min_[label_pos])
    df_real_mse_advanced = pd.concat(real_mse_vectors_advances,axis=1)
    # end of real mse section


    print_and_save_outputs(fn=RESULTS_PATH,df_basics=df_results_sqrt,r2_basics=r2_basics,
                           df_dim_reduction=df_results_sqrt_advanced,r2_dim_reduction=r2_advanced
                           ,df_real_mse_yael = df_real_mse, df_real_mse_yael_advanced = df_real_mse_advanced
                           ,use_case=use_case)


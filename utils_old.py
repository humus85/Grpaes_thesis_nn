import numpy as np
import pandas as pd
import os.path
import torch
from torch.autograd import Variable
from sklearn import preprocessing
from config import WHICH_DATA, ROOT_FILES_PATH
import copy

cols = [
    # 'RF_R__T-0',
    # 'FRF_G__T-0',
    # 'FRF_UV__T-0',
    # 'SFR_G__T-0',
    # 'RF_UV__T-0',
    # 'RF_G__T-0',
    'weight__T-0',
    # 'firmness_Distance__T-0',
    # 'VPD (stdev)__T-0',
    # 'YF_G__T-0',
    'week_T-0',
    # 'firmness_Force__T-0',
    # 'SFR_R__T-0',
    # 'ANTH_RB__T-0',
    # 'Tempt(stdev)__T-0',
    # 'RF_B__T-0',
    # 'FRF_R__T-0',
    # 'ANTH_RG__T-0',
    # 'FLAV__T-0',
    # 'firmness_Area__T-0',
    # 'RH (stdev)__T-0',
    # 'FRF_B__T-0',
    'Acceptance_Score__T-0',
    # 'YF_B__T-0',
    # 'firmness_Strain_Height__T-0',
    # 'YF_UV__T-0', 'TSS__T-0',
    'FER_RG__T-0',
    'Rachis_Score__T-0',
    # 'RH__T-0',
    # 'Temp__T-0',
    'FERARI__T-0'
    # ,
    # 'VPD__T-0',
    # 'YF_R__T-0'
] #,'S.R Incidence 11.8.2020', 'S.R Incidence 11.10.2020']


def get_data(which_data = WHICH_DATA, X_cols = cols, eval=True, as_pandas=False, decay_exp_only=False, cols_labels=None, M = 25, N = 2000, var_data = 1, fields = 4, SEED = 235):
    np.random.seed(SEED)
    if which_data == 'SIMULATION':


        # Synthesized ######
        # Data

        X_sample = np.random.rand(N, M)

        # Assign to groups
        # X_sample = np.append(X_sample,np.ceil(np.random.uniform(size=(N,1)) * fields),axis=1)
        vals_for_fields_ratio = np.random.choice(range(1, 30), size=fields)
        sum_vals = np.sum(vals_for_fields_ratio)
        ratios = vals_for_fields_ratio / sum_vals
        X_sample = np.append(X_sample, np.random.choice(range(1,fields+1), size=(N, 1), p=ratios), axis=1)

        # W Parameters (from gaussian)
        mu_orig = np.zeros(M)
        sigma_orig = 1
        W_orig = np.random.normal(mu_orig, sigma_orig)

        # Wf Parameters (from gaussian) and outputs
        y_sample = np.zeros(N)
        Wf_orig = {}
        for i in range(fields):
            X_temp = X_sample[:, :-1][X_sample[:, -1] == i + 1]

            num_rows = X_temp.shape[0]

            Wf_orig[i + 1] = W_orig + np.random.normal(0, 0.1 * i, M)
            # output
            y_sample[[X_sample[:, -1] == i + 1]] = np.dot(X_temp, Wf_orig[i + 1])

        # noise
        epsilon = np.random.normal(0, var_data, N)
        y_sample += epsilon

        indexes = np.arange(N)
        np.random.shuffle(indexes)
        test_indexes = indexes[:N // 2]
        train_indexes = indexes[N // 2:]

        y_test = y_sample[test_indexes]
        X_test = X_sample[test_indexes]

        y_sample = y_sample[train_indexes]
        X_sample = X_sample[train_indexes]

        return X_sample, X_test, y_sample, y_test, fields

    else:
        print('Start preparing data')
        ## initial data
        FILE_PATH = ROOT_FILES_PATH
        if 1==2:
            # os.path.isfile(FILE_PATH + '/X data processed.csv') and os.path.isfile(FILE_PATH + '/y data processed.csv'):
            X = pd.read_csv(FILE_PATH + '/X data processed.csv')
            y = pd.read_csv(FILE_PATH + '/y data processed.csv')
        else:
            print('####')
            FILE_NAME = '/Users/matanl/IdeaProjects/Yotam_Thesis/final_data_net.xlsx'
            # initial_data_ALL = pd.read_csv(FILE_NAME, na_values=0)
            initial_data_ALL = pd.read_excel(FILE_NAME,sheet_name='Sheet3_yotam')
            initial_data_ALL.fillna(0, inplace=True)
            initial_data_ALL['id'] = initial_data_ALL['id'].apply(fix_id)
            if decay_exp_only:
                initial_data_ALL.drop(['S.R Incidence 11.8.2020','S.R Incidence 11.10.2020'], axis=1, inplace=True)

            ## FIRMNESS FILES

            # FILE_NAME = 'A-F firmness {}wks.csv'
            # GROUP = 'A-F'
            # A_F_firmness = create_df(FILE_PATH, GROUP, FILE_NAME)
            #
            # FILE_NAME = 'G-O firmness {}wks.csv'
            # GROUP = 'G-O'
            # G_O_firmness = create_df(FILE_PATH, GROUP, FILE_NAME)
            # firmness = pd.concat((A_F_firmness, G_O_firmness), ignore_index=True)
            #
            # ## Clusters MPX
            # FILE_NAME = 'clusters/A-F clusters {}wks.csv'
            # GROUP = 'A-F'
            # A_F_clusters_multiplex = create_df(FILE_PATH, GROUP, FILE_NAME)
            #
            # FILE_NAME = 'clusters/G-O clusters {}wks.csv'
            # GROUP = 'G-O'
            # G_O_clusters_multiplex = create_df(FILE_PATH, GROUP, FILE_NAME)
            #
            # clusters_MPX = pd.concat((A_F_clusters_multiplex, G_O_clusters_multiplex), ignore_index=True)

            ## Weight and TSS
            # FILE_NAME = '/base/weight TSS.csv'
            # weight_TSS = pd.read_csv(FILE_PATH + FILE_NAME)

            ##
            # data = initial_data_ALL.merge(firmness, on='id')
            # data = data.merge(clusters_MPX,         on='id')
            # data = data.merge(weight_TSS,           on='id')
            # data['week_T-0'] = data['id'].apply(lambda x: int(x[2])*3)


            ##
            T0_cols = set([col for col in list(initial_data_ALL) if 'T-0' in col or col == 'id'])
            Tend_cols = set([col for col in list(initial_data_ALL) if 'T-end' in col or col == 'id'])

            X = id_col_to_front(initial_data_ALL[T0_cols])
            y = id_col_to_front(initial_data_ALL[Tend_cols])
            # y['Acceptance_Score__T-end'] = y['Acceptance_Score__T-end'].str.replace(',', '.').astype('float64')
            y['Acceptance_Score__T-end'] = y['Acceptance_Score__T-end'].astype('float64')
            X.to_csv(FILE_PATH + '/X data processed.csv')
            y.to_csv(FILE_PATH + '/y data processed.csv')

        if decay_exp_only:
            X = X[~X['id'].apply(lambda x: x[0]).isin(['A','B','C','D','E','F'])]
            y = y[~y['id'].apply(lambda x: x[0]).isin(['A','B','C','D','E','F'])]


            #########################################
        ## need to return a list of indexes that cannot be used for test
        X['field'] = X['id'].apply(lambda x: x[0])
        indexes_to_leave_out_of_test = list(X[X['field'] == 'C'].index)

        #########################################


        if as_pandas:
            return X,y

        if cols_labels[0] != 'Acceptance_Score__T-end':
            raise Exception('first column is not Acceptance Score')

        min_max_scaler_y = preprocessing.MinMaxScaler()
        min_max_scaler_x = preprocessing.MinMaxScaler()

        X = X[X_cols]
        x_temp = min_max_scaler_x.fit_transform(X.astype('float64').values.reshape(X.shape[0], -1))
        X_set = Variable(torch.tensor(x_temp))
        y_temp = min_max_scaler_y.fit_transform(y[cols_labels].values)
        y_set = Variable(torch.tensor(y_temp).reshape(y_temp.shape[0], -1))

        if eval:
            return X_set, y_set, indexes_to_leave_out_of_test

        N = x_temp.shape[0]
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        test_ratio = 0.2

        test_indexes = indexes[:int(N * test_ratio)]
        train_indexes = indexes[int(N * test_ratio):]

        y_test_t = y_set[test_indexes]
        X_test_t = X_set[test_indexes]

        y_train_t = y_set[train_indexes]
        X_train_t = X_set[train_indexes]
        print('Finished preparing data')

        return X_train_t, X_test_t, y_train_t, y_test_t, X_set, y_set, indexes_to_leave_out_of_test   #, X, y, fields
        # return X_set, y_set


def calculate_err(preds, y, metric = 'rmse'):
    if metric == 'rmse':
        err = np.sqrt(np.sum(np.power(preds - y, 2)) / preds.shape[0])
    elif metric == 'mse':
        err = np.sum(np.power(preds - y, 2)) / preds.shape[0]

    return err

class Stop():
    def __init__(self, limit):
        # self.prev_err = float('inf')
        self.min_err = float('inf')
        self.best_params = {}
        self.iter = None
        self.num_seq_err_growth = 0
        self.growth_limit = limit
        self.epoch = 0
        self.weights_last_updated = 0

    def stop(self, err, params = None):
        self.epoch += 1
        if err >= self.min_err:
            self.num_seq_err_growth += 1
            if self.num_seq_err_growth == self.growth_limit:
                return True
        else:
            self.weights_last_updated = self.epoch
            self.best_params = copy.deepcopy(params)
            self.num_seq_err_growth = 0
            self.min_err = err
        return False


def predict_per_field(models, X, fields, FIELD_MODEL = False):
    preds = np.zeros(X.shape[0])
    for f in range(1, fields + 1):
        if FIELD_MODEL:
            model = models[f-1]
        else:
            model = models[0]
        field_idxs_bool = get_field_idxs_bool(X, f)
        x_temp = remove_field_col(get_field_rows(X, field_idxs_bool))
        if model.__module__ in ['VariationalBayesModel', 'MixtureModel']:
            preds[[field_idxs_bool]] = model.predict(x_temp, f)
        else:
            preds[[field_idxs_bool]] = model.predict(x_temp)
    return preds


def models_with_fields_separation(base_model, X_train, y_train, num_fields):
    models = []
    for f in range(1, num_fields + 1):
        field_idxs_bool_train = get_field_idxs_bool(X_train, f)
        x_temp_train = remove_field_col(get_field_rows(X_train, field_idxs_bool_train))
        y_temp_train = get_field_rows(y_train, field_idxs_bool_train)
        models.append(base_model().fit(x_temp_train, y_temp_train))
    return models



    reg_preds = np.zeros(N)
    for f in range(1, num_fields + 1):
        field_idxs_bool_train = get_field_idxs_bool(X_train, f)
        x_temp_train = remove_field_col(get_field_rows(X_train, field_idxs_bool_train))
        y_temp_train = get_field_rows(y_train, field_idxs_bool_train)
        reg = LinearRegression().fit(x_temp_train, y_temp_train)

        field_idxs_bool_test = get_field_idxs_bool(X_test, f)
        X_temp_test = remove_field_col(get_field_rows(X_test, field_idxs_bool_test))
        y_temp_test = get_field_rows(y_test, field_idxs_bool_test)
        Reg_preds = reg.predict(X_temp_test)
        reg_preds[[field_idxs_bool_test]] = Reg_preds
        print('### LR ### Field {} MSE: {}'.format(field_num,
                                                   np.sum(np.power(Reg_preds - y_temp_test, 2)) / Reg_preds.shape[0]))


def remove_field_col(X):
    return X[:, :-1]


def get_field_rows(X, field_indexes):
    if len(X.shape) != 1:
        return X[:,:][field_indexes]
    else:
        return X[field_indexes]


def get_field_idxs_bool(X, f):
    return X[:, -1] == f


def df_to_view(df):
    # df = df.T
    def highlight_cells(col):
        color_list = ['background-color: white']*len(col)
        color_list[np.argmin(col)] = 'background-color: yellow'
        return color_list

    f = open('results.html', 'w')
    f.write(df.style.apply(highlight_cells, axis=0).render())
    f.close()
    return df

## for real data prep

def create_id_col(df, week, WITHBOXNUM = False):
    if WITHBOXNUM:
        id_col = df['Population'] + '-' \
                 + df['sample'].astype('int').astype('str') + '-' \
                 + df['Box number'].astype('int').astype('str') + '-' \
                 + df['measure'].astype('int').astype('str')
    else:
        id_col = df['Population'] + '-' \
                 + df['sample'].astype('int').astype('str') + '-' \
                 + df['measure'].astype('int').astype('str')

    bool_dup = id_col.duplicated(keep='first')
    num_dup = np.sum(bool_dup)
    if num_dup > 0:
        raise Exception('id creation failed due to {} duplicated values in file {}: {}'.format(num_dup,
                                                                                               week,
                                                                                               df[bool_dup]))
        # print('id creation failed due to duplicated values')
    return id_col

def prepare_df(df, week, file_name):
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    if 'firmness' in file_name:
        df.rename(columns={'Force 1':'firmness_Force',
                           'Area (Traditional) F-T 1:2':'firmness_Area',
                           'Distance 1':'firmness_Distance',
                           'Strain Height [h]':'firmness_Strain_Height'}, inplace=True)
    elif 'clusters' in file_name:
        pass

    df['id'] = create_id_col(df, week)
    df.drop(columns=['Population', 'sample', 'Box number', 'measure'], inplace=True)
    return df


def id_col_to_front(df):
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('id')))
    return df.loc[:, cols]


def create_df(file_path, group, file_name):
    FILE_PATH = file_path
    GROUP = group
    FILE_NAME = file_name

    FILE_PATH += '/{}/{}'.format(GROUP, FILE_NAME)

    df0 = pd.read_csv(FILE_PATH.format(0))
    df0 = prepare_df(df0, 0, FILE_NAME)
    if group == 'A-F':
        TIME_LINE = [3, 6, 9, 12]
    else:
        TIME_LINE = [9]
    df_list = []
    for week in TIME_LINE:
        df_temp = pd.read_csv(FILE_PATH.format(week))
        df_temp = prepare_df(df_temp, week, FILE_NAME)
        df_list.append(df_temp)

    dfs = pd.concat(df_list)
    df = id_col_to_front(df0.merge(dfs, on='id', suffixes=('__T-0', '__T-end')))

    return df

def fix_id(id):
    id_split = list(id)
    if id_split[0] in ['G','H','I','J','K','L','M','N','O']:
        id_split[2] = '3'
    id = ''.join(id_split)
    return id
##
from utils import get_data, calculate_err, predict_per_field, remove_field_col, models_with_fields_separation, df_to_view
from config import ERROR_NAME, CHROME_PATH, RESULTS_PATH, PRINT_HTML, CHECKPOINT_NAME, HYPER_PARAM_FILE_NAME, COLUMN_LABELS
from FullyConnected import FullyConnectedModel
import numpy as np
import pandas as pd
from CrossValidate import CrossValidation
import torch


# 0.07327234223484994
# [0.06414172053337097, 0.057966675609350204, 0.08715987950563431, 0.08520076423883438, 0.07189267128705978]

# a1, a2,  a3,  a4, a5,  a6, a7,  a8,  a9, a10, a11, lr, n_hidden, n_h_layers, n_enc, act_func = [100, 0, 100, 0, 0, 100, 100, 100, 0, 0, 0,0.0009922088541036275, 102,0, 18, 1]
a1, a2,  a3,  a4, a5,  a6, a7,  a8,  a9, a10, a11, lr, n_hidden, n_h_layers, n_enc, act_func = [26.171208824605728, 1.765816394151588, 1.997139872083553, 11.385641851703953, 22.293602701083515, 21.854462366227068, 5.504063350642566, 8.78952761007893, 24.536193607428288, 0.35025034330666716, 20.340813002614045, 0.001, 102, 0, 18, 1]
# a1, a2,  a3,  a4, a5,  a6, a7,  a8,  a9, a10, a11, lr, n_hidden, n_h_layers, n_enc, act_func = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0.001, 102, 0, 18, 1]



# X, y = get_data(y_cols_labels=COLUMN_LABELS)
X,y = get_data(eval=True, y_cols_labels=COLUMN_LABELS)
# y.to_csv('test.csv')
params = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'a8': a8, 'a9': a9,
          'a10': a10, 'a11': a11, 'lr': lr, 'n_hidden': n_hidden, 'n_h_layers': n_h_layers, 'n_enc': n_enc, 'act_func': act_func}
FC_model = FullyConnectedModel(limit=3000, label_to_predict=[0], equal_weights=False)
FC_model.set_params(**params)
CV = CrossValidation(N = X.shape[0], folds = 5)
print(CV.CrossValidate(X, y, [], model=FC_model, plot=False, non_aggregated=True))
exit()



# X_train_t, X_test_t, y_train_t, y_test_t, X, y = get_data(eval=False, y_cols_labels=COLUMN_LABELS)



params = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'a8': a8, 'a9': a9,
          'a10': a10, 'a11': a11, 'lr': lr, 'n_hidden': n_hidden,'n_h_layers': n_h_layers, 'n_enc': n_enc, 'act_func': act_func}
FC_model = FullyConnectedModel(limit=3000, label_to_predict=[10], equal_weights=False)
FC_model.set_params(**params)
CV = CrossValidation(N = X.size(0), folds = 5, SEED=555)
print(CV.CrossValidate(X, y, [], model=FC_model, plot=False))
print(CV.errors)
exit()

enc_vecs = FC_model.get_enc_vec(X)

enc_vecs_numpy = enc_vecs.data.numpy()
np.savetxt('enc_vecs_newmodel v2.csv', enc_vecs_numpy, delimiter=",")

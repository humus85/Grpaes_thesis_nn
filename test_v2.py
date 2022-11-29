##
from utils import get_data, calculate_err, predict_per_field, remove_field_col, models_with_fields_separation, df_to_view
from config import COLUMN_LABELS,ERROR_NAME, CHROME_PATH, RESULTS_PATH, PRINT_HTML, CHECKPOINT_NAME, HYPER_PARAM_FILE_NAME
from FullyConnected_v2 import FullyConnectedModel
import numpy as np
import pandas as pd
# from CrossValidate import CrossValidation
import torch

X,Y,_ = get_data(eval=True,cols_labels=COLUMN_LABELS)
y = Y.data.numpy()
y = y[:,0]
# print(X,Y)
# exit()

a1, a2,  a3,  a4, a5,  a6, a7,  lr, n_hidden, n_h_layers, n_enc, act_func = [26.171208824605728, 1.765816394151588, 1.997139872083553, 11.385641851703953, 22.293602701083515,  0.35025034330666716, 20.340813002614045, 0.001, 102, 0, 18, 1]

params = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'lr': lr, 'n_hidden': n_hidden,'n_h_layers': n_h_layers, 'n_enc': n_enc, 'act_func': act_func}

iter = 1
predicted_cols = [0,1,6]


# comparison = pd.DataFrame({'columns':[y]})
for i in range(iter):
    print('run: ', i+1)
    FC_model = FullyConnectedModel(limit=1000, label_to_predict=predicted_cols, equal_weights=False)
    FC_model.set_params(**params)
    # CV = CrossValidation(N = X.size(0), folds = 5, SEED = 132)
    # print(CV.CrossValidate(X, y, idx_to_leave_out, model=FC_model, plot=False))
    # print(CV.errors)
    X_train_t, X_test_t, y_train_t, y_test_t, X_set, y_set, indexes_to_leave_out_of_test = get_data(eval=False,cols_labels=COLUMN_LABELS, SEED=(i+1) *3)
    print(indexes_to_leave_out_of_test)
    # exit()


    FC_model.fit(X_train_t, y_train_t, X_test_t, y_test_t)
    # pred = FC_model.predict(X_set)
    if i==0:
        pred = FC_model.predict(X_set)
    else:
        pred_temp =  FC_model.predict(X_set)
        pred += pred_temp

# y_true = y_set[:, 1] * 0.9 + 0.1
y_true = y_set[:, predicted_cols]
pred = pred/iter
# exit()
print('pred:', pred, pred.shape)
print('y_true:', y_true, y_true.shape)
# exit()


pd.concat([pd.DataFrame(pred.data.numpy()),pd.DataFrame(y_true.data.numpy())], axis=1).to_csv('for_y_true.csv')
# print('dasddas')
print(np.mean(np.abs((pred - y_true).data.numpy())))
# print('dasdas')
# exit()
# 0.11751154846314225

# np.savetxt('pred.csv', pred, delimiter=",")
# np.savetxt('yset.csv', y_set[:,10], delimiter=",")

#depends if multiple or 1
ix = torch.tensor(predicted_cols)
partial_y_set = torch.index_select(y_set, 1, ix)

# yotam
# print(np.mean(np.power((pred - y_set[:,0]),2).data.numpy()))
# print(np.std(np.power((pred - y_set[:,0]),2).data.numpy()))
# print(np.std((pred - y_set[:,0]).data.numpy()))

#matan
#mse
print('mse')
print(np.mean(np.power((pred - partial_y_set).detach().numpy(),2)))
print('finish mse')
# if predict multiple at once
print('multiple mses')
first_vec_pred = pred.detach().numpy()[0:1100, 0:1]
sec_vec_pred = pred.detach().numpy()[0:1100, 1:2]
third_vec_pred = pred.detach().numpy()[0:1100, 2:3]
first_vec_partial = partial_y_set.detach().numpy()[0:1100, 0:1]
sec_vec__partial = partial_y_set.detach().numpy()[0:1100, 1:2]
third_vec__partial = partial_y_set.detach().numpy()[0:1100, 2:3]

print(np.mean(np.power((first_vec_pred - first_vec_partial),2)))
print(np.mean(np.power((sec_vec_pred - sec_vec__partial),2)))
print(np.mean(np.power((third_vec_pred - third_vec__partial),2)))
print('multiple mses finish')


print(np.std(np.power((pred - partial_y_set).detach().numpy(),2)))
print(np.std(((pred - partial_y_set).detach().numpy(),2)))


print(np.power((pred - partial_y_set).detach().numpy(),2))



# enc_vecs_AE = FC_model.get_enc_vec(X_set.float())
# enc_vecs_AE = enc_vecs_AE.detach().numpy()

enc_vecs = FC_model.get_enc_vec(X)
print(enc_vecs)
print('size: ', enc_vecs.size())
#
enc_vecs_numpy = enc_vecs.data.numpy()
print('save_txt')
np.savetxt('enc_vecs_baseline_2hl.csv', enc_vecs_numpy, delimiter=",")
print('done')






from matplotlib import pyplot as plt

WHICH_DATA = 'REAL'

COLUMN_LABELS = ['Acceptance_Score__T-end',

                 'Rachis_Score__T-end',
                 'Bleaching__T-end',
                 'Cracks__T-end',
                 'Shatter__T-end',
                 # 'firmness_Area__T-end',
                 # 'firmness_Force__T-end',
                 # 'firmness_Strain_Height__T-end',
                 # 'firmness_Distance__T-end',
                 'weight_loss__T-end',
                 'Decay%__T-end',
                 'disruption_length_T-end',
                 'disruption_temperature_T-end',
                 'kPa_T-end',
                 'firmness_T-end'
                 ]

COLUMN_LABELS_2020 =[
    'Acceptance_Score__T-end','Cluster weight__T-end','Firmness___T-end','FER_RG__T-end','FERARI__T-end',
    'weight_loss__T-end','Rachis_Score__T-end','Bleaching__T-end','Cracks__T-end',
    'Shatter__T-end','Shattering(%)__T-end','Decay__T-end','Decay%__T-end'
    ]

COLUMN_LABELS_2022 =[
    'Acceptance_Score__T-end','Cluster weight__T-end','Firmness___T-end','FER_RG__T-end','FERARI__T-end',
    'weight_loss__T-end','Rachis_Score__T-end','Bleaching__T-end','Cracks__T-end',
    'Shatter__T-end','Shattering(%)__T-end','Decay__T-end','Decay%__T-end',
    'Shriveling__T-end','Berry weight__T-end','TSS__T-end'
]



NUM_ITER_TO_STOP = 10
MIN_ITER_BEFORE_STOP_CHECK = 10
ERROR_NAME = 'rmse'

PRINT_HTML = False
CHROME_PATH = 'open -a /Applications/Google\ Chrome.app %s'
RESULTS_PATH = '/Users/matanl/IdeaProjects/Yotam_Thesis/VB model for grapes/results.html'
ROOT_FILES_PATH = '/Users/matanl/IdeaProjects/Yotam_Thesis/data'

CHECKPOINT_NAME = "./checkpoint.pkl"
HYPER_PARAM_FILE_NAME = "HyperParamResults.pkl"



# create tests and drek
#
# x = np.concatenate(y_set_np[:,[12]]).ravel()
# data_sorted = np.sort(x)
#
# # calculate the proportional values of samples
# p = 1. * np.arange(len(x)) / (len(x) - 1)
#
# # plot the sorted data:
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.plot(p, data_sorted)
# ax1.set_xlabel('$p$')
# ax1.set_ylabel('$x$')
#
# ax2 = fig.add_subplot(122)
# ax2.plot(data_sorted, p)
# ax2.set_xlabel('$x$')
# ax2.set_ylabel('$p$')
# # plt.show()
# fig.show()
#
# from scipy.stats import t
# from scipy.stats import ttest_ind
# # t_stat, p = ttest_ind(y_tr_np[:,[12]],y_set_np[:,[12]])
# # print(f't = {t_stat}, p= {p}')



# sns.displot(data = pz[pz['year']==2020]/2021
#             ,x = 0
#             ,alpha = .7
#             ,hue = 'year'
#             ,kind= "hist",stat='probability')
# plt.show()
# sns.scatterplot(data=all_data, x="new_time", y="decay", hue="year")
# plt.show()
# all_data['new_time'] = np.where(aall_data = pd.concat([dat_2020,dat_2021])ll_data['time']<0.1,3,np.where(all_data['time']<0.4,6,np.where(all_data['time']<0.7,9,12)))
# all_data = pd.concat([dat_2020,dat_2021])

# Shap
# https://stackoverflow.com/questions/70510341/shap-values-with-pytorch-kernelexplainer-vs-deepexplainer
# https://towardsdatascience.com/explainable-ai-xai-with-shap-multi-class-classification-problem-64dd30f97cea
#
# min_max_scaler_y.data_range_[[0,6,12]]
# array([ 4.        ,  0.7       , 80.96470837])
# min_max_scaler_y.data_min_[[0,6,12]]
# array([0. , 0.3, 0. ])
# min_max_scaler_y.data_max_[[0,6,12]]
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from utils import get_data, calculate_err
import numpy as np
import pandas as pd
from config import COLUMN_LABELS_2020,COLUMN_LABELS
from CrossValidate import CrossValidation
from utils import get_labels_position_to_predict_by_task
from scipy import stats
import seaborn as sns


newmodel_errs = [0.018386641517281532, 0.02105877362191677, 0.03821089491248131, 0.0191960372030735, 0.01802411675453186, 0.01213288214057684, 0.004053402692079544, 0.013001748360693455, 0.009336438030004501, 0.013207224197685719, 0.0417875200510025, 0.023752892389893532, 0.011577553115785122, 0.013087510131299496, 0.04771006107330322, 0.006701844744384289, 0.011403223499655724, 0.00869747344404459, 0.014178875833749771, 0.040072962641716, 0.02351865917444229, 0.059169668704271317, 0.0196086298674345, 0.021933654323220253, 0.05341634899377823, 0.036311738193035126, 0.027132071554660797, 0.030639860779047012, 0.04370133578777313, 0.019604098051786423, 0.0018782413098961115, 0.010242175310850143, 0.024711454287171364, 0.03642870858311653, 0.04031899571418762, 0.03988706320524216, 0.035046909004449844, 0.014742521569132805, 0.030522853136062622, 0.030745962634682655, 0.01590183936059475, 0.04682525247335434, 0.017495324835181236, 0.02594403363764286, 0.022321688011288643, 0.031045831739902496, 0.020129017531871796, 0.028895430266857147, 0.020220188423991203, 0.04594350978732109, 0.03439826890826225, 0.017292741686105728, 0.0276733860373497, 0.05082975700497627, 0.02563178911805153, 0.01813170686364174, 0.02898663654923439, 0.028966572135686874, 0.028274374082684517, 0.06693080067634583, 0.009215876460075378, 0.024257581681013107, 0.02081912010908127, 0.02968697063624859, 0.033323150128126144, 0.010433649644255638, 0.03435084968805313, 0.03281671553850174, 0.05118129029870033, 0.018685681745409966, 0.01460541132837534, 0.017710639163851738, 0.008447344414889812, 0.02977464534342289, 0.038417913019657135, 0.02019636705517769, 0.005545758176594973, 0.01924637146294117, 0.031996142119169235, 0.024218624457716942, 0.012066632509231567, 0.01885101944208145, 0.021598923951387405, 0.016057660803198814, 0.020369859412312508, 0.04396970570087433, 0.05179904028773308, 0.04782618582248688, 0.05484601855278015, 0.03283366560935974, 0.021413041278719902, 0.014813314191997051, 0.029827045276761055, 0.009007962420582771, 0.005778271704912186, 0.0496673621237278, 0.024413201957941055, 0.01919608563184738, 0.00838676281273365, 0.016858354210853577]

# data frame for results prep
results_df = pd.DataFrame(columns=['Acceptance Score', 'Rachis Score', 'Decay'],
                          index=['Linear Regression_Storage','Linear Regression','SVM','Gradient Boosting','Random Forest','ANN'
])


results_df_2 = pd.DataFrame(columns=['mse', 'mse pca', 'mse enc', 'mse AE','t_mse'], index=[
                                                                                'Linear Regression_Storage',
                                                                                'Linear Regression',
                                                                                # 'Lasso',
                                                                                # 'Ridge',
                                                                                'SVM',
                                                                                'Gradient Boosting',
                                                                                'Random Forest',
                                                                                'ANN' ])
X_tensor, y_tensor,_ = get_data(cols_labels=COLUMN_LABELS_2020, eval=True,data_year=2020)
labels_position_to_predict = get_labels_position_to_predict_by_task(COLUMN_LABELS_2020)
seed=42
CV = CrossValidation(N = X_tensor.size(0), folds = 5, SEED = seed)
X = X_tensor.data.numpy()
y = y_tensor.data.numpy()
y = y[:,labels_position_to_predict]
# y = y[:,0] # 0, 1, 10
var_y = np.var(y)
# print('y var is: ')

# for_statistical_significance = {}

models = [LinearRegression(),LinearRegression(),svm.SVR(),GradientBoostingRegressor(random_state=1241),RandomForestRegressor(random_state=1),]
model_names= [
    'Linear Regression_Storage',
    'Linear Regression',
    'SVM',
    'Gradient Boosting',
    'Random Forest',
    'ANN'
]

for j in range(len(labels_position_to_predict)):
    y_predicted = y[:,j]
    x_for_model = X
    for i,model_name in enumerate(models):
        if i==0:
            x_for_model= X[:,0]
            x_for_model = x_for_model.reshape(-1, 1)
        else:
            x_for_model = X
        results_df.loc[model_names[i], results_df.columns[j]] = CV.CrossValidate_sklearn(x_for_model, y_predicted, model= model_name)
sqrt_df = results_df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
print(np.sqrt(sqrt_df))

# add ANN run for 1
FC_model = FullyConnectedModel(limit=2000,y_size=y.size(1), equal_weights=False,label_to_predict=labels_position_to_predict,max_epoch=TEST_EPOCHS)
FC_model.set_params(**best_trial_params)
print("fit model for the best")
print(FC_model.fit(X_train, y_train, X_test, y_test))
# testing
print("testing the model on a given input")
pred = FC_model.predict(X_set)

#run this part manually, change the X
LR_model = LinearRegression()
results_df.loc[['Linear Regression_Storage'], ['t_mse']] = CV.CrossValidate_sklearn(X, y, model= LR_model)

# when predicting acceptence keep baseline errs , when other, simply run the model
# baseline_errs = [0.029026866492829817, 0.03514096046911434, 0.03191342486752316, 0.02379804610493338, 0.013102606927209916, 0.01976229225076969, 0.010538651819623528, 0.02034933158026694, 0.010689153984058501, 0.025467241418848884, 0.050460934618878184, 0.059046996383050304, 0.024897521819813594, 0.019691009276143735, 0.055700961798910896, 0.015849728478322094, 0.02557785048088092, 0.03778132849958046, 0.019230506445502286, 0.027132506502679597, 0.016808952196781895, 0.05953741538007121, 0.034524683409108754, 0.05066132010808417, 0.0683173993279079, 0.03563152038222149, 0.03888794808189863, 0.02718291532267889, 0.049920507296627864, 0.016514480353794467, 0.005047398342556077, 0.009143690123320038, 0.049324921516573465, 0.0319329373417972, 0.045642657183676584, 0.04134811118633146, 0.03439146303671374, 0.02222985559446136, 0.03966971696950235, 0.028224015885641898, 0.022563095788429806, 0.04249991651615496, 0.039058793709113505, 0.047629622949953265, 0.03627421103222639, 0.027336630141049194, 0.03457180092105895, 0.05061068016249582, 0.03478551469663232, 0.05211312042080859, 0.043721805908587125, 0.024719555809945924, 0.0383694154617954, 0.04993388287752825, 0.03860101477180107, 0.03417735520382911, 0.030877259497941092, 0.0340004236534153, 0.053158110376549474, 0.07415417541252459, 0.01697355974615287, 0.03520589644062691, 0.02363287119500998, 0.038725013008277846, 0.04151969553562768, 0.009856984816946569, 0.04574764023421219, 0.041026585360154216, 0.0717220338969764, 0.021215846132899972, 0.010035499732063987, 0.019408341274200793, 0.013617419374011166, 0.028947229048427597, 0.05910342692679557, 0.020836890591362307, 0.012838852530976885, 0.02945093281036691, 0.029587867016436693, 0.03068278260478337, 0.012508064050909132, 0.022961105220855076, 0.024708432985612996, 0.016880134702306187, 0.025604148775402947, 0.05382135423451532, 0.05183260593248135, 0.0747065096547468, 0.0656841753090749, 0.03929519429815279, 0.02540809291953437, 0.016391648018876964, 0.04439112495183325, 0.024340842285803323, 0.012091536345988441, 0.03623416236831835, 0.030221753733718788, 0.03494656895277575, 0.007058150823102399, 0.014010524658857009]
# LR_model = LinearRegression()
# results_df.loc[['Linear Regression'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=LR_model)
results_df.loc[['Linear Regression'], ['t_mse']] = CV.CrossValidate_sklearn(X, y, model=LR_model)
# results_df.loc[['Linear Regression'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['Linear Regression'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)


# baseline_errs = [0.05893439693772938, 0.042071302689907525, 0.12444081632653063, 0.09112438317140586, 0.11686285877604645, 0.07975481911717178, 0.07007882267362439, 0.06213659857445322, 0.11340033163493829, 0.06279809063103878, 0.16604681866296636, 0.09072459467348705, 0.09012287247127343, 0.08543725195175265, 0.05505509937977401, 0.11202321988964596, 0.06602021555255326, 0.08918868817317005, 0.05241477337568251, 0.11793577478887879, 0.10312226782558107, 0.08428996650946204, 0.038490854403811746, 0.07365824391926439, 0.0674471355098222, 0.08105257768669294, 0.09907926976936075, 0.08680113112580574, 0.08837747571900449, 0.07651743029440268, 0.030390566828421475, 0.09258953113449984, 0.10827308371943839, 0.09860140669603562, 0.07365824391926439, 0.10540427566529614, 0.06404599678684218, 0.11558836007274596, 0.15833329112229558, 0.142151358705864, 0.05324993277313315, 0.09688589487095262, 0.10709982000384732, 0.11401201547954726, 0.12625357848236488, 0.06685537495000389, 0.12438267023611974, 0.14637969271258325, 0.13016380945880696, 0.0861894472811968, 0.10651806528816549, 0.1416854548036181, 0.14263613025131813, 0.12976563952464437, 0.11236653091288175, 0.10529435763555622, 0.06173842864029065, 0.11793547902722855, 0.09125688114058367, 0.09422594134372075, 0.06883279574184477, 0.06928218704364902, 0.11502648444011362, 0.11137228099875622, 0.10198291789587523, 0.10548097218642002, 0.09587142591038623, 0.12409677922469414, 0.14462463007956916, 0.09632160569324924, 0.05264953337302551, 0.07711944825826654, 0.0550806648868153, 0.06744713550982218, 0.06623538182808732, 0.035710725442794404, 0.07962565772942398, 0.06866054155664475, 0.11202321988964596, 0.07023688614984348, 0.08742277661281, 0.1473343918187777, 0.10372831464814418, 0.10886518424264804, 0.07261753472507877, 0.07773083634122525, 0.08046081712687463, 0.09257218325115588, 0.14061121598867976, 0.09473800876437648, 0.07835248182822951, 0.1321526050487082, 0.0434736399788498, 0.14581548478808046, 0.11073012070366084, 0.08918868817317006, 0.09179883348842538, 0.06701505698997927, 0.06094875283446713, 0.03816776242701686]
# Lasso_model = Lasso(alpha=0.1)
# results_df.loc[['Lasso'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=Lasso_model)
# results_df.loc[['Lasso'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['Lasso'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)


# baseline_errs = [0.027431702360811834, 0.02854532605564689, 0.03179583547042893, 0.021177392937985942, 0.014212223150115956, 0.018527176013139954, 0.009636003641126578, 0.024441533805795872, 0.011862207126419413, 0.02259039427637201, 0.045591083307164146, 0.050534437022391146, 0.0320904316315717, 0.013946959296818127, 0.06096153404691795, 0.019487637306023722, 0.02197615674597109, 0.041633451300708064, 0.019662557743456495, 0.029124723723536153, 0.021968386211463423, 0.06397404066093375, 0.033419493234936586, 0.04542426282233866, 0.06273444551484027, 0.041526162963441374, 0.037967423220939545, 0.0299181116869536, 0.05187716121546687, 0.020869693640896612, 0.005373482049147036, 0.011987320767490576, 0.04594972726778449, 0.031197542143030128, 0.04472833431746521, 0.046095260628443824, 0.03413375923050667, 0.02061129966651946, 0.035200532116150915, 0.038125745979920356, 0.023570823706288886, 0.04268103998340708, 0.03208384175684661, 0.05246315761086828, 0.031074625388929758, 0.02862684869206005, 0.03222176525814261, 0.047165089068678195, 0.04156796972224769, 0.05458026988973511, 0.04547836285291968, 0.03717332718583013, 0.038987400101908855, 0.054422545232783906, 0.036166543033920286, 0.037634676075340615, 0.03748890326917695, 0.03398971612154122, 0.05045113288170692, 0.06779701804777728, 0.015764299709458095, 0.031249501503390016, 0.022931755014011884, 0.03455896354144593, 0.04458190210616383, 0.009014085574033283, 0.04493514404300804, 0.04475985320756367, 0.07785427194790982, 0.023641986179169832, 0.011362034396830565, 0.020210342682803195, 0.014727654592714785, 0.03067694367743969, 0.056807493938757364, 0.02133303945416239, 0.013476039725700455, 0.030298077310944513, 0.029656858538586503, 0.03369674082062041, 0.011519576912301908, 0.027460553834780896, 0.026097044732699824, 0.01756065999699625, 0.01998505986253924, 0.05033730240187676, 0.046267752490514395, 0.06688487047369253, 0.06139941802218469, 0.042486064230567194, 0.023919081838821576, 0.014613057445376727, 0.03800259551604327, 0.02629086932729774, 0.011521226479367057, 0.046283386298888486, 0.030501190424727036, 0.03313269276136667, 0.008311982218447367, 0.018412573587249046]
# Ridge_model = Ridge(alpha=0.5)
# results_df.loc[['Ridge'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=Ridge_model)
# results_df.loc[['Ridge'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['Ridge'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)



# baseline_errs = [0.020639430922142352, 0.02320236174945302, 0.04879185531813546, 0.01863146009370439, 0.02420098056431725, 0.018857332334984933, 0.005991615523449146, 0.018772344796729717, 0.01611091920178885, 0.024464925687224963, 0.045584517994191026, 0.0362772601552361, 0.030864904874513804, 0.025009421672725716, 0.05869407930546834, 0.017667606867311336, 0.01730704878881632, 0.026935292832706088, 0.01932973280454279, 0.04477890242167989, 0.025362355498880954, 0.061730681445459706, 0.024361485215178624, 0.03645674328958359, 0.06277329037009795, 0.04020218441845629, 0.036646882671329026, 0.04168258101202918, 0.046500728739169654, 0.021911905111144173, 0.002297934705560615, 0.013841887836561751, 0.04120662420867819, 0.03316857835536056, 0.04877726373824567, 0.04961048528594947, 0.03567078348455436, 0.010768157988410722, 0.02936376873957773, 0.043073148061678786, 0.020910732863270456, 0.04852991063929841, 0.021119713315767535, 0.04237709172123273, 0.023502271098192532, 0.028829000610290288, 0.02795759546518728, 0.042608336897333086, 0.02177515866429488, 0.06016885350852055, 0.056143971807041684, 0.023199921195555598, 0.0369292714149745, 0.06899779059364028, 0.02748876242095192, 0.01913279511052561, 0.032604623902994206, 0.030728043097562965, 0.03442462562286344, 0.08064857112236862, 0.020339259165281755, 0.031358254985909634, 0.031563115748121005, 0.035092268759383746, 0.043759302368016895, 0.012296845022581097, 0.037763543730696125, 0.05127128676774656, 0.07667821466607669, 0.01933293009485376, 0.009628077271982732, 0.01939679381051998, 0.012302610739860486, 0.034337356721416606, 0.04173544468319191, 0.03825177056526338, 0.01041098672159093, 0.022313612148251942, 0.05124552470930355, 0.030397695839719012, 0.010055145530143049, 0.027826994395520957, 0.027412103881045397, 0.021025463391440544, 0.021918924499894656, 0.05344564045506853, 0.05749209849596304, 0.06779466544548815, 0.05912625791305004, 0.034871472426636255, 0.025935214444206444, 0.017193717658869404, 0.034251836711980205, 0.022368293853183452, 0.0062679901957647865, 0.06352988230751756, 0.032305331652766375, 0.030802409815558946, 0.010982570524244142, 0.017760147814423373]
SVM_model = svm.SVR()
# results_df.loc[['SVM'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=SVM_model)
results_df.loc[['SVM'], ['t_mse']] = CV.CrossValidate_sklearn(X, y, model=SVM_model)
# results_df.loc[['SVM'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['SVM'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)


# baseline_errs = [0.018810617596434546, 0.027440051105870018, 0.03205465535390069, 0.02680797545556679, 0.02578232921245608, 0.020493590320404022, 0.022510656790512302, 0.02109758317702625, 0.015453683345969851, 0.04008777522052747, 0.04375707103553367, 0.05594532795689856, 0.03811864152184409, 0.014696385019060077, 0.04888861631148953, 0.021446630757668273, 0.01695316398616489, 0.017779212758185687, 0.019876892715673954, 0.05518994289478271, 0.03322700246527295, 0.07462280720514144, 0.028740866146658668, 0.0421973384115994, 0.03970441044908151, 0.05435329881151803, 0.02697922574367814, 0.04002521722230396, 0.05658776066155037, 0.025166905876547747, 0.006231619243576419, 0.010315689885615565, 0.024230736034178652, 0.04319638195524817, 0.038890854935962414, 0.04577127833184748, 0.03597507085924286, 0.013101704926103137, 0.038998209676961375, 0.038733861650285875, 0.011643755671138617, 0.045402268722424234, 0.028048438626516244, 0.05542624215810178, 0.034447164916329244, 0.04838384066789666, 0.031241282732681026, 0.03750034657929748, 0.027272280402063295, 0.05952766055527, 0.029980229601939395, 0.022469285975880077, 0.036565990233102835, 0.05203161524894873, 0.025675981902888022, 0.02404306325242828, 0.02966542078349814, 0.033355299256854105, 0.027033804317614516, 0.042660608227619946, 0.014631293790947056, 0.032004251020172556, 0.032107871484691235, 0.03133056072766212, 0.041725760797027056, 0.011454912514784615, 0.0558078521909767, 0.023924709023154786, 0.052648120758109235, 0.015947714214068504, 0.01582780402151092, 0.021129803678836732, 0.01543707566514786, 0.02939879280863579, 0.04747612418704073, 0.025664904683790568, 0.009461573602926505, 0.0250863508673358, 0.03520767865774668, 0.028851465154776348, 0.01925460702137706, 0.04049615807636639, 0.0346670471920904, 0.023495705747645507, 0.02506192311290288, 0.04672904585702743, 0.04555366321968034, 0.05800167169391557, 0.05633650257474946, 0.038140748392531724, 0.017361515593206943, 0.01983460475489655, 0.037467983538469855, 0.010395421890774982, 0.010932258650120529, 0.04845731544387741, 0.02245819212541735, 0.019520655541431682, 0.00493435046187002, 0.018915163137602657]
Gradient_Boosting = GradientBoostingRegressor(random_state=1)
# results_df.loc[['Gradient Boosting'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=Gradient_Boosting)
results_df.loc[['Gradient Boosting'], ['t_mse']] = CV.CrossValidate_sklearn(X, y, model=Gradient_Boosting)
# results_df.loc[['Gradient Boosting'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['Gradient Boosting'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)


# baseline_errs = [0.016379891156462618, 0.025572326530612262, 0.04795033106575965, 0.029923818594104326, 0.03135140136054415, 0.015113124716553275, 0.017502131519274357, 0.021134439909297018, 0.01311814058956915, 0.024135600907029445, 0.04432708390022671, 0.04458487074829932, 0.03188419954648525, 0.017916807256235852, 0.05378735600907025, 0.01210254875283446, 0.016494195011337864, 0.02389047619047618, 0.015070068027210844, 0.051185043083900233, 0.03140141496598642, 0.07366931519274379, 0.026536054421768724, 0.040376063492063484, 0.0286031836734694, 0.037029052154195025, 0.03491434013605439, 0.041756734693877556, 0.04602688435374149, 0.02751376870748297, 0.006078195011337896, 0.009904253968253951, 0.02557019501133784, 0.0320274376417234, 0.0375895782312925, 0.030751074829931963, 0.04009569160997734, 0.012247111111111101, 0.033574984126984106, 0.05514126077097505, 0.02281139229024939, 0.04731302494331065, 0.02475909297052152, 0.02982404535147392, 0.031238394557823124, 0.046757088435374135, 0.022621886621315174, 0.030605886621315186, 0.029203047619047635, 0.04364452607709749, 0.031238448979591844, 0.01940268480725623, 0.03922799092970525, 0.055661097505668895, 0.021501804988662138, 0.030498131519274347, 0.037378403628117916, 0.03280177777777777, 0.028665079365079338, 0.031834702947845794, 0.02065966439909293, 0.03653572789115647, 0.042040290249433086, 0.030920598639455774, 0.036176689342403624, 0.011523773242630351, 0.051114004535147396, 0.012862004535147381, 0.05538593197278909, 0.012549179138322005, 0.010696888888888895, 0.01545351473922902, 0.017339537414966008, 0.028564598639455777, 0.04074635827664399, 0.023035501133786846, 0.00804478911564624, 0.025383482993197298, 0.02928601360544217, 0.03973242630385482, 0.01803515646258501, 0.03141185487528349, 0.036262031746031764, 0.0244132517006803, 0.02230910657596377, 0.04081667120181406, 0.05328879818594107, 0.07105244444444433, 0.060210594104308415, 0.04464185034013605, 0.01622208616780045, 0.024995201814058947, 0.035233052154195046, 0.014578467120181396, 0.003892517006802742, 0.04614335600907028, 0.02488221315192741, 0.027095464852607706, 0.007926004535147389, 0.017735609977324256]
Random_Forest = RandomForestRegressor(random_state=1)
# results_df.loc[['Random Forest'], ['mse']] = CV.CrossValidate_sklearn(X, y, model=Random_Forest)
results_df.loc[['Random Forest'], ['t_mse']] = CV.CrossValidate_sklearn(X, y, model=Random_Forest)
# results_df.loc[['Random Forest'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=Random_Forest)

# results_df.loc[['Random Forest'], ['mse']] = np.mean(baseline_errs)
# pvalues_df.loc[['Random Forest'], ['reg features']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# pca = PCA(n_components=6)
# X_pca = pca.fit_transform(X)
#
# LR_model = LinearRegression()
# results_df.loc[['Linear Regression'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=LR_model)
# baseline_errs = CV.errors
# pvalues_df.loc[['Linear Regression'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# Lasso_model = Lasso(alpha=0.1)
# results_df.loc[['Lasso'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=Lasso_model)
# baseline_errs = CV.errors
# pvalues_df.loc[['Lasso'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# Ridge_model = Ridge(alpha=0.5)
# results_df.loc[['Ridge'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=Ridge_model)
# baseline_errs = CV.errors
# pvalues_df.loc[['Ridge'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# SVM_model = svm.SVR()
# results_df.loc[['SVM'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=SVM_model)
# baseline_errs = CV.errors
# pvalues_df.loc[['SVM'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# Gradient_Boosting = GradientBoostingRegressor(random_state=1)
# results_df.loc[['Gradient Boosting'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=Gradient_Boosting)
# baseline_errs = CV.errors
# pvalues_df.loc[['Gradient Boosting'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

# Random_Forest = RandomForestRegressor(random_state=1)
# results_df.loc[['Random Forest'], ['mse pca']] = CV.CrossValidate_sklearn(X_pca, y, model=Random_Forest)
# baseline_errs = CV.errors
# pvalues_df.loc[['Random Forest'], ['pca']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

## add dimensions of NN
# a1, a2,  a3,  a4, a5,  a6, a7,  a8,  a9, a10, a11, lr, n_hidden, n_h_layers, n_enc, act_func = [26.171208824605728, 1.765816394151588, 1.997139872083553, 11.385641851703953, 22.293602701083515, 21.854462366227068, 5.504063350642566, 8.78952761007893, 24.536193607428288, 0.35025034330666716, 20.340813002614045, 0.001, 102, 0, 18, 1]
# params = {'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4, 'a5': a5, 'a6': a6, 'a7': a7, 'a8': a8, 'a9': a9,
#           'a10': a10, 'a11': a11, 'lr': lr, 'n_hidden': n_hidden,'n_h_layers': n_h_layers, 'n_enc': n_enc, 'act_func': act_func}
# FC_model = FullyConnectedModel(limit=3000
#                                    , label_to_predict=[0,1,10], equal_weights=False) --
# FC_model.set_params(**params)
# print(CV.CrossValidate(X, y, idx_to_leave_out, model=FC_model, plot=False, non_aggregated=True))
# exit()

import torch
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

n_features = X_tensor.size(1)
n_labels = X_tensor.size(1)
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
#
#
enc_vecs_AE = net.extract_enc_layer(X_tensor.float())
enc_vecs_AE = enc_vecs_AE.detach().numpy()
#
# LR_model = LinearRegression()
# results_df.loc[['Linear Regression'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=LR_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Linear Regression'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
#
# Lasso_model = Lasso(alpha=0.1)
# results_df.loc[['Lasso'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=Lasso_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Lasso'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Ridge_model = Ridge(alpha=0.5)
# results_df.loc[['Ridge'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=Ridge_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Ridge'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# SVM_model = svm.SVR()
# results_df.loc[['SVM'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=SVM_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['SVM'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Gradient_Boosting = GradientBoostingRegressor(random_state=1)
# results_df.loc[['Gradient Boosting'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=Gradient_Boosting)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Gradient Boosting'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Random_Forest = RandomForestRegressor(random_state=1)
# results_df.loc[['Random Forest'], ['mse AE']] = CV.CrossValidate_sklearn(enc_vecs_AE, y, model=Random_Forest)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Random Forest'], ['AE']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
#
#
# enc_vecs = np.genfromtxt('/Users/matanl/IdeaProjects/Yotam_Thesis/enc_vecs_baseline_2hl.csv', delimiter=',')
#
# LR_model = LinearRegression()
# results_df.loc[['Linear Regression'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=LR_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Linear Regression'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Lasso_model = Lasso(alpha=0.1)
# results_df.loc[['Lasso'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=Lasso_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Lasso'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Ridge_model = Ridge(alpha=0.5)
# results_df.loc[['Ridge'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=Ridge_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Ridge'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# SVM_model = svm.SVR()
# results_df.loc[['SVM'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=SVM_model)
# baseline_errs = CV.errors
# # pvalues_df.loc[['SVM'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Gradient_Boosting = GradientBoostingRegressor(random_state=1)
# results_df.loc[['Gradient Boosting'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=Gradient_Boosting)
# baseline_errs = CV.errors
# # pvalues_df.loc[['Gradient Boosting'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)
#
# Random_Forest = RandomForestRegressor(random_state=1)
# results_df.loc[['Random Forest'], ['mse enc']] = CV.CrossValidate_sklearn(enc_vecs, y, model=Random_Forest)
# baseline_errs = CV.errors
# pvalues_df.loc[['Random Forest'], ['enc']] = round(stats.ttest_ind(newmodel_errs,baseline_errs).pvalue,4)

print(results_df)
print('')
print(np.sqrt(results_df))
# print(pvalues_df)
print('done')


# errors = []
# folds = KFold(n_splits=5, shuffle=True, random_state=seed)
# for i_f, (trn_idx, val_idx) in enumerate(folds.split(X=X, y=y)):
#     trn_x, trn_y = X[trn_idx], y[trn_idx]
#     val_x, val_y = X[val_idx], y[val_idx]
#     LR_model = LinearRegression()
#     LR_model.fit(trn_x, trn_y)
#     preds = LR_model.predict(val_x)
#     errors.append(calculate_err(preds, val_y))
# print(errors)
# print(np.mean(errors))

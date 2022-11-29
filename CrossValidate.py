import numpy as np
from utils import calculate_err


class CrossValidation:
    """
    Class to test the model. Uses fold to optimize data.

    Parameters
    ----------
    N : integer
        Size of the data
    folds : integer >= 2
        Number of folds
    SEED : number
        Used to randomize the arrangement of folds

    """
    def __init__(self, N, folds, label_to_predict,SEED=235):
        self.errors = []
        self.errors_train = []
        self.idx = {}
        self.prepare = False
        self.seed = SEED
        self.set_folds(N, folds)
        self.single_errors = np.array([0., 0., 0.])
        self.label_to_predict = label_to_predict

    # Unused.
    def set_folds(self, N, folds):
        np.random.seed(self.seed)
        if not isinstance(folds, int) or folds == 1:
            raise Exception('folds must be int >= 2')
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        fold_ratio = 1 / folds
        num_obs_in_fold = int(N * fold_ratio)
        for f in range(folds):
            fold_idx = indexes[f * num_obs_in_fold: (f + 1) * num_obs_in_fold]
            self.idx[f] = {'fold_idxs': fold_idx}
            self.idx[f]['all_the_rest'] = [i for i in indexes if i not in fold_idx]

        for i in range(folds - 1):
            a = set(self.idx[i]['fold_idxs'])
            for j in range(i + 1, folds):
                b = set(self.idx[j]['fold_idxs'])
                if len(a.intersection(b)) != 0:
                    print(i)
                    print(j)
                    print(a.intersection(b))
                    raise Exception('folds intersect!')
        self.prepare = True
        print('VC Verified\n')

    def CrossValidate(self, X, y, indexes_to_leave_out_of_test, model,plot=True ,non_aggregated=False):
        """
        Test model with X and y test data.

        Parameters
        ----------
        X : numpy array
        y : numpy array
        model : model
        indexes_to_leave_out_of_test : list of integers
            Fold indexes to leave out of test.
        plot : boolean
            Whether to plot test losses

        """
        self.errors = []
        self.single_errors = [0., 0., 0.]
        for f in range(len(self.idx)):
            idx_test = list(self.idx[f]['fold_idxs'])
            idx_train = list(self.idx[f]['all_the_rest'])

            ## clean test from extreme RH values
            add_test = []
            add_train = []
            for i, i_test in enumerate(idx_test):
                if i_test in indexes_to_leave_out_of_test:
                    test_index = idx_test.pop(i)
                    add_train.append(test_index)
                    for j, i_train in enumerate(idx_train):
                        if i_train not in indexes_to_leave_out_of_test:
                            train_index = idx_train.pop(j)
                            add_test.append(train_index)
                            break

            idx_test += add_test
            idx_train += add_train

            idx_test = np.array(idx_test)
            idx_train = np.array(idx_train)

            y_test_t = y[idx_test]
            X_test_t = X[idx_test]

            y_train_t = y[idx_train]
            X_train_t = X[idx_train]

            print('Starting fold #{}'.format(f + 1))
            err, single_errors = model.fit(X_train_t, y_train_t, X_test_t, y_test_t, plot=plot)
            # print('in cv')
            # print(single_errors)
            # print(err)
            # print(self.single_errors)
            self.errors.append(err)
            if len(single_errors) == 3:
                self.single_errors += np.array(single_errors)

        mean_err = np.mean(self.errors)

        # if non_aggregated:
        #     r = self.single_errors / 5
        #     print(r)
        #     return r
        print('CV mean error: {}\n\n'.format(mean_err))
        # np.mean(self.single_errors)
        if len(self.label_to_predict)==1:
            return mean_err,np.sqrt(np.sum(self.errors)/5)
        return mean_err,np.sqrt(self.single_errors/5)


    def CrossValidate_sklearn(self, X, y, model, metric='mse'):  # 'mse'
        """
        Test model with X and y test data.

        Parameters
        ----------
        X : numpy array
        y : numpy array
        model : model
        metric : str
            Name of the metric ('rmse' or 'mse') used to compute differences between y and predicted

        """
        self.errors = []
        self.errors_train = []
        for f in range(len(self.idx)):
            idx_test = self.idx[f]['fold_idxs']
            idx_train = self.idx[f]['all_the_rest']

            y_test_t = y[idx_test]
            X_test_t = X[idx_test]

            y_train_t = y[idx_train]
            X_train_t = X[idx_train]

            # print('Starting fold #{}'.format(f + 1))
            model.fit(X_train_t, y_train_t)

            preds = model.predict(X_test_t)

            self.errors.append(calculate_err(preds, y_test_t, metric=metric))
            self.errors_train.append(calculate_err(np.array(len(y_train_t) * [(np.mean(y_train_t))]),y_train_t,metric=metric))

        mean_err = np.mean(self.errors)
        mean_err_train = np.mean(self.errors_train)
        # print('CV mean error: {}\n\n'.format(mean_err))
        return mean_err

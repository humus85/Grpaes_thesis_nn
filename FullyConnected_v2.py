import numpy as np
import torch
from torch.autograd import Variable
from utils import Stop
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_h_layers, n_enc, n_output, act_func=0):
        super(Net, self).__init__()
        torch.manual_seed(1)
        self.hidden_layers = []

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        for layer in range(n_h_layers):
            self.hidden_layers.append(torch.nn.Linear(n_hidden, n_hidden))  # hidden layer*
        self.enc = torch.nn.Linear(n_hidden, n_enc)  # encoder layer
        self.predict = torch.nn.Linear(n_enc, n_output)  # output layer
        func_list = [torch.nn.Tanh, torch.nn.ReLU]
        self.activation_func = func_list[act_func]()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.activation_func(self.hidden(x))  # activation function for hidden layer
        x = self.dropout1(x)
        for h_layer in self.hidden_layers:
            x = self.activation_func(h_layer(x))
        x = self.activation_func(self.enc(x))  # encoder layer
        x = self.predict(x)  # linear output
        return x

    def extract_enc_layer(self, x):
        x = self.activation_func(self.hidden(x))  # activation function for hidden layer
        for h_layer in self.hidden_layers:
            x = self.activation_func(h_layer(x))
        x = self.enc(x)  # encoder layer
        return x


class FullyConnectedModel():
    def __init__(self, limit, y_size, label_to_predict=[0], lr=0.0002, act_func=0, n_h_layers=0, n_hidden=30, n_enc=20,
                 max_epoch=30000, equal_weights=True):

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.net = None

        # init all values to 0 besides 1 to avoid edge case in running.

        self.a = np.zeros(y_size)
        self.a[0] = 1

        self.lr = lr

        self.loss_weights = None
        self.test_losses = []
        self.train_losses = []
        self.limit = limit
        self.equal_weights = equal_weights
        self.n_hidden = n_hidden
        self.n_h_layers = n_h_layers
        self.n_enc = n_enc
        self.act_func = act_func
        self.label_to_predict = label_to_predict
        self.max_epoch = max_epoch

    def set_train_test(self, X_train_t, y_train_t, X_test_t, y_test_t):
        self.X_train = X_train_t
        self.y_train = y_train_t
        self.X_test = X_test_t
        self.y_test = y_test_t

    def create_model(self):

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)

        n_features = self.X_train.size(1)
        n_labels = self.y_train.size(1)
        self.net = Net(n_feature=n_features, n_hidden=self.n_hidden, n_h_layers=self.n_h_layers, n_enc=self.n_enc,
                       act_func=self.act_func, n_output=n_labels)  # define the network
        self.net = self.net.float()
        self.net.apply(init_weights)
        if self.equal_weights:
            self.loss_weights = [1] * n_labels

    def fit(self, X_train_t, y_train_t, X_test_t, y_test_t, eval=True, stop=False, plot=False, trial=None):
        self.set_train_test(X_train_t, y_train_t, X_test_t, y_test_t)
        self.create_model()
        self.Stop = Stop(self.limit)
        self.n_epochs = 0
        self.train_losses = []
        self.test_losses = []
        single_losses = []
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        n_params_to_loss = self.y_train.size(1)

        self.loss_weights = self.a

        last_10000_loss = float('inf')
        while not stop and self.n_epochs <= self.max_epoch:
            self.n_epochs += 1
            self.net.train()
            prediction = self.net(self.X_train.float())  # input x and predict based on x
            optimizer.zero_grad()
            loss = torch.Tensor([0])
            if self.n_epochs % 2 == 0:
                for i in range(n_params_to_loss):
                    a = self.loss_weights[i]
                    if a != 0:
                        loss += a * loss_func(prediction[:, i], self.y_train[:, i].float())
            else:
                loss = 100 * loss_func(prediction[:, self.label_to_predict],
                                       self.y_train[:, self.label_to_predict].float())
            if loss.data.numpy() > 10 ** 4:
                print('loss exploding with value {}'.format(loss))
                print('self losses')
                print(self.test_losses)
                print('single losses')
                print(single_losses)
                if len(self.test_losses) > 1:
                    # return float(np.min(self.test_losses)), single_losses
                    return (0.999, [0.999, 0.999, 0.999])
                else:
                    # return float(loss.data.numpy()), single_losses
                    return (0.999, [0.999, 0.999, 0.999])
            loss.backward()
            optimizer.step()  # apply gradients

            if eval and self.n_epochs > 1:
                self.net.eval()
                with torch.no_grad():
                    # train_loss = loss_func(prediction[:, 0], self.y_train[:, 0].float())
                    # self.train_losses.append(train_loss.data.numpy())
                    pred_test = self.net(self.X_test.float())  # input x and predict based on x
                    eval_loss = loss_func(pred_test[:, self.label_to_predict], self.y_test[:,
                                                                               self.label_to_predict].float())  # must be (1. nn output, 2. target)
                    self.test_losses.append(eval_loss.data.numpy())
                    stop = self.Stop.stop(eval_loss, params=self.net.state_dict())

                    # if trial is not None:
                    #     trial.report(eval_loss, self.n_epochs)
                    #     if trial.should_prune():
                    #         raise optuna.TrialPruned()

            if self.n_epochs % 10000 == 0:
                curr_loss = eval_loss.data.numpy()
                if curr_loss > last_10000_loss:
                    print('interupt run')
                    err = float(np.min(self.test_losses))
                    arg = np.argmin(self.test_losses)
                    print('error: {} (epoch: {})\n'.format(err, arg))

                    return err, single_losses
                else:
                    last_10000_loss = curr_loss
                print('epoch:', self.n_epochs)
                print('eval_loss: ', curr_loss)

        print('Finish training')
        print('n_epochs: {}'.format(self.n_epochs))
        err = float(np.min(self.test_losses))
        arg = np.argmin(self.test_losses)
        print('error: {} (epoch: {})\n'.format(err, arg + 2))
        self.net.load_state_dict(self.Stop.best_params, strict=False)

        for label in self.label_to_predict:
            pred_test = self.net(self.X_test.float())  # input x and predict based on x
            eval_loss = loss_func(pred_test[:, label], self.y_test[:, label].float())
            print('label {} with loss {}'.format(label, eval_loss))
            # print(eval_loss)

            single_losses.append(float(eval_loss.data.numpy()))

        if plot:
            plt.plot(self.test_losses)
            plt.show()

        return err, single_losses

    def get_enc_vec(self, X):
        return self.net.extract_enc_layer(X.float())

    def predict(self, X):
        return self.net(X.float())[:, self.label_to_predict]

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

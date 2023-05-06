import numpy as np

class perceptron:

    def __init__(self, eta, epocs, accerts_rate_stop = 0.95):

        self.eta = eta
        self.epocs = epocs
        self.accerts_rate_stop = accerts_rate_stop
        
    
    # function for training the perceptron
    def train(self, data, labels, verbose = False):
        self.weights_ = 2 * np.random.rand(data.shape[1]) - 0.5 # initialize tetha vector
        self.errors_ = []
        self.accerts_ = []
        
        for _ in range(self.epocs):
            errors = 0
            accerts = 0
            
            for xi, target in zip(data,labels):
                error = target - self.predict(xi)
                self.weights_ +=  0.5 * self.eta * error * xi
                errors += 1 if error != 0.0 else 0
                accerts += 0 if error != 0.0 else 1
            
            self.errors_.append(errors)
            self.accerts_.append(accerts)

            if (accerts/len(labels)) >= self.accerts_rate_stop:
                if verbose:
                    print("Accertion rate stop reached!", "training Accertion-rate : ", accerts/len(labels))
                break
            else:
                if verbose:
                    print("training epoc: ", _, " train accertion-rate : ", accerts/len(labels)," weights: ", self.weights_)
            
        #test_accerts = self.test(data, labels)
        #print("Test accertion-rate: ", test_accerts)
        
        return self

    # activation function
    def predict(self, xi):
        y = np.where(self.z_value(xi) >= 0.0, 1, -1)
        return y

    def z_value(self, xi):
        z = np.dot(xi, self.weights_)
        return z

    # test the perceptron
    def test(self, data, labels):
        accerts = 0
        for xi, target in zip(data,labels):
            accerts += 0 if self.predict(xi) != target else 1
        #print("Accertion-rate : ", accerts/len(labels))
        return accerts/len(labels)
    
    # n-fold cross validation training
    def n_fold_cross_validation(self, data, labels, n = 5, verbose = False):
        # print("N-fold cross validation: ", n, " folds")
        data_folds = np.array_split(data, n)
        labels_folds = np.array_split(labels, n)
        accerts_rates = []
        for i in range(n):
            if verbose:
                print("Validation fold: ", i)
            # get test data
            x_test = data_folds[i]
            y_test = labels_folds[i]
            # get train data
            x_train = np.concatenate(data_folds[:i] + data_folds[i+1:])
            y_train = np.concatenate(labels_folds[:i] + labels_folds[i+1:])
            # train model
            self.train(x_train, y_train, verbose)
            # test model
            accerts_rates.append(self.test(x_test, y_test))
        return sum(accerts_rates)/n
        

    def leave_one_out_cross_validation(self, data, labels, verbose = False):
        # print("Leave-one-out cross validation")
        n = len(data)
        accerts_rates = []
        for i in range(n):
            if verbose:
                print("Validation fold: ", i)
            # get test data
            x_test = data[i].reshape(1,-1)
            y_test = labels[i].reshape(1,-1)
            # get train data
            x_train = np.delete(data, i, 0)
            y_train = np.delete(labels, i, 0)
            # train model
            self.train(x_train, y_train, verbose = False)
            # test model
            accerts_rates.append(self.test(x_test, y_test))
        return sum(accerts_rates)/n

    
import numpy as np

class perceptron:

    def __init__(self, eta, epocs, accerts_rate_stop = 0.95):

        self.eta = eta
        self.epocs = epocs
        self.accerts_rate_stop = accerts_rate_stop
        
    
    # function for training the perceptron
    def train(self, data, labels, verbose = False):
        self.weights_ =2 * np.random.rand(data.shape[1]) - 0.5 # initialize tetha vector
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
            
            # test_accerts = self.test(data, labels)
            # print("Test accertion-rate: ", test_accerts)
        
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
        print("Accertion-rate : ", accerts/len(labels))
        return accerts/len(labels)
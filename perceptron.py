import numpy as np

class perceptron:

    def __init__(self, eta, epocs, accerts_rate_stop = 0.95):

        self.eta = eta;
        self.epocs = epocs;
        self.accerts_rate_stop = 0.95;

    # function for training the perceptron
    def train(self, data, labels):
        self.tetha = np.zeros(data.shape[1]+1) # initialize tetha vector
        self.errors_ = []
        self.accerts_ = []
        
        for _ in range(self.epocs):
            errors = 0;
            for xi, target in zip(data,labels):
                update = self.eta * (target - self.predict(xi))
                self.tetha[1:] += update * xi
                self.tetha[0] += update
                errors += int(update != 0.0)
                accerts = len(labels) - errors
            
            if (accerts/len(labels)) >= self.accerts_rate_stop:
                print("stoping_training")
                break
            else:
                print("training epocs: ", accerts/len(labels))

            self.errors_.append(errors)
            self.accerts_.append(accerts)
        return self

    # función de activacion
    def predict(self, xi):
        y = np.where(self.z_value(xi) >= 0.0, 1, -1)
        # print(y)
        return y

    def z_value(self, xi):
        z = np.dot(xi, self.tetha[1:] + self.tetha[0])
        return z

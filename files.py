

import pandas as pd #importar la libreria panda para  procesar el data set iris.
import numpy as np
class process_file:
    
	def load_file(self,path):
		# Dataset OR/XOR
		data = pd.read_csv(path, header = None)
		return data

	def train_test_split(self, data, test_size = 0.2):
		train_data = data.sample(frac = test_size, random_state = 200)
		test_data = data.drop(train_data.index)

		# split x and y
		x_train = train_data.iloc[:,:-1].values
		y_train = train_data.iloc[:,-1].values
		x_test = test_data.iloc[:,:-1].values
		y_test = test_data.iloc[:,-1].values
		
		# add bias column with -1
		x_train = np.insert(x_train, 2, -1, axis = 1)
		x_test = np.insert(x_test, 2, -1, axis = 1)

		return x_train, y_train, x_test, y_test

from files import process_file
from perceptron import perceptron
from ploting import graph

if __name__ == '__main__':
	epocs = 50 # iterations, default 10000
	eta = 0.0001	 # learning rate factor
	accerts_rate_stop = 0.9999 # stop training when accerts rate is 95%
	split_test_size = 0.2
	verbose = True
	# path to dataset
	#path = "iris.data"
	path = "OR.csv"
	#path = "XOR.csv"
	
	# initialize objects graph and process_file
	obj_plt = graph();
	obj_files = process_file();

	# load data from file
	data = obj_files.load_file(path);

	# split data into train and test
	x_train, y_train, x_test, y_test = obj_files.train_test_split(data, split_test_size); 
	
	#initialize perceptron and train the perceptron
	obj_perceptron = perceptron(eta, epocs, accerts_rate_stop)
	# model = obj_perceptron.train(x_train, y_train, verbose)
	
	# # plot errors
	# obj_plt.plotting_errors(model);

	# #plot decision regions
	# obj_plt.plotting_decision_regions(x_train, y_train, obj_perceptron);

	# # test the perceptron
	# print("Test perceptron with test data")
	# obj_perceptron.test(x_test, y_test);

	# n-fold cross validation training
	rate = obj_perceptron.n_fold_cross_validation(x_train, y_train, 5, verbose = False)
	print("N-folds accertion-rate (n-folds) : ", rate)
	
	# leave-one-out cross validation training
	rate = obj_perceptron.leave_one_out_cross_validation(x_train, y_train, verbose = False)
	print("N-folds accertion-rate (leave-one-out) : ", rate)

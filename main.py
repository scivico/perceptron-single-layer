from files import process_file
from perceptron import perceptron
from ploting import graph

if __name__ == '__main__':
	epocs = 100; # iterations, default 10000
	eta = 0.001;	 # learning rate factor
	accerts_rate_stop = 0.95; # stop training when accerts rate is 95%
	# path to dataset
	#path = "iris.data"
	path = "OR.csv"
	#path = "XOR.csv"
	obj_plt = graph();
	obj_files = process_file();
	data, labels = obj_files.load_file(path);

	#initialize perceptron
	obj_perceptron = perceptron(eta, epocs, accerts_rate_stop)
 
	#training perceptron
	model = obj_perceptron.train(data,labels);

	# plot errors
	obj_plt.plotting_errors(model);

	#plot decision regions
	obj_plt.plotting_decision_regions(data, labels, obj_perceptron);


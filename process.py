from function import *
import multiprocessing as mp
import cma


data, data_type, thickness, data_name = choose_data_type()

def error(X, data=data, data_name=data_name):
	return error_with_type(X, data, data_name)

X_init = code(0, thickness, data_type, data_name)

print("Initial data error:", error(X_init))

is_regression = input("Start the regression process? [y/N] ")
if (is_regression == 'y'):
	# The regression process
	es = cma.CMAEvolutionStrategy(X_init, 1e-3, {'maxiter':25, 'popsize':16})
	es.optimize(error)
	# pool = mp.Pool(4)
	# while not es.stop():
	# 	X = es.ask()
	# 	f_values = pool.map_async(error, X).get()
	# 	es.tell(X, f_values)
	# 	es.disp()
	# 	es.logger.add()
	# print(es.result)
	X_init = es.result[0]
else:
	print("Regression skipped")

print('Total error: %f' % error(X_init))
_, thick, refractive_index_dict = decode(X_init, data_name)

if (data_name == "GST"):
	idx = np.arange(1500, 1605, 5)
	real_data = [np.real(refractive_index_dict[x]) for x in idx]
	real_part, real_model = quadratic_regression(idx, real_data)
	imag_data = [np.imag(refractive_index_dict[x]) for x in idx]
	imag_part, imag_model = quadratic_regression(idx, imag_data)

	print("Writing in result.csv ...")
	f = open("result.csv", "w")
	print("wavelength     n      k", file=f)
	for i in range(1500, 1605, 5):
		print(i, real_model.predict(i)[0][0], imag_model.predict(i)[0][0], file=f)

	print("The results are saved in local folder")
	refractive_index_plot(data_type, refractive_index_dict, real_part, imag_part, data_name)
	plt.show()
elif (data_name == "GLASS"):
	print("Thickness  : %f" % thick)
	print("Ref start  : %f" % refractive_index_dict[0])
	print("Ref end    : %f" % refractive_index_dict[1])
else:
	raise ValueError("No such data type")

from datafunction import *
import multiprocessing as mp
import cma


data, data_type, gst_thick, data_name = choose_data_type()

def error(X, data=data, figure =False):
	phase, thickness, refractive_index_dict = decode(X)
	err = gst_err(data, phase, thickness, refractive_index_dict, figure)
	return err

X_init = code(0, gst_thick, data_type)

# Example code for am20
# X_init = [-0.216612265916, 2.09552424817, 3.92525080068, 3.95864748603, 3.95805034447, 
# 	3.93927916722, 3.93065582, 3.95262397011, 3.93450784342, 3.93068885371, 3.92746652539, 
# 	3.92288996181, 3.90486957481, 3.90584133671, 3.90558117128, 3.88725848778, 3.89906411621, 
# 	3.85864263518, 3.87090210009, 3.8608088514, 3.84260100248, 3.82464454592, 3.84662684536, 
# 	3.8512311621, 3.82104783472, 17.035691462, 17.1964895216, 15.7022924776, 14.3220131596, 
# 	12.9194098609, 11.6369304205, 10.4376367168, 9.36652791739, 8.39375386721, 7.33615279139, 
# 	6.31095057073, 5.35554136807, 4.74367205725, 4.0046485326, 3.28760241159, 2.72587386087, 
# 	2.20820762076, 1.6893025971, 1.38764456712, 0.78928659798, 0.496302260935, 0.263833583658, 
# 	0.146014295431]

is_regression = input("Start the regression process? [y/N] ")
if (is_regression == 'y'):
	# The regression process
	es = cma.CMAEvolutionStrategy(X_init, 1e-2, {'maxiter':20, 'popsize':16})
	es.optimize(error)
	# pool = mp.Pool(4)
	# while not es.stop():
	# 	X = es.ask()
	# 	f_values = pool.map_async(error, X).get()
	# 	es.tell(X, f_values)
	# 	es.disp()
	# 	es.logger.add()
	# es.result_pretty()
	X_init = es.result[5]
else:
	print("Regression skipped")

print('Total error:', error(X_init, figure=False))
_, _, refractive_index_dict = decode(X_init)

idx = np.arange(1500, 1605, 5)
real_data = [np.real(refractive_index_dict[x]) for x in idx]
real_part, real_model = quadratic_regression(idx, real_data)
imag_data = [np.imag(refractive_index_dict[x]) for x in idx]
imag_part, imag_model = quadratic_regression(idx, imag_data)

f = open("result.csv", "w")
print("wavelength     n      k", file=f)
for i in range(1500, 1605, 5):
	print(i, real_model.predict(i)[0][0], imag_model.predict(i)[0][0], file=f)

print("The results are saved in local folder")

refractive_index_plot(data_type, refractive_index_dict, real_part, imag_part, data_name)
plt.show()

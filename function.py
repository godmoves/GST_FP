from __future__ import print_function

from parameter import * 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


# Define constant
PI = 3.14159265
# Add some randomness to init the algorithm
RAND_PERCENT = 0.005 


class Data():
	'''
		Define data type
	'''
	def __init__(self):
		self.loss = None
		self.wavelength = None

def data_import(dataname, transfer=True):
	'''
		Process the data of the second experiment
	'''
	# As the format of data differ from machines, we need do some tansferomation,
	# the format we expect is loss < 0 and wavelength in nanometer.
	if(transfer):
		data = pd.read_csv(dataname+".csv", \
			names=['wavelength', 'loss', 'junkdata'], skiprows=21)
		data.loss = data.loss * -1
		data.wavelength = data.wavelength * 1e9
	else:
		data = pd.read_csv(dataname+".csv", \
			names=['wavelength', 'loss', 'junkdata'])
	newdata = Data()
	newdata.wavelength = np.array(data.wavelength)
	newdata.loss = np.array(data.loss)	
	print("%d data point(s) found" % len(newdata.loss))
	return newdata

def data_plot(data, title='Experimental Data', label=None):
	'''
		Plot the data
	'''
	# plt.figure(figsize=(11, 7))
	plt.title(title)
	plt.xlabel('Wavelength(nm)')
	plt.ylabel('Loss(dB)')
	plt.plot(data.wavelength, data.loss, label=label)

def data_err(data1, data2):
	'''
		Calculate the data loss
	'''
	minlen = len(data1) if (len(data1)<len(data2)) else len(data2)
	err = np.sum((data1[:minlen] - data2[:minlen])**2)/minlen
	print('Data MSE:', err)
	return err

def linear_regression(X_parameters, Y_parameters, figure=False):
	'''
		Create linear regression object
	'''
	X_parameters = np.array(X_parameters).reshape(-1, 1)
	Y_parameters = np.array(Y_parameters).reshape(-1, 1)

	regr = linear_model.LinearRegression()
	regr.fit(X_parameters, Y_parameters)
	Y_mean = np.mean(Y_parameters)
	Y_predict = regr.predict(X_parameters)

	ssr = np.sum((Y_predict - Y_mean) ** 2)
	sst = np.sum((Y_parameters - Y_mean) ** 2)
	predictions = {}
	predictions['intercept'] = regr.intercept_[0]
	predictions['coefficient'] = regr.coef_[0][0]
	predictions['R2'] = ssr/sst

	if(figure):
		plt.title(predictions)
		plt.ylabel('Wavelength(nm)')
		plt.xlabel('Index')
		plt.plot(X_parameters, Y_parameters, '.')
		plt.plot(X_parameters, Y_predict,linewidth=2)

	print(predictions)
	return predictions

def quadratic_regression(X_parameters, Y_parameters):
	'''
		Create quadratic regression object
	'''
	X_parameters=np.array(X_parameters).reshape(-1, 1)
	Y_parameters=np.array(Y_parameters).reshape(-1, 1)

	model = make_pipeline(PolynomialFeatures(2), Ridge())
	model.fit(X_parameters, Y_parameters)
	Y_predict = model.predict(X_parameters)
	return(Y_predict, model)

def trans_matrix(ref, d, lbd):
	'''
		Define the transmission matrix
	'''
	k = 2 * PI * ref / lbd
	e11 = np.cos(k * d)
	e12 = (1 / k) * np.sin(k * d)
	e21 = -k * np.sin(k * d)
	e22 = np.cos(k * d)
	M = np.array([[e11, e12], [e21, e22]])
	return M

def n_glass(lbd, start=1.52909, end=1.52769):
	'''
		The refractive index n of glass
	'''
	k = (start - end) / (1500 - 1600)
	n = start + k * (lbd - 1500)
	return n

def n_gst(lbd, refractive_index_dict):
	'''
		The refrective index of GST
	'''
	t = np.floor((lbd - 1500) / 5)
	t1 = int(1500 + 5 * t)
	t2 = int(t1 + 5)
	N = refractive_index_dict[t1] + \
		(lbd-t1)/5*(refractive_index_dict[t2]-refractive_index_dict[t1])
	return N

def n_gete(lbd, refractive_index_dict):
	return(n_gst(lbd, refractive_index_dict))

def n_aist(lbd, refractive_index_dict):
	return(n_gst(lbd, refractive_index_dict))

def loss(M, ref, d, lbd):
	'''
		Return loss based on Transmission Matrix Theory
	'''
	k = 2 * PI / lbd
	# This part is eliminated when calculate the module
	up = 2 * k # 2j * k * np.exp(complex(0, -k*d)) 
	down = np.sqrt((-M[1, 0] + (k**2)*M[0, 1])**2 + \
		(k*(M[0, 0] + M[1, 1]))**2) 
	t = up / down
	T = (np.absolute(t)) ** 2
	dataloss = 10 * np.log10(T)

	return dataloss

def glass_err(data, phase=0, d=5e-4, ref_s=1.5076, 
		ref_e=1.5064, k=1, figure=False):
	'''
		Calculate the absolute error of glass data
		The value for ref_s & ref_e are predertermined
	'''
	glasserr = []
	newdata = Data()

	for lbd in data.wavelength:
		lbd = (lbd + phase) * 1e-9
		ref = n_glass(lbd, ref_s, ref_e) + k * 1e-5j
		M = trans_matrix(ref, d, lbd)
		pre_loss = loss(M, 1.0, d, lbd)
		glasserr.append(pre_loss)

	if(figure):
		plt.subplot(2, 1, 1)
		plt.title("Regression & error")
		plt.plot(data.wavelength, data.loss)
		plt.plot(data.wavelength, glasserr)
		
		plt.subplot(2, 1, 2)
		plt.plot(data.wavelength, (data.loss - glasserr) ** 2)
	sae = np.sum(np.abs(data.loss-glasserr))

	if(sae < 500):
		print("Phase:{} Thickness:{} MSE:{}".format(phase, d, sae), end='')
		print("Ref_s:{} Ref_e:{} k:{}".format(ref_s, ref_e, k * 1e-5))

	newdata.wavelength = data.wavelength
	newdata.loss = glasserr
	return sae

def gst_err(data, phase, thickness, refractive_index_dict, figure=False):
	'''
    	Calculate the absolute error of gst data
    '''
	loss_list = []
	for i in range(len(data.wavelength)):
		lbd = data.wavelength[i] + phase
		M1 = trans_matrix(n_glass(lbd, 1.50344546427, 1.50444807737), 
			0.000518041252669, lbd*1e-9) # the excat value we use for ref range
		M2 = trans_matrix(n_gst(lbd, refractive_index_dict), 
			thickness*1e-9, lbd*1e-9) 
		M = np.matmul(M2, M1)
		k = 2*PI/(lbd*1e-9)
		up = 2j * k
		down = (-M[1, 0] + (k**2)*M[0, 1] + 1j*k*(M[0, 0]+M[1, 1])) 
		t = up/down
		T = (np.absolute(t))**2
		loss = 10*np.log10(T)
		loss_list.append(loss)
	total_err = np.sum(np.abs(loss_list - data.loss))

	if(figure):
		plt.figure(1)
		plt.title('Regression Line')
		plt.xlabel("wavelength(nm)")
		plt.ylabel("loss(dB)")
		plt.plot(data.wavelength, data.loss, color='blue')
		plt.plot(data.wavelength, loss_list, color='red')
		plt.figure(2)
		plt.title('Regression Error')
		plt.xlabel("wavelength(nm)")
		plt.ylabel("loss error(dB)")
		plt.plot(data.wavelength, (loss_list-data.loss)**2)

	return total_err

def refractive_index_plot(data, regression_data, realpart, imagpart, title):
	'''
		Plot refractive index of gst
	'''
	# The range of wavelength is 1500-1600 nm
	idx = np.arange(1500, 1605, 5)
	fig = plt.figure()

	# intercept = result['intercept']
	# coefficient = result['coefficient']
	# reg = [coefficient*x+intercept for x in idx]

	# draw n & k in the same graph
	ax1 = fig.add_subplot(111)
	ax1.scatter(idx, [np.real(data[x]) for x in idx], label='n', color='black')
	ax1.plot(idx, [np.real(regression_data[x]) for x in idx], '--', 
		label='n', color='black', alpha=0.3)
	ax1.plot(idx, realpart, color='black')
	ax1.set_ylabel('n value')
	ax1.legend(loc=2)
	ax1.set_title("Refractive index of " + title)

	ax2 = ax1.twinx()
	ax2.scatter(idx, [np.imag(data[x]) for x in idx], label='k', color='red')
	ax2.plot(idx, [np.imag(regression_data[x]) for x in idx], '--', 
		label='k', color='red', alpha=0.3)
	ax2.plot(idx, imagpart, color='red')
	ax2.set_ylabel('k value')
	ax2.legend(loc=1)
	ax2.set_xlabel('wavelength')

def code(phase, thickness, refractive_index_dict, data_name):
	'''
		Encode the data to apply the algorithm
		This function deal with wavelength 1500-1600 nm only
	'''
	# The range of wavelength is 1500-1600 nm
	if (data_name == "GST"):
		idx = np.arange(1495, 1610, 5)
		X_len = 1 + 1 + 2 * 23
		X_init = np.zeros(X_len)
		X_init[0] = phase
		X_init[1] = thickness / 10
		X_init[2:25] = [(1-RAND_PERCENT) * np.real(refractive_index_dict[x]) + \
			2*RAND_PERCENT * np.random.rand(1)[0] * np.real(refractive_index_dict[x]) for x in idx]
		X_init[25:48] = [(1-RAND_PERCENT) * np.imag(refractive_index_dict[x]*100) + \
			2*RAND_PERCENT * np.random.rand(1)[0] * np.imag(refractive_index_dict[x]*100) for x in idx]
		return X_init
	elif (data_name == "GLASS"):
		X_len = 1 + 1 + 2
		X_init = np.zeros(X_len)
		X_init[0] = phase
		X_init[1] = thickness * 1e3
		X_init[2] = refractive_index_dict[0]
		X_init[3] = refractive_index_dict[1]
		return X_init
	else:
		raise ValueError("Code error data type")

def decode(X, data_name):
	'''
		Decode the data back
	'''
	if (data_name == "GST"):
		phase = X[0]
		thickness = X[1] * 10
		refractive_index_dict = {}
		for i in range(23):
			refractive_index_dict[1495+5*i] = \
				X[2+i] + X[25+i] * 0.01j
		return phase, thickness, refractive_index_dict
	elif (data_name == "GLASS"):
		phase = X[0]
		thickness = X[1] * 1e-3
		refractive_index_dict = np.zeros(2)
		refractive_index_dict[0] = X[2]
		refractive_index_dict[1] = X[3]
		return phase, thickness, refractive_index_dict
	else:
		raise ValueError("Decode error data type")

def input_data_path():
	'''
		Input data path
	'''
	data_path = input("Input the path to data file: ")
	print("Using data: data/" + data_path + ".csv")
	if_transfer = input("Whether to use transfer: [y/N] ")
	if (if_transfer == 'y'):
		if_transfer = True
		print("Transfer mode: ON")
	else:
		if_transfer = False
		print("Transfer mode: OFF")
	data = data_import(data_path, transfer=if_transfer)
	return data

def input_glass_info():
	'''
		Input information about glass
	'''
	thickness = input("Input the thickness(mm) of glass: ")
	return float(thickness)*1e-3

def input_gst_info():
	'''
		Input information about GST
	'''
	data_type = input("Choose the data type: 1.AM 2.CR (Now work on 1500-1600nm only) ")
	if (data_type == '1'):
		dt = "AM"
	elif (data_type == '2'):
		dt = "CR"
	else:
		raise ValueError("No such data type")
	print("GST phase: " + dt)
	gst_thick = input("Input the thickness(nm) of gst layer: (Now support 20nm and 80nm only) ")
	if (gst_thick != "20" and gst_thick != "80"):
		raise NotImplementedError("Now support 20nm and 80nm only")
	dt = dt + gst_thick
	gst_thick = int(gst_thick)

	return dt, gst_thick


def choose_data_type():
	'''
		User interface function
	'''
	material = input("The material to test: 1.glass 2.GST 3.others ")
	if (material == '1'):
		data = input_data_path()
		data_type = [1.52909, 1.52769]
		glass_thick = input_glass_info()
		data_name = "GLASS"
		return data, data_type, glass_thick, data_name
	elif (material == '2'):
		data = input_data_path()
		dt, gst_thick = input_gst_info()
		data_dict = {"AM20": GST_AM_20, "CR20": GST_CR_20, "CR80": GST_CR_80}
		data_type = data_dict[dt]
		data_name = "GST"
		return data, data_type, gst_thick, data_name
	else:
		raise NotImplementedError("Material not support now")

def error_with_type(X, data, data_name):
	'''
		Return error based on different data types
	'''
	if (data_name == "GST"):
		phase, thickness, refractive_index_dict = decode(X, data_name)
		err = gst_err(data, phase, thickness, refractive_index_dict)
		return err
	elif (data_name == "GLASS"):
		phase, thickness, refractive_index_dict = decode(X, data_name)
		err = glass_err(data, phase, thickness, refractive_index_dict[0], 
			refractive_index_dict[1])
		return err
	else:
		raise NotImplementedError("Error in type not support")
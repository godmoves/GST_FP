import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

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
		data = pd.read_csv(dataname, \
			names=['wavelength', 'loss', 'junkdata'], skiprows=21)
		data.loss = data.loss * -1
		data.wavelength = data.wavelength * 1e9
	else:
		data = pd.read_csv(dataname, \
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

def quartic_regression(X_parameters, Y_parameters):
	'''
		Create quartic regression object
	'''
	X_parameters=np.array(X_parameters).reshape(-1, 1)
	Y_parameters=np.array(Y_parameters).reshape(-1, 1)

	model = make_pipeline(PolynomialFeatures(4), Ridge())
	model.fit(X_parameters, Y_parameters)
	Y_predict = model.predict(X_parameters)
	return(Y_predict, model)


def refractive_index_plot(data, regression_data=None, real_part_smooth=None, 
		imag_part_smooth=None, title="Material"):
	'''
		Plot refractive index of gst
	'''
	# The range of wavelength is 1500-1600 nm
	idx = np.arange(1500, 1605, 5)
	fig = plt.figure()

	# draw n & k in the same graph
	ax1 = fig.add_subplot(111)
	ax1.scatter(idx, [np.real(data[x]) for x in idx], label='n', color='black')
	if not regression_data is None:
		ax1.plot(idx, [np.real(regression_data[x]) for x in idx], '--', 
			label='n', color='black', alpha=0.3)
		ax1.plot(idx, real_part_smooth, color='black')
	ax1.set_ylabel('n value')
	ax1.legend(loc=2)
	ax1.set_title("Refractive index of " + title)

	ax2 = ax1.twinx()
	ax2.scatter(idx, [np.imag(data[x]) for x in idx], label='k', color='red')
	if not regression_data is None:
		ax2.plot(idx, [np.imag(regression_data[x]) for x in idx], '--', 
			label='k', color='red', alpha=0.3)
		ax2.plot(idx, imag_part_smooth, color='red')
	ax2.set_ylabel('k value')
	ax2.legend(loc=1)
	ax2.set_xlabel('wavelength')

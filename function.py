import numpy as np
import matplotlib.pyplot as plt

from parameters import *


# Define constant
PI = 3.14159265
# Add some randomness to init the algorithm
RAND_PERCENT = 0.005


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


def n_gst_like(lbd, refractive_index_dict):
    '''
            The refrective index of GST/GeTe/AIST
    '''
    t = np.floor((lbd - 1500) / 5)
    t1 = int(1500 + 5 * t)
    t2 = int(t1 + 5)
    N = refractive_index_dict[t1] + \
        (lbd-t1)/5*(refractive_index_dict[t2]-refractive_index_dict[t1])
    return N


def loss(M, ref, d, lbd):
    '''
            Return loss based on Transmission Matrix Theory
    '''
    k = 2 * PI / lbd
    # This part is eliminated when calculate the module
    up = 2 * k  # 2j * k * np.exp(complex(0, -k*d))
    down = np.sqrt((-M[1, 0] + (k**2)*M[0, 1])**2 +
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

    newdata.wavelength = data.wavelength
    newdata.loss = glasserr
    return sae


def gst_like_err(data, phase, thickness, refractive_index_dict, figure=False):
    '''
    Calculate the absolute error of gst data
'''
    loss_list = []
    for i in range(len(data.wavelength)):
        lbd = data.wavelength[i] + phase
        M1 = trans_matrix(n_glass(lbd, 1.50344546427, 1.50444807737),
                          0.000518041252669, lbd*1e-9)  # the excat value we use for ref range
        M2 = trans_matrix(n_gst_like(lbd, refractive_index_dict),
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
        plt.title("Regression Result")
        plt.xlabel("wavelength(nm)")
        plt.ylabel("loss(dB)")
        plt.plot(data.wavelength, data.loss,
                 color='blue', label='original data')
        plt.plot(data.wavelength, loss_list,
                 color='red', label='regression data')
        plt.legend(loc="best")

        plt.figure(2)
        plt.title("Regression Error")
        plt.xlabel("wavelength(nm)")
        plt.ylabel("loss error(dB)")
        plt.plot(data.wavelength, (loss_list-data.loss)**2)

    return total_err


def error_with_type(X, data, data_name):
    '''
            Return error based on different data types
    '''
    if (data_name == "GST"
            or data_name == "GeTe"
            or data_name == "AIST"):
        phase, thickness, refractive_index_dict = decode(X, data_name)
        err = gst_like_err(data, phase, thickness, refractive_index_dict)
        return err
    elif (data_name == "GLASS"):
        phase, thickness, refractive_index_dict = decode(X, data_name)
        err = glass_err(data, phase, thickness, refractive_index_dict[0],
                        refractive_index_dict[1])
        return err
    else:
        raise NotImplementedError("Error in type not support")


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
        X_init[2:25] = [(1-RAND_PERCENT) * np.real(refractive_index_dict[x]) +
                        2*RAND_PERCENT * np.random.rand(1)[0] * np.real(refractive_index_dict[x]) for x in idx]
        X_init[25:48] = [(1-RAND_PERCENT) * np.imag(refractive_index_dict[x]*100) +
                         2*RAND_PERCENT * np.random.rand(1)[0] * np.imag(refractive_index_dict[x]*100) for x in idx]
    elif (data_name == "GeTe" or data_name == "AIST"):
        idx = np.arange(1495, 1610, 5)
        X_len = 1 + 1 + 2 * 23
        X_init = np.zeros(X_len)
        X_init[0] = phase
        X_init[1] = thickness / 10
        X_init[2:25] = [(1-RAND_PERCENT) * np.real(refractive_index_dict[x]) +
                        2*RAND_PERCENT * np.random.rand(1)[0] * np.real(refractive_index_dict[x]) for x in idx]
        X_init[25:48] = [(1-RAND_PERCENT) * np.imag(refractive_index_dict[x]) +
                         2*RAND_PERCENT * np.random.rand(1)[0] * np.imag(refractive_index_dict[x]) for x in idx]
    elif (data_name == "GLASS"):
        X_len = 1 + 1 + 2
        X_init = np.zeros(X_len)
        X_init[0] = phase
        X_init[1] = thickness * 1e3
        X_init[2] = refractive_index_dict[0]
        X_init[3] = refractive_index_dict[1]
    else:
        raise ValueError("Code error data type")

    return X_init


def decode(X, data_name):
    '''
            Decode the data back
    '''
    if (data_name == "GST"):
        phase = X[0]
        thickness = X[1] * 10
        refractive_index_dict = {}
        for i in range(23):
            refractive_index_dict[1495+5*i] = X[2+i] + X[25+i] * 0.01j
    elif (data_name == "GeTe" or data_name == "AIST"):
        phase = X[0]
        thickness = X[1] * 10
        refractive_index_dict = {}
        for i in range(23):
            refractive_index_dict[1495+5*i] = X[2+i] + X[25+i] * 1j
    elif (data_name == "GLASS"):
        phase = X[0]
        thickness = X[1] * 1e-3
        refractive_index_dict = np.zeros(2)
        refractive_index_dict[0] = X[2]
        refractive_index_dict[1] = X[3]
    else:
        raise ValueError("Decode error data type")

    return phase, thickness, refractive_index_dict

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:40:55 2018

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
import os
import h5py
import time

cdll.LoadLibrary(os.environ['CUDA_PATH_V9_0'] + '\\bin\\cudart64_90.dll') # This is needed by the DLL containing CUDA code.
GPUdll = cdll.LoadLibrary('GPU_acc_recon.dll')

datapath = r'Test_data\04_pr.hdf5'

datafile = h5py.File(datapath, 'r')
data = np.array(datafile['data'][:])

data_slices = data.shape[0]
data_rows = data.shape[1]
data_cols = data.shape[2]

coeff_rows = c_int(0);
coeff_cols = c_int(0);

pattern_arr = c_float*4
pattern = pattern_arr(0.0, 0.0, 16.0, 16.0) 

sigma = 3.2

ss_row = np.zeros(dtype=np.uint16, shape=(data_rows, data_cols))
ss_col = np.zeros(dtype=np.uint16, shape=(data_rows, data_cols))
pinv_im = np.zeros(dtype=np.float32, shape=(data_rows, data_cols))

ss_row_ptr = ss_row.ctypes.data_as(POINTER(c_ubyte))
ss_col_ptr = ss_col.ctypes.data_as(POINTER(c_ubyte))
pinv_im_ptr = pinv_im.ctypes.data_as(POINTER(c_ubyte))

GPUdll.init_sig_extraction(c_int(data_rows), c_int(data_cols), byref(coeff_rows), byref(coeff_cols), ss_row_ptr, ss_col_ptr, pinv_im_ptr, byref(pattern), c_float(sigma))

res = np.zeros(dtype=np.float32, shape=(data_slices, coeff_rows.value, coeff_cols.value))
#resptr = res.ctypes.data_as(POINTER(c_ubyte))


p_resptrarr = []
for ptr in np.arange(0, data_slices):
    resptr = res[ptr].ctypes.data_as(POINTER(c_ubyte))
    p_resptrarr.append(resptr)
c_resptrarr = (POINTER(c_ubyte)*data_slices)(*p_resptrarr)

elapsed = c_float(0)
dataptr_arr = POINTER(c_ubyte)*data_slices

p_dataptrarr = []
for ptr in np.arange(0, data_slices):
    dataptr = data[ptr].ctypes.data_as(POINTER(c_ubyte))
    p_dataptrarr.append(dataptr)
c_dataptrarr = (POINTER(c_ubyte)*data_slices)(*p_dataptrarr)

GPUdll.extract_signal_stack(c_int(data_rows), c_int(data_cols), c_int(data_slices), byref(c_dataptrarr), ss_row_ptr, ss_col_ptr, pinv_im_ptr, byref(c_resptrarr), byref(elapsed))


plt.imshow(data[1], 'gray')
plt.figure()
plt.imshow(res[1], 'gray')

#GPUdll = cdll.LoadLibrary(r'C:\Users\andreas.boden\source\repos\CUDA_test\x64\Release\CUDA_test.dll')

#GPUdll.entryfunc(dataptr, res.ctypes.data_as(POINTER(c_ubyte)), data_rows, data_cols)

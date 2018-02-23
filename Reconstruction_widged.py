# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:56:03 2018

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt
from ctypes import *
import os
import h5py
import time

def load_data(datapath):
    
    datafile = h5py.File(datapath, 'r')
    data = np.array(datafile['data'][:])
    
    return data

def make_3d_ptr_array(in_data):

    data = in_data
    slices = data.shape[0]
    rows = data.shape[1]
    cols = data.shape[2]
            
    pyth_ptr_array = []

    for j in np.arange(0, slices):
        ptr = data[j].ctypes.data_as(POINTER(c_ubyte))
        pyth_ptr_array.append(ptr) 
    c_ptr_array = (POINTER(c_ubyte)*slices)(*pyth_ptr_array)
    
    return c_ptr_array
    
def make_4d_ptr_array(in_data):
    
    data = in_data
    groups = data.shape[0]
    slices = data.shape[1]
    rows = data.shape[2]
    cols = data.shape[3]
    
    pyth_ptr_array = []
    
    for i in np.arange(0, groups):
        temp_p_ptr_array = []
        for j in np.arange(0, slices):
            ptr = data[i][j].ctypes.data_as(POINTER(c_ubyte))
            temp_p_ptr_array.append(ptr) 
        temp_c_ptr_array = (POINTER(c_ubyte)*slices)(*temp_p_ptr_array)
        pyth_ptr_array.append(cast(temp_c_ptr_array, POINTER(c_ubyte)))
    c_ptr_array = (POINTER(c_ubyte)*groups)(*pyth_ptr_array)
    
    return c_ptr_array

def show_im_seq(seq):
    for i in np.arange(seq.shape[0]):
        plt.imshow(seq[i], 'gray')
        plt.pause(0.01)

if __name__ == "__main__":
    cdll.LoadLibrary(os.environ['CUDA_PATH_V9_0'] + '\\bin\\cudart64_90.dll') # This is needed by the DLL containing CUDA code.
    GPUdll = cdll.LoadLibrary('GPU_acc_recon.dll')
    
    datapath = r'Test_data\04_pr.hdf5'
    
    data = load_data(datapath)
    
    data_ptr_array = make_3d_ptr_array(data)
    
    p = c_float*4
    c_pattern = p(0,0,11.53,11.53);
    c_nr_bases = c_int(1)
    s = c_float*c_nr_bases.value
    c_sigmas = s(3.2)
    
    c_grid_rows = c_int(0)
    c_grid_cols = c_int(0)
    
    c_im_rows = c_int(data.shape[1])
    c_im_cols = c_int(data.shape[2])
    c_im_slices = c_int(data.shape[0])
    
    GPUdll.calc_coeff_grid_size(c_im_rows, c_im_cols, byref(c_grid_rows), byref(c_grid_cols), byref(c_pattern))
    
    res_coeffs = np.zeros(dtype=np.float32, shape=(c_nr_bases.value, c_im_slices.value, c_grid_rows.value, c_grid_cols.value))
    res_ptr = make_4d_ptr_array(res_coeffs)
    
    GPUdll.extract_signal(c_im_rows, c_im_cols, c_im_slices, byref(c_pattern), c_nr_bases, byref(c_sigmas), byref(data_ptr_array), byref(res_ptr))

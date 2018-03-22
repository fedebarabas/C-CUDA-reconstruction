# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:56:03 2018

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from ctypes import *
import os
import h5py
import time

class ReconWid(QtGui.QMainWindow):
    def __init__(self, image, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
#        self.img.setPxMode(True)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
#        self.hist.vb.setLimits(yMin=0, yMax=2048)
        imageWidget.addItem(self.hist, row=1, col=2)
        
        layout = QtGui.QGridLayout()
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.cwidget.setLayout(layout)

        layout.setColumnMinimumWidth(0, 350)
        layout.setRowMinimumHeight(0, 550)
        layout.addWidget(imageWidget, 0, 0)
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.img.setImage(image)
        
        
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

def add_grid_of_coeffs(im, coeffs, r0, c0, p):
    print(r0, c0, p)
    im[r0::p,c0::p] = coeffs
    
def reconstruct(coeffs, square_side, row_dir, col_dir):
    im = np.zeros([square_side*np.shape(coeffs)[1], square_side*np.shape(coeffs)[2]])
    print('size of im = ', np.shape(im))
    front_back = row_dir
    for i in np.arange(np.shape(coeffs)[0]):
        step = np.mod(i, square_side)
        if step == 0:
            front_back = 1-front_back
        r0 = (1-col_dir)*int(i/square_side) + col_dir*(square_side-1-int(i/square_side))
        c0 = front_back*step + (1-front_back)*(square_side-1-step)
        
        add_grid_of_coeffs(im, coeffs[i], r0, c0, square_side)
    
    plt.imshow(im,'gray')    
    return im
            

def run_recon(sigmas, mode):
    cdll.LoadLibrary(os.environ['CUDA_PATH_V9_0'] + '\\bin\\cudart64_90.dll') # This is needed by the DLL containing CUDA code.
    GPUdll = cdll.LoadLibrary('GPU_acc_recon.dll')
    
    if not 'data' in globals():
        print('Loading data...')
        datapath = r'Test_data\04_pr.hdf5'
        global data
        data = load_data(datapath)
        data[0] = data[1];
    
    data_ptr_array = make_3d_ptr_array(data)
    
    p = c_float*4
    c_pattern = p(8.8875-1,7.3525-1,11.5307,11.5224); #Minus one due to different (1 or 0) indexing in C/Matlab
    c_nr_bases = c_int(np.size(sigmas))
    s = c_float*c_nr_bases.value
    sigmas = np.array(sigmas, dtype=np.float32)
    c_sigmas = np.ctypeslib.as_ctypes(sigmas) #s(1, 10)
    
    c_grid_rows = c_int(0)
    c_grid_cols = c_int(0)
    
    c_im_rows = c_int(data.shape[1])
    c_im_cols = c_int(data.shape[2])
    c_im_slices = c_int(data.shape[0])
    
    print('Calculating grid...')
    GPUdll.calc_coeff_grid_size(c_im_rows, c_im_cols, byref(c_grid_rows), byref(c_grid_cols), byref(c_pattern))
    
    res_coeffs = np.zeros(dtype=np.float32, shape=(c_nr_bases.value, c_im_slices.value, c_grid_rows.value, c_grid_cols.value))
    res_ptr = make_4d_ptr_array(res_coeffs)
    
    t = time.time()
    if mode == 'cpu':
        print('Extracting signal on CPU...')
        GPUdll.extract_signal_CPU(c_im_rows, c_im_cols, c_im_slices, byref(c_pattern), c_nr_bases, byref(c_sigmas), byref(data_ptr_array), byref(res_ptr))
    elif mode == 'gpu':
        print('Extracting signal on GPU...')
        GPUdll.extract_signal_GPU(c_im_rows, c_im_cols, c_im_slices, byref(c_pattern), c_nr_bases, byref(c_sigmas), byref(data_ptr_array), byref(res_ptr))
    elapsed = time.time() - t
    print('Signal extraction perfomrmed in', elapsed, 'seconds')
    return res_coeffs
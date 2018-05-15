# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:56:03 2018

@author: andreas.boden
"""

import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg
import ctypes
import os
import h5py
import tifffile as tiff
import time

# Recipy from:
# https://code.activestate.com/recipes/460509-get-the-actual-and-usable-sizes-of-all-the-monitor/
user = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [
            ('left', ctypes.c_long),
            ('top', ctypes.c_long),
            ('right', ctypes.c_long),
            ('bottom', ctypes.c_long)
            ]

    def dump(self):
        return map(int, (self.left, self.top, self.right, self.bottom))


def n_monitors():
    retval = []
    CBFUNC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                                ctypes.POINTER(RECT), ctypes.c_double)

    def cb(hMonitor, hdcMonitor, lprcMonitor, dwData):
        r = lprcMonitor.contents
        data = [hMonitor]
        data.append(r.dump())
        retval.append(data)
        return 1
    cbfunc = CBFUNC(cb)
    temp = user.EnumDisplayMonitors(0, 0, cbfunc, 0)

    return len(retval)


class ReconParTree(ParameterTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parameter tree for the reconstruction
        params = [{'name': 'Pixel size', 'type': 'float', 'value': 65, 'suffix': 'nm'},
                  {'name': 'CPU/GPU', 'type': 'list', 'values': ['GPU', 'CPU']},
                {'name': 'Pattern', 'type': 'group', 'children': [
            {'name': 'X-offset', 'type': 'float', 'value': 7.8875, 'limits': (0, 9999)},
            {'name': 'Y-offset', 'type': 'float', 'value': 6.3525, 'limits': (0, 9999)},
            {'name': 'X-period', 'type': 'float', 'value': 11.5307, 'limits': (0, 9999)},
            {'name': 'Y-period', 'type': 'float', 'value': 11.5224, 'limits': (0, 9999)}]},
                {'name': 'Reconstruction options', 'type': 'group', 'children': [
            {'name': 'PSF size', 'type': 'float', 'value': 250, 'limits': (0,9999), 'suffix': 'nm'},
            {'name': 'BG modelling', 'type': 'list', 'values': ['Constant', 'Gaussian'], 'children': [
                    {'name': 'BG Gaussian size', 'type': 'float', 'value': 500, 'suffix': 'nm'}]}]},
                {'name': 'Scanning', 'type': 'group', 'children': [
            {'name': 'Dim 1', 'type': 'list', 'values': ['0', '1']},
            {'name': 'Dim 2', 'type': 'list', 'values': ['0', '1']},
            {'name': 'Unidirection', 'type': 'list', 'values': ['0', '1']},
            {'name': 'Flip row/col', 'type': 'list', 'values': ['0', '1']}]},
            {'name': 'Load data', 'type': 'action'},
            {'name': 'Show pattern', 'type': 'bool'},
            {'name': 'Reconstruct', 'type': 'action'}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True


class ReconWid(QtGui.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.datapath = r'Test_data\04_pr.hdf5'
        self.reconstructor = Reconstructor('GPU_acc_recon.dll')
        self.data_frame = Data_Frame()
        self.recon_frame = Recon_Frame()

        self.partree = ReconParTree()

        parameterFrame = QtGui.QFrame()
        parameterGrid = QtGui.QGridLayout()
        parameterFrame.setLayout(parameterGrid)
        parameterGrid.addWidget(self.partree, 0, 0)

        layout = QtGui.QGridLayout()
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        self.cwidget.setLayout(layout)

        if n_monitors() == 1:
            layout.setColumnMinimumWidth(0, 500)
            layout.setColumnMinimumWidth(1, 1500)
            layout.setRowMinimumHeight(0, 500)
            layout.setRowMinimumHeight(1, 500)

            layout.addWidget(parameterFrame, 0, 0)
            layout.addWidget(self.data_frame, 1, 0)
            layout.addWidget(self.recon_frame, 0, 1, 2, 1)

        pg.setConfigOption('imageAxisOrder', 'row-major')

        self.reconstruct_btn = self.partree.p.param('Reconstruct')
        self.reconstruct_btn.sigStateChanged.connect(self.run_reconstruction)
        self.show_pat_bool = self.partree.p.param('Show pattern')
        self.show_pat_bool.sigStateChanged.connect(self.toggle_pattern)
        self.load_btn = self.partree.p.param('Load data')
        self.load_btn.sigStateChanged.connect(self.load_data)

        self.update_pattern()
        self.partree.p.param('Pattern').sigTreeStateChanged.connect(self.update_pattern)

    def test(self):
        print('Test fcn run')

    def toggle_pattern(self):
        print('Toggling pattern')
        if self.show_pat_bool.value():
            self.data_frame.pattern = self.pattern
            self.data_frame.show_pat = True
        else:
            self.data_frame.show_pat = False

    def update_pattern(self):
        print('Updating pattern')
        pattern_pars = self.partree.p.param('Pattern')
        self.pattern = [pattern_pars.param('X-offset').value(),
           pattern_pars.param('Y-offset').value(),
           pattern_pars.param('X-period').value(),
           pattern_pars.param('Y-period').value()]

        if self.data_frame.show_pat:
            print('Update pattern grid')
            self.data_frame.pattern = self.pattern
            self.data_frame.make_pattern_grid()

    def load_data(self):
        dlg = QtGui.QFileDialog()
        datapath = dlg.getOpenFileName()[0]
        print('Loading data at:', datapath)

        ext = os.path.splitext(datapath)[1]

        if ext in ['.hdf5', '.hdf']:
            with h5py.File(datapath, 'r') as datafile:
                data = np.array(datafile['data'][:])

        elif ext in ['.tiff', '.tif']:
            with tiff.TiffFile(datapath) as datafile:
                data = datafile.asarray()

        self.data_frame.setData(data)

        return data

        print('Data loaded')

    def run_reconstruction(self):
        print('Running reconstruction')
        recon_pars = self.partree.p.param('Reconstruction options')
        sigmas_nm = recon_pars.param('PSF size').value()
        if recon_pars.param('BG modelling').value() == 'Constant':
            sigmas_nm = np.append(sigmas_nm, np.finfo(np.float32).max)
        else:
            print('In Gasussian version')
            sigmas_nm = np.append(sigmas_nm, recon_pars.param('BG modelling').param('BG Gaussian size').value())
            print('Appended to sigmas')

        sigmas = np.divide(sigmas_nm, self.partree.p.param('Pixel size').value())

        scan_par = self.partree.p.param('Scanning')
        dim1 = np.int(scan_par.param('Dim 1').value())
        dim2 = np.int(scan_par.param('Dim 2').value())
        uni_dir = np.int(scan_par.param('Unidirection').value())
        fliprc = np.int(scan_par.param('Flip row/col').value())
        if self.partree.p.param('CPU/GPU').value() == 'CPU':
            self.recon_images = self.reconstructor.reconstruct(self.data_frame.data, sigmas, self.pattern, dim1, dim2, uni_dir, fliprc, 'cpu')
        elif self.partree.p.param('CPU/GPU').value() == 'GPU':
            self.recon_images = self.reconstructor.reconstruct(self.data_frame.data, sigmas, self.pattern, dim1, dim2, uni_dir, fliprc, 'gpu')

        print('Recieved images')
        self.recon_frame.update(self.recon_images[0])


class Data_Frame(QtGui.QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = []
        self.mean_data = []
        self.pattern = []
        self.pattern_grid = []
        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.img_vb = imageWidget.addViewBox(row=0, col=0)
        self.img_vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.img_vb.addItem(self.img)
        self.img_vb.setAspectLocked(True)
        self.img_hist = pg.HistogramLUTItem(image=self.img)
        imageWidget.addItem(self.img_hist, row=0, col=1)

        self.show_mean_btn = QtGui.QPushButton()
        self.show_mean_btn.setText('Show mean image')
        self.show_mean_btn.pressed.connect(self.show_mean)

        frame_label = QtGui.QLabel('Frame # ')
        self.frame_nr = QtGui.QLineEdit('0')
        self.frame_nr.textChanged.connect(self.setImgSlice)
        self.frame_nr.setFixedWidth(45)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(np.shape(self.data)[0])
        self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        self.slider.valueChanged[int].connect(self.slider_moved)

        self.pattern_scatter = pg.ScatterPlotItem()
        self.pattern_scatter.setData(pos=[[0, 0], [10, 10], [20, 20], [30, 30],[40, 40]],
                                     pen=pg.mkPen(color=(255, 0, 0), width=0.5,style=QtCore.Qt.SolidLine, antialias=True),
                                     brush=pg.mkBrush(color=(255, 0, 0), antialias=True),
                                     size=1,
                                     pxMode=False)

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(self.show_mean_btn, 0, 0)
        layout.addWidget(self.slider, 0, 1)
        layout.addWidget(frame_label, 0, 2)
        layout.addWidget(self.frame_nr, 0, 3)
        layout.addWidget(imageWidget, 1, 0, 1, 4)

        self._show_pat = False
        self.pat_grid_made = False

    @property
    def show_pat(self):
        return self._show_pat
    @show_pat.setter
    def show_pat(self, b_value):
        if b_value:
            self._show_pat = True
            print('Showing pattern')
            if not self.pat_grid_made:
                self.make_pattern_grid()
            self.img_vb.addItem(self.pattern_scatter)
        else:
            print('Hiding pattern')
            self._show_pat = False
            self.img_vb.removeItem(self.pattern_scatter)

    def slider_moved(self):
        self.frame_nr.setText(str(self.slider.value()))
        self.setImgSlice()

    def setImgSlice(self):
        try:
            i = int(self.frame_nr.text())
        except TypeError:
            print('ERROR: Input must be an integer value')

        self.slider.setValue(i)
        self.img.setImage(self.data[i])
        self.frame_nr.setText(str(i))

    def show_mean(self):
        self.img.setImage(self.mean_data)

    def setData(self, in_data):
        self.data = in_data
        self.mean_data = np.array(np.mean(self.data, 0), dtype=np.float32)
        self.show_mean()
        self.slider.setMaximum(np.shape(self.data)[0])

    def make_pattern_grid(self):
        #Pattern is now [Row-offset, Col-offset, Row-period, Col-period] where offser is
        #calculated from the upper left corner (0, 0), while the scatter plot
        #plots from lower left corner, so a flip has to be made in columns
        nr_cols = np.size(self.data, 1)
        nr_rows = np.size(self.data, 2)
        nr_points_col = int(1 + np.floor(((nr_cols - 1) - self.pattern[1]) / self.pattern[3]))
        nr_points_row = int(1 + np.floor(((nr_rows - 1) - self.pattern[0]) / self.pattern[2]))
        col_coords = np.linspace(self.pattern[1], self.pattern[1] + (nr_points_col - 1)*self.pattern[3], nr_points_col)
        row_coords = np.linspace(self.pattern[0], self.pattern[0] + (nr_points_row - 1)*self.pattern[2], nr_points_row)
        col_coords = np.repeat(col_coords, nr_points_row)
        row_coords = np.tile(row_coords, nr_points_col)
        self.pattern_grid = [col_coords, row_coords]
        self.pattern_scatter.setData(x=self.pattern_grid[0], y=self.pattern_grid[1])
        self.pat_grid_made = True
        print('Made new pattern grid')


class Recon_Frame(QtGui.QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.img_vb = imageWidget.addViewBox(row=0, col=0)
        self.img_vb.setMouseMode(pg.ViewBox.PanMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
#        self.img.setPxMode(True)
        self.img_vb.addItem(self.img)
        self.img_vb.setAspectLocked(True)
        self.img_hist = pg.HistogramLUTItem(image=self.img)
#        self.hist.vb.setLimits(yMin=0, yMax=2048)
        imageWidget.addItem(self.img_hist, row=0, col=1)

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(imageWidget, 0, 0)

    def update(self, image):
        self.img.setImage(image)


class Reconstructor(object):
    def __init__(self, dll_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This is needed by the DLL containing CUDA code.
#        ctypes.cdll.LoadLibrary(os.environ['CUDA_PATH_V9_0'] + '\\bin\\cudart64_90.dll')
        ctypes.cdll.LoadLibrary(os.path.join(os.getcwd(), 'cudart64_90.dll'))
        self.ReconstructionDLL = ctypes.cdll.LoadLibrary(dll_path)

        self.data_shape_msg = QtGui.QMessageBox()
        self.data_shape_msg.setText("Data does not have the shape of a square scan!")
        self.data_shape_msg.setInformativeText("Do you want to append the data with tha last frame to enable reconstruction?")
        self.data_shape_msg.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

    def reconstruct(self, data, sigmas, pattern, dim1, dim2, uni_dir, fliprc, dev):

        c = self.extract_signal(data, sigmas, pattern, dev)
        print('Back in reconstruct func')
        frames = np.shape(c)[1]
        missing = np.power(np.int(np.ceil(np.sqrt(frames))), 2) - frames;
        print('Calculated missing frames')
        if np.sqrt(frames) != np.round(np.sqrt(frames)):
            if self.data_shape_msg.exec_() == QtGui.QMessageBox.Yes:
                c = np.pad(c, ((0,0), (0,missing), (0,0), (0,0)), 'constant')
                for i in range(0, len(sigmas)):
                    for j in range(0, missing):
                        c[i][-(1+j)] = c[i][-(1+missing)]
                print('Appended data with last frame new shape of c = ', np.shape(c))
        frames = np.shape(c)[1]

        images = [self.coeffs_to_image(c[0],
                                       np.int(np.sqrt(frames)),
                                       dim1,
                                       dim2,
                                       uni_dir,
                                       fliprc) for i in range(0, len(sigmas))]

        return images

    def make_3d_ptr_array(self, in_data):
        print('In make_3D_ptr_array')
        print('In make_3D_ptr_array with data dimensions:', len(np.shape(in_data)))
        assert len(np.shape(in_data)) == 3, 'Trying to make 3D pointer array out of non-3D data'
        data = in_data
        slices = data.shape[0]

        pyth_ptr_array = []

        for j in np.arange(0, slices):
            ptr = data[j].ctypes.data_as(POINTER(c_ubyte))
            pyth_ptr_array.append(ptr)
        c_ptr_array = (POINTER(c_ubyte)*slices)(*pyth_ptr_array)
        print('Finished creating 3D pointer array')
        return c_ptr_array

    def make_4d_ptr_array(self, in_data):
        assert len(np.shape(in_data)) == 4, 'Trying to make 4D pointer array out of non-4D data'
        data = in_data
        groups = data.shape[0]
        slices = data.shape[1]

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

    def add_grid_of_coeffs(self, im, coeffs, r0, c0, p):
        im[r0::p,c0::p] = coeffs

    def coeffs_to_image(self, coeffs, square_side, row_dir, col_dir, uni_dir, fliprc):
        im = np.zeros([square_side*np.shape(coeffs)[1], square_side*np.shape(coeffs)[2]])
        if uni_dir:
            for i in np.arange(np.shape(coeffs)[0]):
                step = np.mod(i, square_side)

                r0 = (col_dir)*int(i/square_side) + (1-col_dir)*(square_side-1-int(i/square_side))
                c0 = row_dir*step + (1-row_dir)*(square_side-1-step)

                if fliprc:
                    self.add_grid_of_coeffs(im, coeffs[i], c0, r0, square_side)
                else:
                    self.add_grid_of_coeffs(im, coeffs[i], r0, c0, square_side)
        else:
            front_back = row_dir
            for i in np.arange(np.shape(coeffs)[0]):
                step = np.mod(i, square_side)
                if step == 0:
                    front_back = 1-front_back
                r0 = (col_dir)*int(i/square_side) + (1-col_dir)*(square_side-1-int(i/square_side))
                c0 = front_back*step + (1-front_back)*(square_side-1-step)

                if fliprc:
                    self.add_grid_of_coeffs(im, coeffs[i], c0, r0, square_side)
                else:
                    self.add_grid_of_coeffs(im, coeffs[i], r0, c0, square_side)

        return im


    def extract_signal(self, data, sigmas, pattern, dev):

        data_ptr_array = self.make_3d_ptr_array(data)
        p = c_float*4
        c_pattern = p(pattern[0], pattern[1], pattern[2], pattern[3]); #Minus one due to different (1 or 0) indexing in C/Matlab
        c_nr_bases = c_int(np.size(sigmas))
        s = c_float*c_nr_bases.value
        print('Sigmas = ', sigmas)
        sigmas = np.array(sigmas, dtype=np.float32)
        c_sigmas = np.ctypeslib.as_ctypes(sigmas) #s(1, 10)
        c_grid_rows = c_int(0)
        c_grid_cols = c_int(0)

        c_im_rows = c_int(data.shape[1])
        c_im_cols = c_int(data.shape[2])
        c_im_slices = c_int(data.shape[0])

        self.ReconstructionDLL.calc_coeff_grid_size(c_im_rows, c_im_cols, byref(c_grid_rows), byref(c_grid_cols), byref(c_pattern))
        res_coeffs = np.zeros(dtype=np.float32, shape=(c_nr_bases.value, c_im_slices.value, c_grid_rows.value, c_grid_cols.value))
        res_ptr = self.make_4d_ptr_array(res_coeffs)
        t = time.time()
        if dev == 'cpu':
            self.ReconstructionDLL.extract_signal_CPU(c_im_rows, c_im_cols, c_im_slices, byref(c_pattern), c_nr_bases, byref(c_sigmas), byref(data_ptr_array), byref(res_ptr))
        elif dev == 'gpu':
            self.ReconstructionDLL.extract_signal_GPU(c_im_rows, c_im_cols, c_im_slices, byref(c_pattern), c_nr_bases, byref(c_sigmas), byref(data_ptr_array), byref(res_ptr))
        elapsed = time.time() - t
        print('Signal extraction performed in', elapsed, 'seconds')
        return res_coeffs

class pattern_finder(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_pattern(self, image):
        nr_rows = np.size(image, 1)
        nr_cols = np.size(image, 2)
        r = np.linspace(0, nr_cols-1, nr_cols)
        c = np.linspace(0, nr_rows-1, nr_rows)
        self.cm, self.rm = np.meshgrid(c, r)

    def make_ref(self, pattern):
        ref = np.cos(self.cm)*np.cos(self.rm)
        plt.imshow(ref)

def show_im_seq(seq):
    for i in np.arange(seq.shape[0]):
        plt.imshow(seq[i], 'gray')
        plt.pause(0.01)

if __name__ == "__main__":

    wid = ReconWid()
    wid.show()
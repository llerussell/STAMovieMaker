# STA Movie Maker
# Lloyd Russell 2017

from GUI import GUI
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer, QRectF, QUrl
from PyQt5.QtWidgets import (QComboBox, QCheckBox, QLineEdit, QSpinBox,
							 QDoubleSpinBox, QFileDialog, QApplication,
							 QDesktopWidget, QMainWindow, QMessageBox)
from PyQt5.QtGui import QColor, QIcon, QPalette, QDesktopServices

import sys
import os
import ctypes
import json
import numpy as np
import difflib
import glob
import colorsys
import scipy
from scipy import io as sio
from skimage.external import tifffile
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import time
import tensorflow as tf

from utils import sta, PrairieLink, paq2py, ThorLink
from skimage import exposure

class Worker(QObject):

	status_signal = pyqtSignal(str, name='statusSignal')
	finished_signal = pyqtSignal()

	def __init__(self, gui_values):
		super().__init__()
		self.p = gui_values

	def work(self):
		error = False

		# make save directory (/STA)
		base_path, file_name = os.path.split(self.p['moviePath'])
		file_name_no_ext = os.path.splitext(file_name)[0]
		save_dir = self.p['savePath']
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		# get params
		sync_frame_channel = self.p['syncFrameChannel']
		sync_stim_channel = self.p['syncStimChannel']
		sync_start = self.p['syncStartSec']
		sync_stop = self.p['syncStopSec']
		num_diff_stims = self.p['numDiffStims']
		start_on_stim = self.p['startOnStim'] - 1
		only_every_x_stims = self.p['everyXStims']
		pre_sec = self.p['preSeconds']
		post_sec = self.p['postSeconds']
		frame_rate = self.p['frameRate']
		avg_image_start_sec = pre_sec + self.p['averageImageStart']
		avg_image_stop_sec = pre_sec + self.p['averageImageStop']
		avg_image_frames = range(int(avg_image_start_sec*frame_rate), int(avg_image_stop_sec*frame_rate))

		if sync_stop == 0:
			sync_stop = np.Inf

		# decide which normalisation methods
		methods = []
		if self.p['methodDF']:
			methods.append('dF')
		if self.p['methodDFF']:
			methods.append('dFF')
		if self.p['methodZscore']:
			methods.append('Zscore')		

		# load sync file
		self.status_signal.emit('Loading sync file')
		sync_ext = os.path.splitext(self.p['syncPath'])[1]
		if sync_ext == '.paq':
			try:
				paq = paq2py.paq_read(self.p['syncPath'], plot=False)
				frame_trace = paq['data'][paq['chan_names'].index(sync_frame_channel)]
				stim_trace = paq['data'][paq['chan_names'].index(sync_stim_channel)]
				rate = paq['rate']

			except:
				self.status_signal.emit('Error. Channel names: ' + str(paq['chan_names']))
				error = True

		elif sync_ext == '.h5':
			sync_data = ThorLink.ReadSyncFile(self.p['syncPath'], 
				datasets=[sync_frame_channel, sync_stim_channel], listdatasets=True)
			frame_trace = sync_data[sync_frame_channel]
			stim_trace = sync_data[sync_stim_channel]
			rate = 250000
		elif sync_ext == '.txt':
			# comma separated list of stim frames (rows can be diff stim types)
			pass

		if not error:
			# load movie
			self.status_signal.emit('Loading movie')
			movie_ext = os.path.splitext(self.p['moviePath'])[1]
			if movie_ext == '.bin':
				movie = PrairieLink.ReadRawFile(self.p['moviePath'])
			elif movie_ext == '.raw':
				movie = ThorLink.ReadRawFile(self.p['moviePath'])
			elif movie_ext == '.tif':
				movie = tifffile.TiffFile(self.p['moviePath'], multifile=True).asarray()


			# get movie dimensions
			num_frames = movie.shape[0]
			frame_dims = movie.shape[1:]

			# get frame times
			frame_times = sta.threshold_detect(frame_trace, 1).astype(np.float) / rate
			frame_times = frame_times[(frame_times > sync_start) & (frame_times < sync_stop)]
			frame_times = frame_times[0:num_frames]
			frame_times_all = frame_times

			# get stim times
			all_stim_times = sta.threshold_detect(stim_trace, 1).astype(np.float) / rate
			all_stim_times = all_stim_times[(all_stim_times > sync_start) & (all_stim_times < sync_stop)]
			all_stim_times = all_stim_times[(all_stim_times-pre_sec > min(frame_times)) & (all_stim_times+post_sec < max(frame_times))]


			# NUMBER OF PLANES HERE
			numZPlanes = self.p['zPlanes']

			for z in range(numZPlanes):
				frame_times = frame_times_all[z::numZPlanes]
				frame_indices = np.arange(z,num_frames,numZPlanes)

				est_frame_rate = np.round(1/np.diff(frame_times).mean())
				if not est_frame_rate == frame_rate:
					self.status_signal.emit('Error. Estimated frame rate: ' + str(est_frame_rate) + ', Specified frame rate: ' + str(frame_rate))
					time.sleep(2)

				# make STA template
				pre_samples = round(pre_sec * frame_rate)
				post_samples = round(post_sec * frame_rate)
				sta_template = np.arange(-pre_samples*numZPlanes, post_samples*numZPlanes, numZPlanes)


				# search for stim order file
				if self.p['useStimOrder']:
					stim_list = sio.loadmat(self.p['stimOrder'])
					stim_order = list(stim_list['oris'])  # this should change!
					stim_order = stim_order[:len(all_stim_times)]
					unique_stims = np.unique(stim_order)
					self.status_signal.emit('Using stim order file')

				# store all average images in dict to combine later
				avg_img = {}
				if methods:
					for method in methods:
						avg_img[method] = np.ndarray([num_diff_stims, frame_dims[0], frame_dims[1]], dtype=np.float32)

				for i in range(num_diff_stims):

					# get stim times
					if self.p['useStimOrder']:
						use_stims = unique_stims[i::num_diff_stims]
						indices = np.in1d(stim_order, use_stims)
						indices = indices[:len(all_stim_times)]
						stim_times = all_stim_times[indices[start_on_stim::only_every_x_stims]]
					else:
						stim_times = all_stim_times[start_on_stim+i::only_every_x_stims]  # start:stop:step
					num_trials = len(stim_times)

					# make and display status message
					msg = 'Plane ' + str(z+1) + ' of ' + str(numZPlanes) + '. Stim ' + str(i+1) + ' of ' + str(num_diff_stims) + ' (' + str(num_trials) + ' trials)'
					self.status_signal.emit(msg)

					# make frame indices
					frames_with_stim = np.searchsorted(frame_times, stim_times)
					frames_with_stim = frame_indices[frames_with_stim]

					all_trials_sta_frames = []
					for stim_frame_idx in frames_with_stim:
						all_trials_sta_frames.append(sta_template + stim_frame_idx)

					# get data
					trials = np.zeros([len(sta_template), frame_dims[0], frame_dims[1], num_trials], dtype=np.float32)
					for j, trial_sta_frames in enumerate(all_trials_sta_frames):
						#print(trial_sta_frames)
						for k, frame_idx in enumerate(trial_sta_frames):
							trials[k,:,:,j] = movie[frame_idx]

					self.status_signal.emit(msg + ' - Raw')

					avg_movie = trials.mean(axis=3)
					save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_raw.tif'
					tifffile.imsave(save_name, avg_movie.astype(np.uint16))

					if methods:
						for method in methods:
							self.status_signal.emit(msg + ' - ' + method)

							if self.p['useSingleTrials']:
								norm_trials = sta.normalise_movie(trials, range(pre_samples), method=method)

								if self.p['doThreshold']:
									norm_trials = sta.threshold(norm_trials, threshold=self.p['threshold'])

								trial_imgs = sta.make_avg_image(norm_trials, avg_image_frames)
								trial_imgs = trial_imgs.transpose([2,0,1])
								save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_' + method + '_All' + str(num_trials) + 'Trials' + '.tif'
								tifffile.imsave(save_name, trial_imgs)
							
							norm_movie = sta.normalise_movie(avg_movie, range(pre_samples), method=method)

							if self.p['doThreshold']:
								norm_movie = sta.threshold(norm_movie, threshold=self.p['threshold'])

							save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_' + method + '.tif'
							tifffile.imsave(save_name, norm_movie)

							avg_img[method][i] = sta.make_avg_image(norm_movie, avg_image_frames)
							save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_' + method + '_AvgImage' + '.tif'
							tifffile.imsave(save_name, avg_img[method][i])

							if self.p['colourByTime']:
								results = sta.colour_by_time(norm_movie, avg_image_frames, smooth=int(frame_rate/2), useCorrelationImage=self.p['useCorrelationImage'], blurHandS=self.p['blurHandS'])
								rgb = results['RGB']
								hsv = results['HSV']
								corr = results['Corr']
								
								save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_' + method + '_TimePeak' + '.tif'
								tifffile.imsave(save_name, rgb)

								save_name = save_dir + os.sep + file_name_no_ext + '_Stim' + str(i+1) + '_STA_Plane' + str(z+1) + '_' + method + '_Corr' + '.tif'
								tifffile.imsave(save_name, (corr*65535).astype(np.uint16))

								# tifffile.imsave(save_name.replace('.tif', '_H.tif'), H)
								# tifffile.imsave(save_name.replace('.tif', '_S.tif'), S)
								# tifffile.imsave(save_name.replace('.tif', '_V.tif'), V)



				if num_diff_stims > 1:
					for method in methods:
						if self.p['makeMaxImage']:
							save_name = save_dir + os.sep + file_name_no_ext + '_' + str(num_diff_stims) + 'Stims' + '_STA_Plane' + str(z+1) + '_' + method + '_MaxResponseImage' + '.tif'
							tifffile.imsave(save_name, avg_img[method].max(axis=0))

						if self.p['makeColourImage']:
							msg = 'Making colour image'
							self.status_signal.emit(msg)

							# calculate indices
							pref_stim = np.argmax(avg_img[method], axis=0).astype(np.int)
							
							# this will only work with 4 visual orientations!
							orth_stim = pref_stim + 2
							orth_stim[orth_stim]
							orth_stim[orth_stim>3] = orth_stim[orth_stim>3]-3;

							x, y = np.meshgrid(np.arange(frame_dims[0]), np.arange(frame_dims[1]))

							pref_img = avg_img[method][pref_stim,y,x]
							pref_img[pref_img<0] = 0
							if num_diff_stims == 4:
								orth_img = avg_img[method][orth_stim,y,x]
								orth_img[orth_img<0] = 0

							other_imgs = avg_img[method].copy()
							other_imgs[pref_stim, y,x] = np.nan
							mean_other_imgs = np.nanmedian(other_imgs, axis=0).astype(np.float32)
							if num_diff_stims == 4:
								mean_other_imgs = orth_img

							# hue
							# from scipy.stats import vonmises
							# data = avg_img[method]
							# blank = np.zeros(frame_dims)
							# for y_px in range(frame_dims[0]):
							# 	for x_px in range(frame_dims[1]):
							# 		this_data = data[:,y_px,x_px]
							# 		print(this_data)
							# 		kappa, loc, scale = vonmises.fit(data[:,y_px,x_px], fscale=1)
							# 		blank[y_px, x_px] = kappa
							

							

							H = pref_stim.astype(np.float32) / num_diff_stims
							H[H==0.0] = 0.025
							H[H==0.75] = 0.8
							# H = H + 0.05
							# H = H/2
							# H = gaussian_filter(H,1)

							# H = blank

							# brightness
							V = pref_img
							# V = V / np.percentile(V, 99)
							if method.lower() == 'df':
								V = V/1000
							elif method.lower() == 'dff':
								V = V/100
							elif method.lower() == 'zscore':
								V = V/5


							V[V<0] = 0
							V[V>1] = 1

							
							if self.p['useCorrelationImage']:
								V = sta.makeCorrImg(sta.downsample(movie[frame_indices],10), 4)
								tifffile.imsave(save_name.replace('.tif', '_Corr.tif'), (V*65535).astype(np.uint16))
							v_min, v_max = np.percentile(V, (1, 99))
							V = exposure.rescale_intensity(V, in_range=(v_min, v_max))
							# V = exposure.equalize_adapthist(V)
							V = V/V.max()


							# saturation
							S = (pref_img - mean_other_imgs) / (pref_img + mean_other_imgs)
							# S[V<np.nanpercentile(V,90)] = S[V<np.nanpercentile(V,90)]/10
							# # S[S > np.percentile(S, 95)] = 0
							# # S = S - np.percentile(S, 1)
							# S = S / np.nanpercentile(S, 90)
							# S = S*V
							S[np.isnan(S)] = 0
							# print(S.max())
							# print(S.min())
							# S = S*2 
							S[S<0] = 0
							S[S>1] = 1
							# S = gaussian_filter(S,1)
							

							# convert HSV to RGB
							# rgb_img = sta.hsv2rgb(H, S, V)
							
							hsv = np.stack((H, S, V), axis=-1)
							hsv_tf = tf.convert_to_tensor(hsv, np.float32)

							rgb_tf = tf.image.hsv_to_rgb(hsv_tf)
							sess = tf.Session()
							with sess.as_default():
								rgb = rgb_tf.eval()

							# blur the rgb image then convert back to hsv and keep the blurred H and S channels.
							if self.p['blurHandS']:
								rgb = gaussian_filter(rgb,(1,1,0))
								rgb_tf = tf.convert_to_tensor(rgb, np.float32)
								hsv_tf = tf.image.rgb_to_hsv(rgb_tf)

								with sess.as_default():
									hsv2 = hsv_tf.eval()

								hsv2[:,:,2] = hsv[:,:,2]
								hsv_tf = tf.convert_to_tensor(hsv2, np.float32)
								rgb_tf = tf.image.hsv_to_rgb(hsv_tf)
								with sess.as_default():
									rgb = rgb_tf.eval()


							rgb_img	= (rgb * 65535).astype(np.uint16)

							# tifffile.imsave(save_name.replace('.tif', 'pref_stim.tif'), pref_stim)
							# tifffile.imsave(save_name.replace('.tif', 'mean_other_imgs.tif'), mean_other_imgs)
							# tifffile.imsave(save_name.replace('.tif', 'H.tif'), H)
							# tifffile.imsave(save_name.replace('.tif', 'S.tif'), S)
							# tifffile.imsave(save_name.replace('.tif', 'V.tif'), (V*65535).astype(np.uint16))

							# for v in range(num_diff_stims):
							# 	idx = np.where(H==(np.float(v)/num_diff_stims))
							# 	# V[idx] = 0.2 * V[idx] / np.mean(V[idx])
							# 	single_img = np.zeros(frame_dims[0]*frame_dims[1], dtype=np.float)
							# 	single_img[idx] = V[idx]
							# 	single_img = np.reshape(single_img, [frame_dims[0],frame_dims[1]])
							# 	save_name = save_dir + os.sep + file_name.replace('.bin', '_' + str(num_diff_stims) + 'stims_STA_PrefImage_' + str(v+1) + '.tif')
							# 	tifffile.imsave(save_name, single_img.astype(np.int16))
							# 	# V[idx] = V[idx] / np.max(V[idx])

							save_name = save_dir + os.sep + file_name_no_ext + '_' + str(num_diff_stims) + 'Stims' + '_STA_Plane' + str(z+1) + '_' + method + '_PrefImage' + '.tif'
							tifffile.imsave(save_name, rgb_img)

							# hsv_img = np.stack([H,S,V], axis=2).astype(np.float16)
							# print(hsv_img.shape)
							# save_name = save_dir + os.sep + file_name_no_ext + '_' + str(num_diff_stims) + 'Stims' + '_STA_Plane' + str(z+1) + '_' + method + '_PrefImage_HSV' + '.tif'
							# tifffile.imsave(save_name, hsv_img)


			self.status_signal.emit('Done')
			if self.p['openResultsDir']:
				QDesktopServices.openUrl(QUrl.fromLocalFile(save_dir));

		self.finished_signal.emit()


class MainWindow(QMainWindow, GUI.Ui_MainWindow):
	'''
	The GUI window
	'''

	def __init__(self):
		QMainWindow.__init__(self)
		self.setupUi(self)

		# set up install paths
		self.install_dir = os.getcwd()
		self.presets_dir = os.path.join(self.install_dir, 'Presets')
		if not os.path.exists(self.presets_dir):
			os.makedirs(self.presets_dir)

		# make worker thread
		self.workerThread = QThread()

		# restore default values
		self.loadDefaults()

		# allow drag and drop
		self.setAcceptDrops(True)

		# signal/slot connections
		self.setConnects()

		# get gui elements
		self.getValues()


	def dragEnterEvent( self, event ):
		data = event.mimeData()
		urls = data.urls()
		if ( urls and urls[0].scheme() == 'file' ):
			event.acceptProposedAction()

	def dragMoveEvent( self, event ):
		data = event.mimeData()
		urls = data.urls()
		if ( urls and urls[0].scheme() == 'file' ):
			event.acceptProposedAction()

	def dropEvent( self, event ):
		data = event.mimeData()
		urls = data.urls()
		if ( urls and urls[0].scheme() == 'file' ):
			# for some reason, this doubles up the intro slash
			if os.name == 'nt':
				filepath = str(urls[0].path()[1:])
			else:
				filepath = str(urls[0].path())
			self.moviePath_lineEdit.setText(filepath)
			self.newMoviePathLoaded(filepath)


	def setConnects(self):
		self.loadMoviePath_pushButton.clicked.connect(self.loadMoviePath)
		self.loadSyncPath_pushButton.clicked.connect(self.loadSyncPath)
		self.loadStimOrderPath_pushButton.clicked.connect(self.loadStimOrderPath)
		self.setSavePath_pushButton.clicked.connect(self.setSavePath)
		self.loadPreset_pushButton.clicked.connect(self.loadPreset)
		self.savePreset_pushButton.clicked.connect(self.savePreset)
		self.setDefaults_pushButton.clicked.connect(self.setDefaults)
		self.run_pushButton.clicked.connect(self.clickRun)

		# auto add connects to update p and trial config plot whenever anything changes
		widgets = (QComboBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox)
		for obj in self.settings_groupBox.findChildren(widgets):
			if isinstance(obj, QComboBox):
				obj.currentIndexChanged.connect(self.getValues)
			if isinstance(obj, QCheckBox):
				obj.stateChanged.connect(self.getValues)
			if isinstance(obj, QLineEdit):
				obj.textChanged.connect(self.getValues)
			if isinstance(obj, QSpinBox):
				obj.valueChanged.connect(self.getValues)
			if isinstance(obj, QDoubleSpinBox):
				obj.valueChanged.connect(self.getValues)


	def getValues(self):
		# extract gui values store in self.p
		widgets = (QComboBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox)
		p = {}
		for obj in self.findChildren(widgets):
			fullname = str(obj.objectName())
			trimmed_name = fullname.split('_')[0]
			if isinstance(obj, QComboBox):
				p[trimmed_name] = str(obj.currentText())
			if isinstance(obj, QCheckBox):
				p[trimmed_name] = bool(obj.isChecked())
			if isinstance(obj, QLineEdit):
				if 'spinbox' not in fullname:
					p[trimmed_name] = str(obj.text())
			if isinstance(obj, QSpinBox):
				p[trimmed_name] = int(obj.value())
			if isinstance(obj, QDoubleSpinBox):
				p[trimmed_name] = float(obj.value())


		# do stuff with parameters:
		if p['frameRate'] > 0:
			step_size = 1/p['frameRate']
			self.preSeconds_doubleSpinBox.setSingleStep(step_size)
			self.postSeconds_doubleSpinBox.setSingleStep(step_size)
			self.averageImageStart_doubleSpinBox.setSingleStep(step_size)
			self.averageImageStop_doubleSpinBox.setSingleStep(step_size)

		# save parameters
		self.p = p

	def setValues(self, p):
		# populate gui with new values
		widgets = (QComboBox, QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox)
		for obj in self.settings_groupBox.findChildren(widgets):
			fullname = str(obj.objectName())
			trimmed_name = fullname.split('_')[0]
			try:
				if isinstance(obj, QComboBox):
					value = p[trimmed_name]
					index = obj.findText(value)  # get the corresponding index for specified string in combobox
					obj.setCurrentIndex(index)  # preselect a combobox value by index
				if isinstance(obj, QLineEdit):
					value = p[trimmed_name]
					if 'spinbox' not in trimmed_name:
						obj.setText(value)  # restore lineEditFile
				if isinstance(obj, QCheckBox):
					value = p[trimmed_name]
					if value is not None:
						obj.setChecked(value)  # restore checkbox
				if isinstance(obj, QSpinBox):
					value = p[trimmed_name]
					obj.setValue(value)  # restore lineEditFile
				if isinstance(obj, QDoubleSpinBox):
					value = p[trimmed_name]
					obj.setValue(value)  # restore lineEditFile
			except:
				continue

	def loadMoviePath(self):
		movie_path = str(QFileDialog.getOpenFileName(self, 'Load movie', '', 'All files (*.*);;Binary (*.bin);;MPTIFF (*.tif);;Raw (*.raw)')[0])
		if movie_path:
			self.moviePath_lineEdit.setText(movie_path)
			self.newMoviePathLoaded(movie_path)

	def newMoviePathLoaded(self, movie_path):
		# get folder path names
		base_path, file_name = os.path.split(movie_path)
		file_name_no_ext = os.path.splitext(file_name)[0]

		# change directory so next loads are easier
		os.chdir(base_path)

		# make save directory (/STA)
		save_dir = os.path.join(base_path, 'STA', file_name_no_ext)
		self.savePath_lineEdit.setText(save_dir)

		# find and auto load sync file
		file_list = glob.glob(os.path.join(base_path, '*.paq'))
		if file_list:
			# fuzzy matching to find paq file name closest to tiff movie
			paq_path = difflib.get_close_matches(movie_path, file_list)[0]
			self.syncPath_lineEdit.setText(paq_path)

		# search for visual stim expt file
		file_list = glob.glob(os.path.join(base_path, '*_StimOrder.mat'))
		if file_list:
			# fuzzy matching to find paq file name closest to tiff movie
			stim_order_path = difflib.get_close_matches(movie_path, file_list)[0]
			if os.path.exists(stim_order_path):
				self.useStimOrder_checkBox.setChecked(True)
				self.stimOrder_lineEdit.setText(stim_order_path)


	def loadSyncPath(self):
		syncPath = str(QFileDialog.getOpenFileName(self, 'Load sync file', '', 'PAQ file (*.paq);; h5 file (*.h5)')[0])
		if syncPath:
			self.syncPath_lineEdit.setText(syncPath)


	def loadStimOrderPath(self):
		stim_order_path = str(QFileDialog.getOpenFileName(self, 'Load stim order file', '', 'MATLAB (*.mat);;All files (*.*)')[0])
		if stim_order_path:
			self.stimOrder_lineEdit.setText(stim_order_path)


	def loadPreset(self):
		filepath = str(QFileDialog.getOpenFileName(self, 'Load preset', self.presets_dir, 'Config file (*.cfg)')[0])
		if filepath:
			p = json.load(open(filepath, 'r'))
			self.setValues(p)

		# set preset name label
		drive, path = os.path.splitdrive(filepath)
		path, filename = os.path.split(path)
		filename, ext = os.path.splitext(filename)
		self.loadedPreset_label.setText('Loaded preset: ' + filename)

	def savePreset(self):
		filepath = str(QFileDialog.getSaveFileName(self, 'Save as preset...', self.presets_dir, 'Config file (*.cfg)')[0])
		if filepath:
			json.dump(self.p, open(filepath, 'w'), sort_keys=True, indent=4)

	def setDefaults(self):
		defaults_file = os.path.join(self.presets_dir, 'GUIdefaults.cfg')
		json.dump(self.p, open(defaults_file, 'w'), sort_keys=True, indent=4)

	def loadDefaults(self):
		defaults_file = os.path.join(self.presets_dir, 'GUIdefaults.cfg')
		if os.path.isfile(defaults_file):
			p = json.load(open(defaults_file, 'r'))
			self.setValues(p)
			self.p = p

	def clickRun(self):
		self.getValues()

		# setup threading
		self.workerObject = Worker(self.p)
		self.workerObject.moveToThread(self.workerThread)
		self.workerObject.status_signal.connect(self.updateStatusBar)
		self.workerThread.started.connect(self.workerObject.work)
		self.workerObject.finished_signal.connect(self.workerThread.exit)
		self.workerThread.start()

	def setSavePath(self):
		save_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
		if save_dir:
			self.savePath_lineEdit.setText(save_dir)

	def updateStatusBar(self, msg):
		self.statusBar.showMessage(msg)




# Main entry to program.  Sets up the main app and create a new window.
def main(argv):
	# create Qt application
	app = QApplication(argv)

	# create main window
	window = MainWindow()

	# fix for windows to show icon in taskbar
	if os.name == 'nt':
		myappid = 'STAMovieMaker' # arbitrary string
		ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

	# show the window icon
	if os.path.isfile('GUI/colourIcon.ico'):
		window.setWindowIcon(QIcon('GUI/colourIcon.ico'))

	# show it and bring to front
	window.show()
	window.raise_()

	# start the app
	sys.exit(app.exec_())

if __name__ == '__main__':
	# launch program
	main(sys.argv)

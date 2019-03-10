# ThorLink
# Lloyd Russell 2017
# simple utilities to load Thor files into python

import os
import time
import numpy as np
import glob
import h5py

# import matplotlib
# matplotlib.use('TkAgg')

# import sys
# sys.path.append("tools")
# from tools import training

# try:
# 	# for Python2
# 	import Tkinter as tk
# 	import tkFileDialog as filedialog
# except ImportError:
# 	# for Python3
# 	import tkinter as tk
# 	from tkinter import filedialog


def ReadRawFile(movie_path=None, start=1, stop=np.Inf, num_frames=np.Inf, verbose=False, pixels_per_line=512, lines_per_frame=512):
	'''
	Input
	-----
	movie_path : string, optional
		full path to *.bin file, if none provided dialog box will open
	start : int, optional
		the first frame to begin reading from
	stop : int, optional
		the frame to stop reading at
	num_frames : int, optional
		the total number of frames to read
	verbose : bool, optional
		choose whether to display output progress and stats

	Output
	------
	data : np.ndarray
		frames x rows x cols array (dtype=np.uint16)
	'''

	# open file diaog if input not provided
	if not movie_path:
		root = tk.Tk()
		root.withdraw()
		movie_path = filedialog.askopenfilename(filetypes=(("RAW files", "*.raw"), ("All files", "*.*") ))

	if verbose:
		print('File path: ' + movie_path)

	# begin timer
	time_started = time.time()

	# open file and read data
	with open(movie_path, 'rb') as f:
		# extract details from header
		samples_per_frame = pixels_per_line * lines_per_frame

		# Find number of frames in file
		filesize_bytes = os.path.getsize(movie_path)
		total_num_frames = int((filesize_bytes/2) / samples_per_frame)  # divide by 2 because format is int16  subtract 2 because file header

		if verbose:
			print('File size: ' + str(total_num_frames) + ' frames (' + str(pixels_per_line) + ', ' + str(lines_per_frame) + ')')

		# Define read limits
		if stop == np.Inf:
			stop = start + num_frames 
		if stop > total_num_frames:
			num_frames = total_num_frames - (start-1)
			stop = total_num_frames
		else:
			num_frames = stop - start

		if verbose:
			print('Requested: ' + str(num_frames) + ' frames, from ' + str(start) + ' to ' + str(stop))

		# Read data
		start_on_byte = (((start-1) * samples_per_frame)) *2  # plus 2 because header size, x2 because 1 uint16 is 2 bytes
		num_chars_to_read = (num_frames * samples_per_frame)  # note size in bytes of char defined by fread function argument
		f.seek(start_on_byte, 0) 
		data = np.fromfile(f, dtype=np.int16, count=num_chars_to_read)

	# Reshape data into frame array
	data = data.reshape(num_frames, lines_per_frame, pixels_per_line, order='C')

	if verbose:
		print('Data size: ' + str(data.shape[0]) + ' frames (' + str(data.shape[1]) + ', ' + str(data.shape[2]) + ')')

	time_taken = time.time() - time_started
	if verbose:
		print('Time taken: ' + '{0:.2f}'.format(time_taken) + ' s')

	return data


def WriteRawFile(data, file_name):
	# construct filename
	if not file_name[-4:] == '.raw':
		file_name = file_name + '.raw'

	with open(file_name, 'wb') as raw_file:
		# write the data
		data.tofile(raw_file)


def ReadSyncFile(file_path=None, datasets=None, listdatasets=False):
	# datasets = list of dataset names to load. load all if none provided

	# open file diaog if input not provided
	if not file_path:
		root = tk.Tk()
		root.withdraw()
		file_path = filedialog.askopenfilename(filetypes=(("H5 files", "*.h5"), ("All files", "*.*") ))

	# read all data
	data = {}
	dataset_names = []
	with h5py.File(file_path,'r') as h5:
		for group_name in h5:
			for dataset_name in h5[group_name]:
				dataset_names.append(dataset_name)
				if datasets == None or dataset_name in datasets:
					data[dataset_name] = h5[group_name][dataset_name][:]
	
	if listdatasets:
		print(dataset_names)

	return data


def ReadVRData(folder_path=None):
	# open file diaog if input not provided
	if not folder_path:
		root = tk.Tk()
		root.withdraw()
		folder_path = filedialog.askdirectory(parent=root, title='Please select a directory')

	# read move file
	move_file_path = glob.glob(os.path.join(folder_path, '*.move'))[0]
	vrtimes, m1x, m1y, m2x, m2y = training.read_move(move_file_path)
	maxn = len(vrtimes)

	# read position file
	is2d = True
	position_file_path = glob.glob(os.path.join(folder_path, '*.position'))[0]
	posx, posy, posz, resets, mdposy = training.read_pos(position_file_path, maxn, is2d)
	postimes = vrtimes.copy()

	# read events file
	events_file_path = glob.glob(os.path.join(folder_path, '*.events'))[0]
	teleport_times = ((np.array(vrtimes)))[np.diff(posy) < -0.25]
	evlist, timeev = training.read_events(events_file_path, teleport_times=teleport_times)
	# evpos = np.array([maps.get_event_pos(postimes, posx, posy, ev, 0) for ev in evlist])
	speed = 1.0e3 * np.sqrt(np.gradient(posx)**2+np.gradient(posy)**2)/np.gradient(postimes)

	return {'evlist': evlist,
		'timeev': timeev,
		'postimes': postimes,
		'posx': posx,
		'posy': posy,
		'speed': speed
		}


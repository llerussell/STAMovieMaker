import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from scipy.ndimage import correlate
from skimage import exposure
import tensorflow as tf

def threshold_detect(signal, threshold):
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]

def normalise(arr):
    arr = arr - np.nanmin(arr)
    arr = arr / np.nanmax(arr)
    return arr
    
def remove_inf_nan(arr):
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    return arr

def normalise_movie(movie, baseline_indices, method='dff'):
    if method.lower() == 'dff':
        baseline = movie[baseline_indices, ...].mean(axis=0)
        norm_movie = ((movie - baseline) / baseline) * 100

    elif method.lower() == 'df':
        baseline = movie[baseline_indices, ...].mean(axis=0)
        norm_movie = (movie - baseline)

    elif method.lower() == 'zscore':
        baseline = movie[baseline_indices, ...].mean(axis=0)
        baseline_sd = movie[baseline_indices, ...].std(axis=0)
        norm_movie = (movie - baseline) / baseline_sd

    elif method.lower() == 'norm':
        baseline = movie[baseline_indices, ...].mean(axis=0)
        norm_movie = movie - baseline
        baseline2 = np.sqrt((norm_movie[baseline_indices, ...] **2).mean(axis=0))
        norm_movie = norm_movie * (1/baseline2)

    return remove_inf_nan(norm_movie)


def make_avg_image(movie, avg_image_frames):
    avg_image = np.median(movie[avg_image_frames], axis=0)
    return avg_image


def threshold(arr, threshold=0):
    arr[arr<threshold] = 0
    return arr


def make_tuning_image(avg_images, num_diff_stims, im_dims, divisor=3):
    avg_img_reshaped = np.reshape(avg_images, [num_diff_stims, im_dims[0]*im_dims[1]]).astype(np.float)
    avg_img_reshaped[avg_img_reshaped < 0] = 0
    pref_stim = np.argmax(avg_img_reshaped, axis=0).astype(np.float)
    H = pref_stim / (num_diff_stims)
    S = np.ones([im_dims[0]*im_dims[1]], dtype=np.float)
    V = np.ones([im_dims[0]*im_dims[1]], dtype=np.float)

    for p in range(im_dims[0]*im_dims[1]):
        pref = H[p] * num_diff_stims
        idx = np.ones(num_diff_stims, dtype=np.bool)
        idx[pref] = 0
        mean_val = avg_img_reshaped[idx, p].mean()
        
        orth = pref + (num_diff_stims/2)
        if orth > num_diff_stims-1:
            orth = orth - num_diff_stims
        orth_idx = np.zeros(num_diff_stims, dtype=np.bool)
        orth_idx[orth] = 1
        orth_val = avg_img_reshaped[orth_idx, p].mean()

        V[p] = (avg_img_reshaped[pref, p] - 0)
#         S[p] = (V[p] - orth_val) / (V[p] + orth_val)
        S[p] = V[p] - mean_val

    for v in range(num_diff_stims):
        idx = np.where(H==(np.float(v)/num_diff_stims))
        # V[idx] = 0.2 * V[idx] / np.mean(V[idx])
        single_img = np.zeros(512*512, dtype=np.float)
        single_img[idx] = V[idx]
        single_img = np.reshape(single_img, [512,512])
        #savename = save_dir + file_name + ('_' + str(num_diff_stims) + 'stims_STA_PrefImage_' + str(v+1) + '.tif')
        #tifffile.imsave(savename, single_img.astype(np.float32))
        # V[idx] = V[idx] / np.max(V[idx])

    V[V<5] = 0
    V = V / (divisor*V.mean())
    V[V>1.0] = 1
    V[V<0.0] = 0
    
    S = S / (2*S.mean())
    S[S>1.0] = 1
    S[S<0.0] = 0

    R = np.zeros(im_dims[0]*im_dims[1], dtype=np.float)
    G = np.zeros(im_dims[0]*im_dims[1], dtype=np.float)
    B = np.zeros(im_dims[0]*im_dims[1], dtype=np.float)
    for p in range(im_dims[0]*im_dims[1]):
        R[p], G[p], B[p] = colorsys.hsv_to_rgb(H[p], S[p], V[p])

    R = np.reshape(R, [im_dims[0],im_dims[1]])
    G = np.reshape(G, [im_dims[0],im_dims[1]])
    B = np.reshape(B, [im_dims[0],im_dims[1]])

    rgb_img = np.zeros([3,im_dims[0],im_dims[1]], dtype=np.float)
    rgb_img[0] = (R * 65535)
    rgb_img[1] = (G * 65535)
    rgb_img[2] = (B * 65535)
    return rgb_img


def colour_by_time(movie, frame_range, smooth=0, useCorrelationImage=False, blurHandS=False):
    if useCorrelationImage:
        corr = makeCorrImg(downsample(movie,5), 8)

    if smooth > 0:
        movie[frame_range] = gaussian_filter(movie[frame_range], sigma=[smooth,0,0])
    

    max_img = movie[frame_range].max(axis=0).astype(np.float32)
    mean_img = movie[frame_range].mean(axis=0).astype(np.float32)

    # hue
    H = np.argmax(movie[frame_range], axis=0)
    H = H.astype(np.float32) / len(frame_range)
    print(H)

    # brightness
    V = max_img
    if useCorrelationImage:
        V = corr
    else:
        corr = V

    # brightness
    V = max_img
    V = V / np.percentile(V, 99)
    V[V<0] = 0
    V[V>1] = 1

    # saturation
    S = (max_img - mean_img) / (max_img + mean_img)
    S[V<np.nanpercentile(V,90)] = S[V<np.nanpercentile(V,90)]/3
    S = S / np.nanpercentile(S, 90)
    S[S<0] = 0
    S[S>1] = 1

    # convert HSV to RGB
    hsv = np.stack((H, S, V), axis=-1)
    hsv2 = hsv
    hsv_tf = tf.convert_to_tensor(hsv, np.float32)

    rgb_tf = tf.image.hsv_to_rgb(hsv_tf)
    sess = tf.Session()
    with sess.as_default():
        rgb = rgb_tf.eval()

    # blur the rgb image then convert back to hsv and keep the blurred H and S channels.
    if blurHandS:
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


    rgb_img = (rgb * 65535).astype(np.uint16)

    return {'RGB':rgb_img, 'HSV':hsv2, 'Corr':corr}


def hsv2rgb(h, s, v):
    # from http://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion (Tomas)
    orig_shape = h.shape
    h = h.ravel()
    s = s.ravel()
    v = v.ravel()
    shape = h.shape
    i = np.int_(h*6.)
    f = h*6.-i
    q = f
    t = 1.-f
    i = np.ravel(i)
    f = np.ravel(f)
    i%=6
    t = np.ravel(t)
    q = np.ravel(q)
    clist = (1-s*np.vstack([np.zeros_like(f), np.ones_like(f),q,t]))*v
    order = np.array([[0,3,1],[2,0,1],[1,0,3],[1,2,0],[3,1,0],[0,1,2]])  #0:v 1:p 2:q 3:t
    rgb = clist[order[i], np.arange(np.prod(shape))[:,None]]
    return rgb.reshape([orig_shape[0], orig_shape[1], 3])



def make_increase_movie(movie):
    inc_movie = movie - 0
    inc_movie[inc_movie<0] = 0
    return inc_movie

def make_decrease_movie(movie):
    dec_movie = movie - 0
    dec_movie[dec_movie>0] = 0
    dec_movie = (dec_movie * -1)
    return dec_movie

def make_inc_dec(movie, avg_image_frames):
    rgb_img = np.zeros([3,im_dims[0],im_dims[1]], dtype=np.int16)
    rgb_img[0] = make_avg_image(make_increase_movie(movie), avg_image_frames)
    rgb_img[2] = make_avg_image(make_decrease_movie(movie), avg_image_frames)
    return rgb_img


def std_chunked(arr, chunk_size=50):    
    num_blocks, remainder = divmod(arr.shape[1], chunk_size)
    sd = np.zeros(arr.shape, arr.dtype)
    
    for start in xrange(0, chunk_size*num_blocks, chunk_size):
        view = arr[:,start:start+chunk_size,:,:]
        sd[:,start:start+chunk_size] = view.std(axis=0)

    if remainder:
        view = arr[:,-remainder:,:,:]
        sd[:,-remainder:] = view.std(axis=0)

    return sd.mean(axis=0)


def makeCorrImg(movie, numNeighbours):
    # taken from epnev ca_source_extraction

    if numNeighbours == 8:
        neighbourhood = np.array(((1,1,1), (1,0,1), (1,1,1)))
    elif numNeighbours == 4:
        neighbourhood = np.array(((0,1,0), (1,0,1), (0,1,0)))
    neighbourhood = neighbourhood.reshape((1,3,3))


    d0,d1,d2 = movie.shape

    # centering
    mY = movie.mean(axis=0)
    movie = movie - mY

    # normalizing
    sY = np.sqrt(np.mean(movie**2, axis=0))
    movie =  movie * (1/sY)

    # compute the correlation
    Yconv   = correlate(movie, neighbourhood)        # sum over the neighbouring pixels
    MASK    = correlate(np.ones((1,d1,d2)), neighbourhood)  # count the number of neighbouring pixels
    corrImg = np.mean(Yconv * movie, axis=0) / MASK         # compute correlation and normalize
    corrImg = corrImg.reshape((d1,d2))
    
    return corrImg


def downsample(array, factor):
    nd = len(array.shape)
    # if nd == 1:  # 1D array (traces)
    #     num_elems = numel(array)
    #     num_rows = ceil(num_elems/factor)
    #     downsampled_array = nan(factor,num_rows)
    #     downsampled_array(1:num_elems) = array
    #     downsampled_array = nanmean(downsampled_array,1)

    # elif nd == 2:  # 2D array (traces)
    #     [d1,d2] = size(array)
    #     new_d2 = ceil(d2/factor)

    #     downsampled_array = nan(d1, new_d2*factor)
    #     downsampled_array[1:d1, 1:d2] = array
    #     downsampled_array = reshape(downsampled_array, factor, new_d2, d1)
    #     downsampled_array = squeeze(nanmean(downsampled_array, 1))
        
    if nd==3:  # 3D array (movie)
        [d1,d2,d3] = array.shape
        
        remainder = np.mod(d1,factor)
        remainder_array = []
        if remainder:
            array = array[:-remainder,:,:]
            [d1,d2,d3] = array.shape
            remainder_array = array[-remainder-1:-1,:,:]

        downsampled_array = array.reshape((int(d1/factor), factor, -1)).mean(axis=1).reshape((int(d1/factor),d2,d3))

        return downsampled_array

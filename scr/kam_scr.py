'''
# ------------------------------------------------------------------------------
# Handy Functions for the Kernel Additive Modelling framework
# ------------------------------------------------------------------------------

NOTE:   Only the vocal kernel is implemented here
        kam_hub : method presented at LVA/ICA 2018 - automatic tailored k

AUTHOR: Delia Fano Yela
DATE: July 2018
'''

import numpy as np
import math
import librosa
from copy import deepcopy
import soundfile
import scipy


# ------------------------------------------------------------------------------
# TEST function for KAM method
# ------------------------------------------------------------------------------
def kam_tester(mix, k_range, fs=44100, samples2compare = 'all', n_fft = 2048, hop_length = 1024, win_length=2048, window='hann'):
    '''
    Input   : mixture of a song
    Output  : estimated sources (vocals/background music)

    Parameters          :
    -> mix              :   mixture array of the song to be processed
    -> k_range          :   range of integers to pick k from, recommended to use %'s of the total number of frames in the mixture (N).
                            for example: k_range = np.multiply(N, range(5,50,5))/100
                            If it is a single int, k will take that value.
                            Otherwise, k will be pick to maximise the hubness of the k-NN graph.
    -> fs               :   sampling frequency
    -> samples2compare  :   the samples in the mixture to be processed. One could include vocal activation information here.


    '''
    # --------------------------------------------------------------------------
    # Compute Magnitude STFT input mixture:
    V = librosa.stft(mix, n_fft, hop_length, win_length, window) #center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')
    Va = np.abs(V)
    # --------------------------------------------------------------------------
    # Compute Vocal separation KAM for the samples2compare
    if samples2compare == 'all':
        iframes2compare='all'
        Vout, hub = kam_hub(Va,k_range, iframes2compare='all', kernel ='vocal')

    else:
        start = int(np.floor(samples2compare[0] / hop_length))
        end = int(np.ceil(samples2compare[-1] / hop_length))
        iframes2compare = np.arange(start, end)
        M, hub = kam_hub(Va,k_range, iframes2compare='all', kernel ='vocal')
        Vout = deepcopy(Va)
        Vout[:,iframes2compare] = M
    # --------------------------------------------------------------------------
    # Compute mask and complex estimate of background music and vocals
    # To avoid /0 case add minimal noise to all : + np.spacing(1)
    Vback, Vvox = mask2complex(Vout+np.spacing(1), Va+np.spacing(1), V+np.spacing(1), type='gaussian')
    # --------------------------------------------------------------------------
    # Convert back to time domain
    back_out = librosa.istft(Vback, hop_length, win_length, window='hann')#, length=len(mix))
    vox_out = librosa.istft(Vvox, hop_length, win_length, window='hann')#, length=len(mix))
    # --------------------------------------------------------------------------
    # Output 2D array containing estimates: number sources x number samples
    estimated_sources = np.stack((back_out,vox_out))

    return estimated_sources


# ------------------------------------------------------------------------------
# KAM method for vocal separation mono mixtures
# ------------------------------------------------------------------------------
'''
Baseline implementation of KAM for vocal separation (i.e. D.FitzGerald - see bellow)
The number of nearest neighbours k is chosen from k_range according to:
"Does k matter? k-NN Hubness Analysis for Kernel Additive Modelling Vocal Separation"
by Delia Fano Yela, Dan Stowell and Mark Sandler - LVA/ICA 2018

'''
def kam_hub(magmat, k_range , iframes2compare = 'all', kernel = 'vocal'):
    '''
    Parameters          :
    -> magmat           : magnitude matrix input mixture
    -> k_range          : range of k integer values to chose from
    -> iframes2compare  : frames in magmat to processe
    -> kernel           : type of kernel (only vocal has been implemented here so far)

    Output              :
    -> outmat           : estimate magnitude of source (background music for vocal sep case)
    -> hub              : maximum hubness value for the k_range given
    '''
    if kernel == 'vocal':
        D = distance_mat(magmat, iframes2compare)
        hub = []
        for k in k_range:
            index_kNN = k_NN(D,k)
            hub.append(hubness(index_kNN, type = 'null model'))

        imx_hub = np.argmax(np.array(hub))
        print "The chosen k is:"
        print k_range[imx_hub]
        index_kNN = k_NN(D,k_range[imx_hub])
        outmat = median_mat(magmat,index_kNN)

    else:
        print('Please pick a kernel:')
        print('Only the "vocal" kernel has been implemented for now .. sorry')
        outmat = 0
        hub = 0
    return outmat, hub

# ------------------------------------------------------------------------------
# Baseline KAM method for vocal separation mono mixtures
# ------------------------------------------------------------------------------
'''
Baseline implementation of KAM for vocal separation following
"Vocal Separation Usgn Nearest Neighbours and Median Filtering"
by Derry FitzGerald, IET Signals and Systems Conference, 2012
'''
def kam(magmat,k=90, iframes2compare = 'all', kernel = 'vocal'):
    '''
    Parameters          :
    -> magmat           : magnitude matrix input mixture
    -> k                : number of nearest neighbours
    -> iframes2compare  : frames in magmat to processe
    -> kernel           : type of kernel (only vocal has been implemented here so far)

    Output              :
    -> outmat           : estimate magnitude of source (background music for vocal sep case)
    -> hub              : hubness for the k value given
    '''
    if kernel == 'vocal':
        D = distance_mat(magmat, iframes2compare)
        index_kNN = k_NN(D,k)
        outmat = median_mat(magmat,index_kNN)
        hub = hubness(index_kNN, type = 'raw')
    else:
        print('Please pick a kernel')
        outmat = 0
        hub = 0
    return outmat, hub


# ------------------------------------------------------------------------------
# HUBNESS of the k-NN graph
# ------------------------------------------------------------------------------
def hubness(index_kNN, type = 'null model'):
    '''
    Parameters      :
    -> index_kNN    : nearest neighbours frames idexes matrix (k x Number of frames)
    -> type         : 'null model' accounts for random conections and measures the gain in hubness
                      'raw' is the gross hubness value

    Output          :
    -> hub          : hubness of the k-NN graph
    '''
    # Take out self-similarity
    index_kNN = index_kNN[1:,:]
    # k -ocurrence
    k_ocurrence = np.bincount(index_kNN.flatten())

    if type == 'raw':
        hub = scipy.stats.skew(k_ocurrence, bias=False)
    elif type == 'null model':
        k = index_kNN.shape[0]
        colm = index_kNN.shape[1]
        null = (1 - 2*k/colm) / np.sqrt( [k*(1-k/colm)])
        hub = scipy.stats.skew(k_ocurrence) - null

    return hub

# ------------------------------------------------------------------------------
# Self-similarity matrix
# ------------------------------------------------------------------------------
def distance_mat(matrix, iframes2compare='all', type='Euclidean'):
    if iframes2compare == 'all':
        X = matrix
        W = matrix
    else:
        X = matrix[:,iframes2compare]
        W = matrix

    cx = X.shape[1]
    cw = W.shape[1]

    # Do sum of squares
    Xa = np.einsum('ij,ij->j',X,X)
    Wa = np.einsum('ik,ik->k',W,W)
    Wa = Wa.reshape(-1,1)

    if type=='Euclidean':
        D = np.tile(Xa, [cw, 1]) + np.tile(Wa,[1, cx]) - 2*np.einsum('ik,ij->kj',W,X)
        #D = np.tile(Xa, [cw, 1]) + np.tile(Wa,[1, cx]) - 2*np.einsum('ki,ij->kj',W.transpose(),X)
    return D


# ------------------------------------------------------------------------------
# Select K Nearest Neighbours
# ------------------------------------------------------------------------------
def k_NN(distance_mat, k):
    index_all = distance_mat.argsort(axis=0)
    index_knn = index_all[0:k,:]
    return index_knn


# ------------------------------------------------------------------------------
# MEDIAN
# ------------------------------------------------------------------------------
def median_mat(matrix, index_knn):
    M = np.median(matrix[:,index_knn], axis = 1)
    return M

# ------------------------------------------------------------------------------
# MASKING
# ------------------------------------------------------------------------------
def mask2complex(source_mag, mix_mag, mix_complex, type='gaussian'):
    mask_type = {'gaussian': gaussian, 'soft':soft, 'binary':binary}
    return mask_type[type](source_mag, mix_mag, mix_complex)

def gaussian(source_mag, mix_mag, mix_complex):
    num = np.square(np.log10(mix_mag) - np.log10(source_mag))
    W = np.exp(-num/2)
    source = np.einsum('ij,ij->ij',W,mix_complex)
    noise = np.einsum('ij,ij->ij',(1-W),mix_complex)
    return source, noise

def soft(source_mag, mix_mag, mix_complex):
    noise_model = np.maximum(0,mix_mag-source_mag)
    mask_source = source_mag / (source_mag + noise_model)
    mask_noise = noise_model / (source_mag + noise_model)
    source = np.einsum('ij,ij->ij',mask_source,mix_complex)
    noise = np.einsum('ij,ij->ij',mask_noise,mix_complex)
    return source, noise

def binary(source_mag, mix_mag, mix_complex):
    noise_model = np.maximum(0,mix_mag-source_mag)
    W = source_mag >=  noise_model
    source = np.einsum('ij,ij->ij',W,mix_complex)
    noise = np.einsum('ij,ij->ij',(1-W),mix_complex)
    return source, noise

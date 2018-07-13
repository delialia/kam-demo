'''
# ------------------------------------------------------------------------------
# Example implementation of baseline KAM for vocal separation using the hubness
# of the k-NN graph to automatically fix k (LVA/ICA 2018 D. Fano Yela et al)
# ------------------------------------------------------------------------------

AUTHOR: Delia Fano Yela
DATE: July 2018
'''
import numpy as np
from librosa import load
from kam_scr import kam_tester
from scipy.io.wavfile import write

# Load song
mix, fs = load('audio_sample.wav', sr=44100, mono=True, duration=30)
# Set the range of values to optimise k:
#k_range = range(10,600, 10) # To set k manually to a value * set: k_range = [*]
k_range = [10]
# Separate background/vocals sources
estimated_sources = kam_tester(mix, k_range, fs)
# Normalise
back =estimated_sources[0,:]
vox = estimated_sources[1,:]
back = back/np.max(back)
vox = vox/np.max(vox)
# Listen to the results!
write('./background.wav', fs, back )
write('./vocals.wav', fs, vox)

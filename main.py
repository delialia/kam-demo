'''
# ------------------------------------------------------------------------------
# DOES k MATTER?
# ------------------------------------------------------------------------------

Demo accompanying the poster presented in LVA/ICA 2018 Conference - Ref Paper:
"Does k matter? k-NN Hubness Analysis for Kernel Additive Modelling Vocal Separation"

AUTHOR: Delia Fano Yela
DATE: July 2018

ACKNOWLEDGEMENTS: Fabian-Robert Stoeter, Florian Thalmann, Gijs Wijnholds and Giulio Moro - thank you all :)

NOTE: Feel free to contact me -> d.fanoyela@qmul.ac.uk <- with QS/Feedback!

'''

from flask import Flask
from flask import render_template, request, make_response, jsonify
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scr.kam_scr as kam_scr
import os
import scipy.io.wavfile  as wav
from io import BytesIO
import soundfile as sf



app =  Flask(__name__)
app.secret_key = os.urandom(24) # secret key to encript the values in the dictionary session while the server is live

# ------------------------------------------------------------------------------
# Start html template!
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

app.config["UPLOAD_FOLDER"] = "./static"

# ------------------------------------------------------------------------------
# Get the WAV dropped and do stft
# ------------------------------------------------------------------------------
@app.route("/sendfile", methods=["POST"])
def send_file():
    global V, Va, fileob# <----------------- Using global variables is NOT the best practise tbh (OK for a demo). Look-up Flask Sessions
    fileob = request.files["file2upload"]
    # Read WAV file
    fs, input = wav.read(fileob)
    if input.ndim == 2:
        mix = np.mean(input, axis=1)
        mix = mix[:1323000] # <--------- Signal is chopped to first 30 seconds: COMMENT LINE for FULL SONG processing
    else:
        mix = input[:1323000]
    # STFT
    V = librosa.core.stft(mix, n_fft=4096, hop_length=2048, win_length=4096, window='hann') #center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')
    Va = np.abs(V)

    return "Done uploading WAV file"


# ------------------------------------------------------------------------------
# Separate vocals/background music with KAM given the k by the slider
# ------------------------------------------------------------------------------
@app.route( '/separate', methods = ["GET"] )
def separate():
    global back_out, vox_out, Vsource, Vnoise # <----------------- Using global variables is NOT the best practise tbh (OK for a demo). Look-up Flask Sessions
    # GET k value from slider
    k = request.args.get( 'k', type = int )

    # Process separation with that k value
    outmat, hub = kam_scr.kam(Va,k, iframes2compare = 'all', kernel = 'vocal')
    # + np.spacing(1) to avoid further dissapointing with "/ 0 "
    Vsource, Vnoise = kam_scr.mask2complex(outmat+np.spacing(1), Va+np.spacing(1), V+np.spacing(1), type='gaussian')
    print "Done with separation and masking"

    # Convert back to time domain
    back_out = librosa.istft(Vsource, hop_length=2048, win_length=4096, window='hann')
    vox_out = librosa.istft(Vnoise, hop_length=2048, win_length=4096, window='hann')
    # Normalise for WAV
    back_out = back_out/np.max(back_out)
    vox_out = vox_out/np.max(vox_out)
    print "Done converting back to time domain"

    return jsonify(result = "Done with the separation")

# ------------------------------------------------------------------------------
# GET images of magnitude spectograms : Avoids having to write/read images
# ------------------------------------------------------------------------------
@app.route("/image_<track>.png")
def get_image(track):
    # Fabian's buffer trick :
    buf = BytesIO()

    if track == 'mix':

        fig = plt.figure()
        librosa.display.specshow(Va[:500,:]**0.6, x_axis='frames', y_axis='linear')
        plt.title("Mixture")

        plt.savefig(buf, format = "PNG")

    elif track == 'estimates':

        plt.subplot(1,2,1)
        librosa.display.specshow(np.abs(Vsource[:500,:])**0.6, x_axis='frames', y_axis='linear')
        plt.title('Background')
        plt.subplot(1,2,2)
        librosa.display.specshow(np.abs(Vnoise[:500,:])**0.6, x_axis='frames', y_axis='linear')
        plt.title('Vocals')

        plt.savefig(buf, format = "PNG")


    response = make_response(buf.getvalue())
    buf.close()
    dis = 'attachment; filename=image_%s.png' % track
    response.headers['Content-Type'] = 'image/PNG'
    response.headers['Content-Disposition'] = dis
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'

    return response


# ------------------------------------------------------------------------------
# GET estimates WAV: Avoids having to write/read WAV files
# ------------------------------------------------------------------------------
@app.route("/result_<source>.wav")
def get_audio(source):
    fs = 44100 # <--------------------------- HARDCODED: Change if different!
    # Fabian's buffer trick :
    buf = BytesIO()
    if source == 'back':
        sf.write(buf,back_out, fs, format='WAV')
    elif source == 'vocals':
        sf.write(buf ,vox_out, fs, format='WAV')

    response = make_response(buf.getvalue())
    buf.close()
    dis = 'attachment; filename=result_%s.wav' % source
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = dis

    return response



# Lines of code to quick off our sever and to launch script:
if __name__ == "__main__":  # we only run the server when this file is called directly
    app.run()     # debug=True when in developer mode

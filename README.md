# kam-demo
Does k matter? Code for the method and demo presented at LVA/ICA 2018

------------------------------------------------------------------------------
About the METHOD
------------------------------------------------------------------------------
The implementation of KAM with the proposed method to automatically find a tailored k using the hubness of the k-NN graph as an indicator can be found in ./scr/kam_scr function: kam_hub  
An example implementation of such can be found in ./scr/example.py :
```
python example.py
```

For further detail about the proposed method please refer to:  
"Does k matter? k-NN Hubness Analysis for Kernel Additive Modelling Vocal Separation" 
Delia Fano Yela, Dan Stowell and Mark Sandler, LVA/ICA Surrey July 2018.


------------------------------------------------------------------------------
About the DEMO
------------------------------------------------------------------------------
![This is how it looks like..](https://github.com/delialia/kam-demo/blob/master/Demo_pic.png)

In your terminal run 
```
python main.py
```
to deploy the demo locally.    
Should be running on http://127.0.0.1:5000/ - copy/paste this URL in your browser (only tested on chrome)  
Drag and drop a WAV file   
Use the slider to select a value for "k" (i.e. the number of nearest neighbours) - measured in time frames (not samples)   
Click button to start the separation    
Listen to the separated vocals/background estimates     
Change the k value and run again    
Does k matter?


Only the first 30seconds of the song will be processed. Feel free to change this in the code.   
Sampling frequency set to 44100. Change if different. 


Dependencies
------------------------------------------------------------------------------
Running on Python 2.7 
Flask  
numpy  
librosa  
matplotlib  
scipy  


------------------------------------------------------------------------------
AUTHOR: Delia Fano Yela  
DATE: July 2018  
CONTACT: d.fanoyela@qmul.ac.uk  

import sys
import os
import shutil
import librosa
import glob
import numpy as np
from librosa import display as ld
from IPython import display as ipd
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load, dump

sys.path += ['../src/filecheck', '../src/preprocessing',]
from flatfielding import *
from filepath import *


def mel_spectrogram(file,
                    scaler=None,
                    n_fft=1024,
                    hop_length=512,
                    n_mels=64,
                    power=2,
                    window='hann'):
    """
    Calculates spectrogram of the given audio
    :param file (str): path to wav
    :param n_fft (int): no.of samples in each frame
    :param hop_length (int): hop samples
    :param n_mels (int): no. of mel-bands
    :param power (int): 1 for energy, 2 for power
    :param window (str): 'STFT' window, e.g. 'Hann'

    :return: Mel spectrogram matrix (n_mels, n_time_frames)
    """
    sr, y = read_audio(file)

    mel_spec = librosa.feature.melspectrogram(y,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              sr=sr,
                                              n_mels=n_mels,
                                              window=window,
                                              power=power)

    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = log_mel_spec.T
    #log_mel_spec = scaler.transform(log_mel_spec)

    return log_mel_spec


def mel_spectrogram_list(filepath, 
                         fit_scaler=True,
                         scale_dir='/scaler',
                         out_dir='/specdata',
                         n_fft=1024,
                         hop_length=512,
                         n_mels=64,
                         power=2,
                         window='hann'):
    """
    Generate mel spectrograms for all the
    Audio files in 'filepath'.
    
    If fit_scaler=True, fit with StandardScaler.
    To transform scaler, set fit_scaler=False
    
    :param filepath (list): Path to all .wav files
        Output from 'extract_filepath' func.
    :param scale_dir (str): path to store the scaler
    :param out_dir (str): path to store the features
    
    :return: Transposed mel spectrograms (file_no, time_length, mels)
        (Only for visualization. All features stored in 'out_dir')
    """
    # Fit scaler
    if fit_scaler:
        scaler = StandardScaler()
        
        for i, filename in enumerate(filepath):
            
            log_mel_spec = mel_spectrogram(filename,
                                           n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels, 
                                           power=power, 
                                           window=window)
            
            if i==0:
                dirpath = '/'.join(filename.split("/")[:-2])
                scalerpath = dirpath + scale_dir
                unscaled_spec = np.empty((len(filepath), 
                                        log_mel_spec.shape[0], 
                                        log_mel_spec.shape[1]
                                       ))
                
            unscaled_spec[i,] = log_mel_spec
            
            scaler.partial_fit(X=log_mel_spec)
        # Save scaler
        dump(scaler, scalerpath+ "/" + "scaler.bin", compress=True)
        return unscaled_spec
    
    # Load and Transform scaler
    else:
        try:
            for i, filename in enumerate(filepath):
            
                log_mel_spec = mel_spectrogram(filename,
                                               n_fft=n_fft, 
                                               hop_length=hop_length, 
                                               n_mels=n_mels, 
                                               power=power, 
                                               window=window)
            
                if i==0:
                    dirpath = '/'.join(filename.split("/")[:-2])
                    outpath = dirpath + out_dir
                    scalerpath = dirpath + scale_dir
        
                    myscaler = load(scalerpath+ "/" + "scaler.bin")
            
                    scaled_spec = np.empty((len(filepath), 
                                            log_mel_spec.shape[0], 
                                            log_mel_spec.shape[1]
                                           ))
                
                log_mel_spec = myscaler.transform(log_mel_spec)
                scaled_spec[i,] = log_mel_spec
            # Save features
            np.save(outpath+'/'+'data.npy', scaled_spec)
            return scaled_spec
        
        except FileNotFoundError:
            print("Path to scaler not found!")
            print("set 'fit_scaler=True' and rerun. Exiting...")
import numpy as np
import librosa
import soundfile
import librosa.display
import IPython.display as ipd
import os
import sys
import shutil

def read_audio(file, mono=True):
    """
    file (str) : path to wav file
    mono (bool) : Whether mono or stereo
    
    return librosa sr and audio
    """
    try:
        y, sr = librosa.load(file, sr=None, mono=mono)
        return sr, y
    except FileNotFoundError:
        print("The given file {} does not exist. Exiting...".
              format(file))


def extract_filepath(BASE, inst='fan', id='id_00', cond='normal', extradir=None):
    """
    Gives the path to the wav files for
    chosen instrumental id and condition

    :param BASE: str
        Path of the base data directory
    :param inst: str
        Name of the machine
    :param id: str
        Id of the machine
    :param cond: str
        Normal or Abnormal
    :param extradir: str
        additional dir under 'cond'
        e.g.: '/flatfield'
    return: Path of all wav files of a given instrument id

    """
    if extradir==None:
        data_path = BASE + '/' + inst + '/' + id + '/' + cond
    else:
        data_path = BASE + '/' + inst + '/' + id + '/' + cond + '/' + extradir
        
    try:
        all_files = [file for file in os.listdir(data_path)]
        file_path = [data_path + '/' + file for file in all_files
                     if file.endswith('.wav')]
        return file_path
    except FileNotFoundError:
        exit_code=0
        print("The given filepath does not exist...")
        print("Returning zero...")
        return exit_code


    
def check_data_shape(filepath, sr):
    """
    Checks the shape of input files
    and returns the unique shapes

    :param filepath: str
        Path of the wav files
        Output of the func: 'extract_filepath'
    :param sr: Sampling rate (Hz)

    return: unique shapes of all the files
    """
    shape_list = []

    if type(filepath)=='list':
        print("Total number of wav files: {}".format(len(filepath)))

    try:
        for i, path in enumerate(filepath):
            signal, sr = librosa.load(path, sr=sr)
            signal_shape = signal.shape
            shape_list.append(signal_shape)

        unique_shape = set(shape_list)
        print("The wav files consist of the following shapes={}"
              .format(unique_shape))
        return unique_shape
    except TypeError:
        exit_code=0
        print("The filepath is an 'int'. 'List' expected...")
        print("Returning zero...")
        return exit_code
        

def read_features(filepath):
    """
    Read the mel spectrogram features
    Stored as .npy files
    """
    dirpath = '/'.join(filepath[0].split("/")[:-2]) 
    feature_path = dirpath + '/specdata/data.npy'
    feature_path = os.path.abspath(feature_path)
    #features = sorted(glob.glob(feature_path))
    
    if not os.path.exists(feature_path):
        print("No feature files! Exiting...")
        
    return np.load(feature_path)
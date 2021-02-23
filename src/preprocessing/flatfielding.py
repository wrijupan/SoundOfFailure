import numpy as np
import librosa
import soundfile
import librosa.display
import IPython.display as ipd
import os
import sys
import shutil



def pad_audio(audio, target_T, sr):
    """
    Add zero padding at the beginning
    of audio if audio length is smaller
    than the target length

    :param audio: librosa audio file
    :param target_T: Target time length (s)
    :param sr: sampling rate (Hz)

    return: Zero padded audio
    """
    # Calculate target number of samples
    n_samples = int(target_T * sr)
    # Calculate input shape
    inp_shape = audio.shape
    # Calculate no. of zero samples to be added
    n_pad = n_samples - inp_shape[0]

    pad_shape = (n_pad,) + (inp_shape[1:])

    # Pad if there is something to pad
    if pad_shape[0] > 0:
        print("Adding {} secs of zero-padding".format(n_pad / sr))
        if len(pad_shape) > 1:
            return np.vstack((np.zeros(pad_shape),
                              audio))
        else:
            return np.hstack((np.zeros(pad_shape),
                              audio))
    elif pad_shape[0] == 0:
        print("Padding not required")
        return audio


def trim_audio(audio, target_T, sr):
    """
        Trim audio at the end if audio
        length is greater
        than the target length

        :param audio: librosa audio file
        :param target_T: Target time length (s)
        :param sr: sampling rate (Hz)

        return: Trimmed audio
        """
    # Calculate target number of samples
    n_samples = int(target_T * sr)
    # Calculate input shape
    inp_shape = audio.shape
    # Calculate no. of samples to trim
    n_trim = inp_shape[0] - n_samples

    if n_trim > 0:
        print("Trimming {} secs of sample".format(n_trim / sr))
        audio = audio[: inp_shape[0] - n_trim,]
        return audio

    else:
        print("Trimming not required")
        return audio


def flatfield_audio(filepath, target_T, sr):
    """
    Check length of the audio files under
    filepath. Then if length < target_length,
    use zero-padding at the beginning of audio.
    Else length > target_length, trim audio end.

    :param filepath: Path to all wav files of
    a given instrument model (normal/abnormal)
    :param target_T: Target time length (s)
    :param sr: Sampling rate (Hz)
    :return: No return
    Creates a new dir 'flatfield' under filepath
    and stores preprocessed audio files.
    """
    # Name of the parent directory excluding filenames
    sound_dir = '/'.join(filepath[0].split('/')[:-1])
    # Directory name to store preprocessed files
    flatfield_dir = sound_dir + '/flatfield'
    # Create the directory to store preprocessed files
    if os.path.exists(flatfield_dir):
        print('Path Exists. Removing and adding...')
        shutil.rmtree(flatfield_dir)
        os.mkdir(flatfield_dir)
    else:
        print("Path does not exists. Creating...")
        os.mkdir(flatfield_dir)

    for file in filepath:
        if file.endswith('.wav'):
            filename = file.split('/')[-1]
            outdir = flatfield_dir + '/' + filename
            print("File: {}".format(file))
            audio, sr = librosa.load(file, sr=sr)
            target_shape = int(target_T * sr)

            if target_shape > audio.shape[0]:
                # Zero-padding
                zero_padded = pad_audio(audio, target_T, sr)
                soundfile.write(outdir, data=zero_padded, samplerate=sr)
            elif target_shape < audio.shape[0]:
                # Trimming
                trimmed = trim_audio(audio, target_T, sr)
                soundfile.write(outdir, data=trimmed, samplerate=sr)
            else:
                # Save audio as it is
                print("Zero padding or trimming not required...")
                soundfile.write(outdir, data=audio, samplerate=sr)

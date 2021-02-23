# Project Overview
#### The problem: Industries experience an average downtime of ~800 hours/year. The average cost of downtime can be as high as ~$20,000 per hour! Often a major cause of downtime is malfunctioning machines. During downtime, the overhead operating costs keeps growing without a significant increase in productivity. A survey in 2017 had found that 70% of companies cannot estimate when an equipment starts malfunctioning and only realise when it’s too late. If malfunctions can be detected early, downtime costs can be drastically reduced.

#### The proposed solution: The idea is to diagnose machine faillure using their acoustic footprint over time. A machine will produce a different acoustic signature in its abnormal state compared to its normal state. An algorithm should be able to differentiate between the two sounds.

# Dataset
Recently Hitachi released a first of its kind data set containning real-world machine sounds from industrial equipements: https://zenodo.org/record/3384388#.YDOLexNKiAw. This data set will be used in the project for diagnosing machine failure.

# Tech Stack
The following are the main technical software packages that will be required for the project -
1. Numpy
2. Librosa
3. Scipy
4. Tensorflow, Keras
5. Scikit-learn

# Methods
#### Data preprocessing:

(the current progress in the project involves this part)
The sound problem will be converted to a computer vision problem by converted the sound to its image representation (Mel spectrograms). The preprocessing steps include trimming and zero-padding audio, generating Mel spectrograms, standardization, chunking the spectrograms to smaller blocks for generating training and validation data batches.

#### Machine Learning:
In general, only machine sounds from a normal state of an instrument will be available, i.e. the algorithm will not know beforehand how an abnormal sound looks like. So the training would be unsupervised using only normal sound data. Then during validation when the algorithm encounters an abnormal sound, it will identify that as an outlier.

The approaches that will be used are (under active development. The results will appear very very soon here!):
1. Variational Autoencoder
2. Transfer Learning models for feature extraction and using anomaly detection models on the extracted features.

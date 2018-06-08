# Voice-Emotion-Classifier
A neural network model to classify emootion of the speaker into 6 diferent categories.


The voice corpus used for this project is RAVDESS emotion dataset, which is available here https://zenodo.org/record/1188976#.WxpDfo74lPY
“The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)” by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.

## Description of Dataset:

Audio Files consists of:
Speech file contains 1440 files: 60 trials per actor x 24 actors = 1440. 
Song file contains 1012 files: 44 trials per actor x 23 actors = 1012.

## File naming convention

Each of the 7356 RAVDESS files has a unique filename. The filename consists of a 7-part numerical identifier (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:

Filename identifiers 

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


## Model 

The voice files are parsed and the mfcc features of each file is extracted and stored. Then the data is split into training
and validation set. Normalization is done on the data so no one feature can dominate others.

Then it is fed to a simple 3 layer ANN with SGD optimizer with momentum.

An accuracy of 85% was achieved by this simple model. In future this can be extended to a CNN, by using voice samples like images
and try to improve accuracy even futher.


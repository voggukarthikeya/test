# Speech Emotion Recognition (SER)

A comprehensive machine learning project for recognizing emotions from speech audio using deep learning techniques and multiple emotional speech datasets.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green)](https://librosa.org/)

## 📋 Overview

Speech Emotion Recognition (SER) is the process of identifying human emotions from speech signals. This project implements a deep learning approach to classify emotions from audio files using advanced feature extraction techniques and neural networks.

The system can detect various emotions including:
- **Happy** 😊
- **Sad** 😢  
- **Angry** 😠
- **Fear** 😨
- **Disgust** 🤢
- **Surprise** 😲
- **Neutral** 😐
- **Calm** 😌

## 🚀 Features

- **Multi-Dataset Integration**: Combines four major emotional speech datasets
- **Advanced Feature Extraction**: Utilizes MFCC, Chroma, and Mel-spectrogram features
- **Deep Learning Models**: Implements CNN, LSTM, and hybrid architectures
- **Data Visualization**: Comprehensive EDA and emotion distribution analysis
- **Real-time Processing**: Capable of processing live audio input
- **High Accuracy**: Achieves competitive performance on benchmark datasets

## 📊 Datasets Used

This project integrates four well-known emotional speech datasets:

### 1. RAVDESS (Ryerson Audio-Visual Database)
- **Size**: 1,440 audio files
- **Actors**: 24 professional actors (12 male, 12 female)
- **Emotions**: 8 emotions with 2 intensity levels
- **Format**: 16-bit, 48kHz WAV files

### 2. CREMA-D (Crowd Sourced Emotional Multimodal Actors)
- **Size**: 7,442 audio clips  
- **Actors**: 91 actors (48 male, 43 female)
- **Age Range**: 20-74 years
- **Diversity**: Multiple ethnicities and backgrounds

### 3. TESS (Toronto Emotional Speech Set)
- **Size**: 2,800 audio files
- **Speakers**: 2 female actresses (aged 26 and 64)
- **Words**: 200 target words
- **Emotions**: 7 emotion categories

### 4. SAVEE (Surrey Audio-Visual Expressed Emotion)
- **Size**: 480 audio files
- **Speakers**: 4 male native English speakers
- **Age Range**: 27-31 years
- **Content**: Phonetically balanced sentences

## 🛠️ Technical Implementation

### Feature Extraction
The system extracts three types of audio features:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**
   - Represents short-term power spectrum of sound
   - Captures spectral characteristics of speech

2. **Chroma Features**
   - Relates to 12 different pitch classes
   - Captures harmonic content

3. **Mel-Spectrogram**
   - Mel-scaled frequency representation
   - Better matches human auditory perception

### Model Architecture
- **Input Layer**: Audio feature vectors
- **Convolutional Layers**: For spatial feature learning
- **LSTM Layers**: For temporal sequence modeling
- **Dense Layers**: For final classification
- **Output**: Emotion probability distribution

### Data Preprocessing
1. Audio normalization and resampling
2. Feature extraction using Librosa
3. Data augmentation techniques
4. Train-validation-test split
5. Feature scaling and normalization

## 🔧 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- Git

### Required Libraries
```bash
pip install tensorflow
pip install librosa
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install keras
pip install IPython
```



## Download datasets
   - RAVDESS: [Download here](https://zenodo.org/record/1188976)
   - CREMA-D: [Download here](https://github.com/CheyneyComputerScience/CREMA-D)
   - TESS: [Download here](https://tspace.library.utoronto.ca/handle/1807/24487)
   - SAVEE: [Download here](http://kahlan.eps.surrey.ac.uk/savee/)

## 📈 Performance Metrics

The model's performance is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-emotion precision scores
- **Recall**: Per-emotion recall scores
- **F1-Score**: Balanced performance metric
- **Confusion Matrix**: Detailed classification results

### Results
- Test Accuracy: ~97%

*Note: Results may vary based on dataset splits and hyperparameter tuning*

## 🎯 Usage Examples

### Basic Emotion Prediction
```python
# Load trained model
model = load_model('emotion_model.h5')

# Extract features from audio file
features = extract_features('audio_file.wav')

# Predict emotion
emotion = model.predict(features)
print(f"Detected emotion: {emotion}")
```

### Real-time Emotion Detection
```python
# Record audio and predict emotion in real-time
import pyaudio
import numpy as np

# Setup audio recording
# Process audio chunks
# Extract features and predict
```

## 🔬 Research Applications

This project can be applied in various domains:

- **Healthcare**: Mental health assessment and therapy
- **Education**: Student engagement analysis
- **Customer Service**: Call center quality monitoring
- **Entertainment**: Interactive gaming and media
- **Human-Computer Interaction**: Emotion-aware interfaces
- **Market Research**: Consumer sentiment analysis

## 🙏 Acknowledgments

- **RAVDESS Dataset**: Livingstone & Russo (2018)
- **CREMA-D Dataset**: Cao et al. (2014)
- **TESS Dataset**: University of Toronto
- **SAVEE Dataset**: University of Surrey
- **Librosa Library**: For audio processing capabilities
- **TensorFlow/Keras**: For deep learning framework

## 📞 Contact

**Author**: RAJASUBBU1809  
**GitHub**: [@RAJASUBBU1809](https://github.com/RAJASUBBU1809)

For questions or collaboration opportunities, please open an issue on GitHub.

## 📚 References

1. Livingstone, S.R. & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLoS ONE*, 13(5).

2. Cao, H., Cooper, D.G., Keutmann, M.K., Gur, R.C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377-390.

3. McFee, B., Raffel, C., Liang, D., Ellis, D.P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. *Proceedings of the 14th python in science conference*.

---

⭐ **Star this repository if you found it helpful!** ⭐# test

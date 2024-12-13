# Machine Learning Setup Guide

## Table of Contents
1. [Python Environment Setup](#python-environment-setup)
2. [Virtual Environment Configuration]()
3. [Package Installation]()

## Python Environment Setup

### Requirements
- Python 3.9 or above
- Tensorflow 2.12 or above
- [Google Colaboratory](https://colab.research.google.com/) or [Jupyter Notebook](https://jupyter.org/install)

### Installation Methods

#### Option 1: Anaconda (Recommended)
```bash
# Download Anaconda from official website
# Create new environment
conda create -n mlproject python=3.9
conda activate mlproject
```

#### Option 2: Python venv
```bash
# Create virtual environment
python3 -m venv mlenv
source mlenv/bin/activate  # On Unix/MacOS
mlenv\Scripts\activate  # On Windows
```

### Package Installation

```bash
pip install -r requirements.txt
```

#### GPU Support (Optional)
For TensorFlow GPU:
```bash
pip install tensorflow-gpu
```

### Running the Code
1. Open the [.ipynb](https://github.com/TCHWG/Machine-Learning/blob/main/piano_mistake_generator.py) file in Google Colab or Jupyter Notebook: 
2. Click `Copy to Drive` or Click `File > Save a copy` in Drive. This will allow you to run and edit the `.ipynb` file in your own Google Drive account
3. If using the AIRA_dataset and models:
  - Ensure that the AIRA_dataset and AIRA_model directories are included
  - Run the `Predict` code to generate predictions
4. If you're not using the AIRA dataset and models:
  - Provide the MIDI file you want to generate predictions for by setting the following parameters on `Parameters` cell:
  ```python
  input_midi = 'input/Twinkle_Twinkle_Little_Star_Easy.mid'
  output_folder = Path('AIRA_dataset')
  midi_features = 'AIRA_dataset/midi-features.csv'
  num_var = 2000
  ...
  ```
  - Run every cell in the `.ipynb` file
  - The model `.keras` file will be updated in the `AIRA_model/v3_layer(16)` directory, along with its scalar and encoded labels. If you wish to retain the old models, you can change the directory to a new one.


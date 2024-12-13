# AIRA - Machine Learning

<p align="center" ><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/512px-TensorFlow_logo.svg.png?20211220215155" width="325"/> </p>

## Overview

Machine Learning (ML) is the core of AIRA, responsible for analyzing and evaluating piano performance based on user-uploaded audio. Machine learning model is designed to classify piano performance mistakes. The model analyzes user-uploaded MIDI files by comparing them to ideal reference MIDI files, identifying and categorizing mistakes into four types: 
1. Wrong Note
2. Missing Note
3. Extra Note
4. No Error

## Technology Stack
### Libraries and Frameworks
| Library            | Functionality              |
|:------------------:|:--------------------------|
| TensorFlow         | Deep learning and neural network development | 
| Keras              | High-level API for neural network design     | 
| scikit-learn       | Machine learning utilities                   | 
| pretty_midi        | MIDI file processing | 
| NumPy              | Numerical computations                          | 
| Pandas             | Dataset manipulation and analysis               | 
| Matplotlib         | Data visualization               | 
| Seaborn            | Statistical data visualization               | 

## Dataset
A custom synthetic dataset created using the [Twinkle Twinkle Little Star](https://musescore.com/juliathezhu/twinkle-twinkle-little-star-easy) to train the model and simulate piano mistake recognition. These methods are inspired by approaches outlined in the [paper](https://repositori.upf.edu/bitstream/handle/10230/60657/morsi_SMC_simu.pdf?sequence=1&isAllowed=y) and are applied to identify and correct musical mistakes.

## Model Architecture
Our model employs a **Feedforward Neural Network** (FNN) for piano mistake detection
- **Input**: Variable input shape
- **Hidden Layer** Layers
  - 16 neurons
  - ReLU activation
- **Batch Normalization**
- **Dropout (10%)**
- **Output Layer**: Softmax activation for multi-class classification

## Results
The following metrics are used to evaluate the model's performance for each type of piano mistake:
- ROC AUC: Measures the ability of the model to distinguish between classes (mistake vs. no mistake) for each category.
- Precision-Recall AUC: Focuses on the trade-off between precision and recall, particularly useful for imbalanced datasets.
- Training and Validation Metrics: Tracks the model's accuracy, precision, and recall during training and validation.

### Performance Summary
| **Metric**          | **Training Value**  | **Validation Value**  |
|:------------------  |:--------------------|:----------------------|
| Accuracy            | 77.7%               | 72.4%                 |
| Precision           | 78.2%               | 73.6%                 |
| Recall              | 76.2%               | 72.5%                 |

### AUC Scores by Mistake Type
| **Mistake Type**    | **ROC AUC**  | **Precision-Recall AUC**  |
|:------------------  |:-------------|:--------------------------|
| Extra Note          | 91%          | 68%                       |
| Missing Note        | 92%          | 72%                       |
| Wrong Note          | 85%          | 70%                       |
| No Mistake          | 100%         | 100%                      |

## Usage
### ML Environtment Setup Guide:
🚀 Set up your machine learning environment
[Click here for machine learning Setup Guide](https://github.com/TCHWG/)

## Potential Improvements
- Implement cross-validation to enhance the model's robustness and prevent overfitting
- Explore alternative audio representations such as spectrograms or piano rolls, to capture different musical features
- Experiment with deeper neural network architectures to improve performance and accuracy
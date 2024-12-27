# Fake News Detection Using LSTM

This project implements a deep learning model to classify news articles as **REAL** or **FAKE** using text analysis. The model leverages the TensorFlow library and an LSTM-based architecture to capture the sequential patterns in text data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Introduction

The rise of fake news has led to the need for robust systems to detect and filter misinformation. This project builds a text classification model using natural language processing (NLP) techniques and an LSTM network to classify news articles as REAL or FAKE.

---

## Dataset

The dataset is expected to contain the following columns:
- `text`: The content of the news article.
- `label`: The classification label, either "REAL" or "FAKE".

The dataset file should be named `news.csv` and placed in the project directory.

---

## Project Files

- **`main.ipynb`**: A Jupyter Notebook containing all the code for preprocessing, training, evaluation, and visualization.
- **`news.csv`**: The dataset file containing the news articles and their respective labels.

---

## Project Workflow

1. **Data Preprocessing**:
   - Split the data into training, development, and test sets.
   - Tokenize the text and convert it into sequences.
   - Pad the sequences to ensure uniform length.

2. **Model Design**:
   - Embed the words into dense vectors.
   - Use LSTM layers to capture sequential information.
   - Apply dropout to prevent overfitting.
   - Add fully connected layers for classification.

3. **Model Training**:
   - Compile the model using the Adam optimizer and binary cross-entropy loss.
   - Train the model on the training data and validate it on the development set.

4. **Evaluation**:
   - Evaluate the model on the test set.
   - Analyze performance using accuracy, confusion matrix, and visualization.

5. **Visualization**:
   - Plot confusion matrices and accuracy comparison charts.

---

## Model Architecture

- **Embedding Layer**: Converts words into dense vectors.
- **LSTM Layers**: Extracts sequential patterns.
- **Dropout Layers**: Prevents overfitting.
- **Dense Layers**: Fully connected layers for binary classification.
- **Activation Functions**:
  - ReLU for intermediate layers.
  - Sigmoid for the final output.

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.0+
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detection-lstm.git
   cd fake-news-detection-lstm
   ```
   
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset file (`news.csv`) in the project directory.

4. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

---

## Usage

1. Open the `main.ipynb` file in Jupyter Notebook.

2. Follow the code blocks to:
   - Preprocess the data
   - Train the model
   - Evaluate the model
   - Visualize the results

3. View the confusion matrices and accuracy charts in the notebook output.

---

## Results

- **Development Set Accuracy**: 82.46%
- **Test Set Accuracy**: 79.34%

Confusion matrices and accuracy comparison charts are generated during evaluation for better insights.

---

## Future Improvements

- Incorporate additional preprocessing steps (e.g., stopword removal, lemmatization).
- Experiment with advanced architectures like Bidirectional LSTM or transformers.
- Add explainability features using SHAP or LIME.
- Fine-tune hyperparameters using Keras Tuner or grid search.
- Add a user-friendly front-end for live news classification.

---

## Acknowledgments

- Dataset source: [DataFlair's Advanced Python Project: Detecting Fake News](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/).
- TensorFlow and Keras libraries for model implementation.
- Matplotlib and Seaborn for data visualization.

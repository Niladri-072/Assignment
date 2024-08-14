# BERT-based Sentiment Analysis for Movie Reviews

This repository contains a project that implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers) to classify movie reviews as positive or negative. The model is trained on the IMDB movie reviews dataset and includes an interactive interface for real-time predictions.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup and Execution](#setup-and-execution)
- [Usage](#usage)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Introduction

This project leverages the power of BERT for sentiment analysis on movie reviews. The model is trained on the IMDB dataset and fine-tuned to classify reviews into positive or negative sentiments. The project is deployed with an interactive interface, allowing users to input their own reviews and get real-time predictions.

## Requirements

To run this project, you need:

- A Google Colab account
- A Hugging Face account and API token

## Setup and Execution

Follow these steps to set up and run the project in Google Colab:

1. **Create a New Notebook:**
   - Open Google Colab and create a new notebook.

2. **Install Required Libraries:**
   - Run the following command to install necessary libraries:
     ```python
     !pip install transformers tensorflow keras-tuner
     ```

3. **Import Necessary Libraries:**
   - Import the required libraries with the following code:
     ```python
     import pandas as pd
     import numpy as np
     import transformers
     import tensorflow as tf
     from keras_tuner.tuners import RandomSearch
     import ipywidgets as widgets
     from IPython.display import display, clear_output
     import matplotlib.pyplot as plt
     from transformers import BertTokenizer, TFBertForSequenceClassification
     from sklearn.model_selection import train_test_split
     from huggingface_hub import login
     ```

4. **Authenticate with Hugging Face:**
   - Authenticate using your Hugging Face API token:
     ```python
     login(token="your_huggingface_token", add_to_git_credential=True)
     ```
   - Replace `your_huggingface_token` with your actual Hugging Face API token.

5. **Load and Preprocess the IMDB Dataset:**
   - Upload the IMDB dataset CSV file to your Google Colab environment.
   - Load and preprocess the dataset according to the provided code.

6. **Train and Fine-Tune the Model:**
   - Run the training and fine-tuning code cells in your Colab notebook to train the model and optimize hyperparameters.

7. **Deploy the Model:**
   - Implement the deployment code using `ipywidgets` for an interactive user interface.
   - Run the deployment cells to create the UI for real-time predictions.

## Usage

After running all cells in the Colab notebook, you'll see an interactive interface with a text area and a "Predict" button. Simply enter a movie review in the text area and click "Predict" to get the sentiment analysis result.

## Notes

- The model training process may be time-consuming. Using a GPU runtime in Colab is recommended for faster execution.
- Adjust file paths if your dataset or saved model is stored in different locations within the Colab environment.
- The notebook includes optional data exploration and visualization steps for gaining insights into the dataset.

## Troubleshooting

If you encounter any issues:

- Ensure that all required libraries are installed correctly.
- Verify that your Hugging Face token is valid and you have the necessary permissions.
- Confirm that the IMDB dataset is correctly uploaded and accessible in your Colab environment.

For persistent issues, consult the documentation for [Google Colab](https://colab.research.google.com/), [Hugging Face](https://huggingface.co/docs), and [TensorFlow](https://www.tensorflow.org/).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

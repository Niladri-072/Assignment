```
# BERT-based Sentiment Analysis for Movie Reviews

## Introduction

This project implements a sentiment analysis model using BERT (Bidirectional Encoder Representations from Transformers) to classify movie reviews as positive or negative. The model is trained on the IMDB movie reviews dataset and deployed with an interactive interface for real-time predictions.

## Requirements

- Google Colab account
- Hugging Face account and API token

## Setup and Execution in Google Colab

1. Create a new notebook in Google Colab.

2. Install required libraries:

   ```python
   !pip install transformers tensorflow keras-tuner
   ```

3. Import necessary libraries:

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

4. Authenticate with Hugging Face:

   ```python
   login(token="your_huggingface_token", add_to_git_credential=True)
   ```

   Replace `your_huggingface_token` with your actual Hugging Face API token.

5. Load and preprocess the IMDB dataset:

   - Upload the IMDB Dataset CSV file to your Google Colab environment.
   - Load the dataset and preprocess it as shown in the code.

6. Train and fine-tune the model:

   - Copy the training and fine-tuning code sections into your Colab notebook.
   - Run the cells to train the model and find the best hyperparameters.

7. Deploy the model:

   - Implement the deployment code with ipywidgets for the interactive interface.
   - Run the deployment cells to create the user interface for predictions.

## Usage

After running all the cells, you'll see a text area and a "Predict" button. Enter a movie review in the text area and click "Predict" to get the sentiment analysis result.

## Notes

- The model training process can be time-consuming. Consider using a GPU runtime in Colab for faster execution.
- Make sure to adjust file paths if you store the dataset or save the model in different locations within your Colab environment.
- The provided code includes data exploration and visualization. You can run these cells to gain insights into the dataset.

## Troubleshooting

If you encounter any issues:
- Ensure all required libraries are installed.
- Check that your Hugging Face token is correct and you have the necessary permissions.
- Verify that the IMDB dataset is correctly uploaded and accessible in your Colab environment.

For any persistent problems, refer to the Colab, Hugging Face, and TensorFlow documentation.
```

This README provides a step-by-step guide on how to set up and run the project in Google Colab. It covers the installation of required libraries, data preparation, model training, and deployment of the interactive interface. Users can follow these instructions to replicate the project in their own Colab environment.

# Machine Learning
2024-2 Assessmet task 3

This project implements a fake news detection model using a combination of LSTM for processing news titles and a Dense layer for incorporating source domain information. The model is trained on the FakeNewsNet dataset.

[View Notebook on GitHub](https://github.com/sohyun902/Machine-Learning/blob/main/ML_project_Fake%20news%20detection.ipynb)

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Implementation](#model-implementation)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)

## Project Overview

The goal of this project is to build a model that can classify news articles as real or fake based on their title and source domain. The approach involves:

1.  Loading and preprocessing the data.
2.  Using Word2Vec to create embeddings for the news titles.
3.  One-hot encoding the source domains.
4.  Building a hybrid model that combines an LSTM for title processing and a Dense layer for source information.
5.  Training and evaluating the model.

## Data

The dataset used is "FakeNewsNet.csv", which is loaded from Google Drive. The dataset contains the following columns:

-   `title`: The title of the news article.
-   `news_url`: The URL of the news article.
-   `source_domain`: The domain of the news source.
-   `tweet_num`: The number of tweets related to the article.
-   `real`: The target variable, indicating whether the news is real (1) or fake (0).

## Preprocessing

The data undergoes the following preprocessing steps:

-   Handling duplicate titles by dropping them.
-   Handling missing values by dropping rows with missing `news_url` or `source_domain`.
-   Analyzing the class distribution to identify data imbalance.
-   Creating a `preprocessed_title` column by:
    -   Removing punctuation.
    -   Converting to lowercase.
    -   Lemmatizing words.
    -   Removing stop words.
-   Extracting the main part of the `source_domain`.
-   Splitting the data into training and testing sets.
-   Computing class weights to address data imbalance.
-   Generating Word2Vec embeddings for the preprocessed titles.
-   Padding the title sequences to a fixed length.
-   One-hot encoding the `source_domain`.

## Model Implementation

Two models are implemented and evaluated:

1.  **Title-only model**: An LSTM model that uses only the preprocessed title as input.
2.  **Title + Source model**: A hybrid model that concatenates the output of an LSTM processing the title and a Dense layer processing the one-hot encoded source domain.

Both models use a sigmoid activation function in the output layer for binary classification and are compiled with the Adam optimizer and binary crossentropy loss. Early stopping is used during training to prevent overfitting.

## Evaluation

The models are evaluated using the `classification_report` from scikit-learn, which provides precision, recall, and F1-score for each class, as well as overall accuracy. The loss curves for training and validation sets are also plotted to visualize the training progress.

## Saving the Model

The trained "Title + Source" model is saved to Google Drive in HDF5 format.

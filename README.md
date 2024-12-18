# Sentiment_Analysis
# Project Overview
The Sentiment Analysis project utilizes the IMDB dataset, which contains 50,000 movie reviews for natural language processing (NLP) and text analytics. This dataset is used for binary sentiment classification, providing 25,000 highly polarized movie reviews for training and 25,000 for testing. The goal of the project is to predict the sentiment of movie reviews as either positive or negative using machine learning algorithms.
# Features
- **Text Preprocessing**: Data cleaning, tokenization, and stop word removal of the text data to prepare it for analysis.
- **Visualization**: Insights into the sentiment distribution using pie charts and histograms.
- **Sentiment Classification**: Implementation of binary sentiment classification with two machine learning algorithms: Naive Bayes and Random Forest.
- **Evaluation**: Model performance evaluation using accuracy and other metrics.
- **Data**: 50,000 movie reviews split into 25,000 training and 25,000 testing reviews.
# Requirements
1. Python 3.x
2. numpy
3. pandas
4. matplotlib
5. seaborn
6. scikit-learn
7. nltk
# Usage
1. **Preprocess the dataset:**

   - Clean and tokenize the text reviews.
   - Perform lemmatization and remove stopwords.
3. **Train the models:**
    - he project implements two classification models: Naive Bayes and Random Forest.
    - Use the training data to train the models.
4. **Evaluate the models:**
    - After training, evaluate the models using accuracy, confusion matrix, and other classification metrics.
5. **Visualize results:**
     - View sentiment distribution in pie charts and analyze the frequency of review lengths and sentiments in histograms.
# Model Training
The training process involves:

    1. **Text Preprocessing:** Cleaning the dataset, tokenizing the text, and performing remove stop words.
    2. **Feature Extraction:** Converting text data into numerical vectors using techniques like Bag of Words or TF-IDF.
    3. **Model Training:** Training both Naive Bayes and Random Forest classifiers.
    4. **Hyperparameter Tuning:** Optionally tuning model parameters for optimal performance.
    5. **Evaluation:** Using metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
    
# Evaluation

After training, the models are evaluated using the following metrics:

    - **Accuracy:** The percentage of correct predictions.
    - **Confusion Matrix:** To visualize the performance of the models.
    - **Precision, Recall, F1-score:** For a deeper analysis of the model's classification performance.
    
# Contributing

Contributions are welcome! Feel free to open issues, discuss ideas, or submit pull requests.




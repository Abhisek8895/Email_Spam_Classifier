# Email Spam Classification

This project classifies emails as spam or not spam using a machine learning model based on the Naive Bayes algorithm. The email texts are vectorized using the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the text data into numerical form before applying the classification model.

## Dataset

The dataset used contains two columns:
- `label`: Indicates whether the email is spam (1) or not spam (0).
- `text`: The body of the email.

## Project Workflow

1. **Data Loading and Preprocessing**
   - The dataset is loaded using pandas.
   - Basic information about the dataset is displayed using `.head()` and `.info()`.
   - Checks for missing values and class distribution.
   - A new feature, `word_count`, is added to analyze the length of emails in terms of the number of words.

2. **TF-IDF Vectorization**
   - A `TfidfVectorizer` is used to convert the email text into numerical form.
   - The maximum number of features used by the vectorizer is set to 3000.

3. **Splitting the Dataset**
   - The dataset is split into training (80%) and testing (20%) subsets using `train_test_split`.

4. **Model Selection and Training**
   - A Naive Bayes model (`MultinomialNB`) is selected due to its effectiveness in text classification tasks.
   - The model is trained on the training data.

5. **Evaluation**
   - The model’s accuracy on the test data is evaluated using:
     - **Accuracy Score**
     - **Classification Report** (Precision, Recall, F1-Score)
     - **Confusion Matrix** with a heatmap for visualization.

6. **Spam Classification Example**
   - A sample email text is classified as spam or not spam using the trained model.

7. **Metrics**
   - Accuracy of the model: **95%**
   - F1 score: **Calculated and displayed in the output**.

## Files

- `email_text.csv`: The dataset used for training and testing.
- `email spam classifier.ipynb`: The Python script containing the complete code for data preprocessing, model training, evaluation, and prediction.
- `README.md`: This file, providing an overview of the project.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Abhisek8895/Email_Spam_Classifier.git
   cd Email_Spam_Classifier
   ```

2. Install the required Python libraries:

   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

3. Ensure the dataset (`email_text.csv`) is in the project directory.

## Usage

1. Run the Python script:

   ```bash
   python spam_classification.py
   ```

2. The model will output the accuracy, classification report, confusion matrix, and F1 score. You can also classify new emails by modifying the `new_email` variable in the script.

## Visualization

The confusion matrix is visualized as a heatmap using Seaborn:

![Confusion Matrix](image.png)

## Model Performance

- **Accuracy**: 95%
- **F1 Score**: Displayed in the output.

## Conclusion

The Naive Bayes model performs well on this dataset, achieving a high accuracy in classifying emails as spam or not spam. The use of TF-IDF vectorization effectively converts the email text into numerical features suitable for the Naive Bayes algorithm.
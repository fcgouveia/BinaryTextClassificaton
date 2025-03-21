import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Read the training TSV file
train_df = pd.read_csv('trainset.tsv', sep='\t', header=None, names=['text', 'label'])

# Ensure all texts are strings
train_df['text'] = train_df['text'].astype(str)

# Split the data into texts and labels
texts = train_df['text'].tolist()
labels = train_df['label'].tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define classifiers with improved parameters to fix convergence warnings
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(dual='auto', max_iter=10000, tol=1e-4),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Dictionary to store accuracy scores
accuracy_scores = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train_vectorized, y_train)
    y_pred = clf.predict(X_test_vectorized)
    
    # Store accuracy score
    current_accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = current_accuracy
    
    print(f"Results for {name}:")
    print(f"Accuracy: {current_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Type A", "Type A"]))

# Choose the best performing classifier
best_classifier_name = max(accuracy_scores, key=accuracy_scores.get)
best_classifier_model = classifiers[best_classifier_name]
print(f"\nBest performing classifier: {best_classifier_name} with accuracy: {accuracy_scores[best_classifier_name]:.4f}")

# Function to classify new texts using the best classifier
def classify_text(text, clf):
    text = str(text)
    text_vectorized = vectorizer.transform([text])
    prediction = clf.predict(text_vectorized)
    return "Type A" if prediction[0] == 1 else "Not Type A"

# Read the TSV file with texts to classify
to_classify_df = pd.read_csv('to_classify.tsv', sep='\t', header=None, names=['text'])

# Ensure all texts to classify are strings
to_classify_df['text'] = to_classify_df['text'].astype(str)

# Classify each text using the best classifier
to_classify_df['classification'] = to_classify_df['text'].apply(lambda x: classify_text(x, best_classifier_model))

# Save the results to a new TSV file
to_classify_df.to_csv('classified_results.tsv', sep='\t', index=False)
print("\nClassification complete. Results saved to 'classified_results.tsv'")
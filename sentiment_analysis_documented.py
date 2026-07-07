# ==================== CODE CELL 1 ====================
# Purpose: Converted from Jupyter notebook.
# 1 — Imports and load cleaned data
import pandas as pd

df = pd.read_excel("dataset -P684.xlsx")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==================== CODE CELL 2 ====================
# Purpose: Converted from Jupyter notebook.
#2 — NLTK imports and setup
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==================== CODE CELL 3 ====================
# Purpose: Converted from Jupyter notebook.
# 3 — clean_text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ==================== CODE CELL 4 ====================
# Purpose: Converted from Jupyter notebook.
#4 - Import Libraries for Model Building
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==================== CODE CELL 5 ====================
# Purpose: Converted from Jupyter notebook.
#5 - Split the dataset into training and testing sets
X=df['clean_text']
y=df['sentiment']

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==================== CODE CELL 6 ====================
# Purpose: Converted from Jupyter notebook.
#6 - Convert text to Number using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)

# ==================== CODE CELL 7 ====================
# Purpose: Converted from Jupyter notebook.
#7 - Model 1 Train the Logistic Regression model
model= LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model train Successfully")

# ==================== CODE CELL 8 ====================
# Purpose: Converted from Jupyter notebook.
#8 - Make predictions on the test set
y_pred=model.predict(X_test_tfidf)
print("Sample Predictions:", y_pred[:10])
print("Actual labels:", y_test[:10].values)

# ==================== CODE CELL 9 ====================
# Purpose: Converted from Jupyter notebook.
#9 - Calculate the overall accuracy 
accuracy=accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

# ==================== CODE CELL 10 ====================
# Purpose: Converted from Jupyter notebook.
#10 - Classification Report(per-class performance)
print(classification_report(y_test, y_pred))

# ==================== CODE CELL 11 ====================
# Purpose: Converted from Jupyter notebook.
#11 — Confusion matrix (visual)
cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 12 ====================
# Purpose: Converted from Jupyter notebook.
#12 - Test the model on custom review
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

sample_review = "The camera quality is okay, not great but not bad either"
print("Review:", sample_review)
print("Predicted Sentiment:", predict_sentiment(sample_review))

# ==================== CODE CELL 13 ====================
# Purpose: Converted from Jupyter notebook.
#13 - Model 2 Train the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

nb_model= MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

nb_pred=nb_model.predict(X_test_tfidf)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))

# ==================== CODE CELL 14 ====================
# Purpose: Converted from Jupyter notebook.
#14 - Naive Bayes Classification Report
print(classification_report(y_test, nb_pred))

# ==================== CODE CELL 15 ====================
# Purpose: Converted from Jupyter notebook.
#15 - Naive Bayes Confusion Matrix (visual)
cm_nb=confusion_matrix(y_test, nb_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 16 ====================
# Purpose: Converted from Jupyter notebook.
#16 - Model 3 Train the Random Forest model
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

rf_pred = rf_model.predict(X_test_tfidf)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# ==================== CODE CELL 17 ====================
# Purpose: Converted from Jupyter notebook.
#17 - Random Forest Classification Report
print(classification_report(y_test, rf_pred))

# ==================== CODE CELL 18 ====================
# Purpose: Converted from Jupyter notebook.
#18 - Random Forest Confusion Matrix (visual)
cm_rf = confusion_matrix(y_test, rf_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix — Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 19 ====================
# Purpose: Converted from Jupyter notebook.
#19 - Model 4 Support Vector Machine (SVM)
from sklearn.svm import LinearSVC

svm_model=LinearSVC(max_iter=1000, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

svm_pred=svm_model.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# ==================== CODE CELL 20 ====================
# Purpose: Converted from Jupyter notebook.
#20 - SVM Classification report
print(classification_report(y_test, svm_pred))

# ==================== CODE CELL 21 ====================
# Purpose: Converted from Jupyter notebook.
#21 - SVM Confusion matrix (visual)
cm_svm=confusion_matrix(y_test, svm_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix --SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 22 ====================
# Purpose: Converted from Jupyter notebook.
#22 - Final Model Comparison
results={
    'Model':['Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM'],
    'Accuracy':[
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, nb_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, svm_pred)                                                
    ]
}

import pandas as pd
results_df=pd.DataFrame(results)
results_df['Accuracy']=results_df['Accuracy'].apply(lambda x: f"{x:2%}")
print(results_df.to_string(index=False))

# ==================== CODE CELL 23 ====================
# Purpose: Converted from Jupyter notebook.
#23 - Final Model Comparison (visual)
models = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM']
accuracies = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, svm_pred)
]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracies, 
               color=['#3498db','#e74c3c','#2ecc71','#9b59b6'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.60, 0.85)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 0.002,
             f'{acc:.2%}', 
             ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== CODE CELL 24 ====================
# Purpose: Converted from Jupyter notebook.
pip install xgboost vaderSentiment

# ==================== CODE CELL 25 ====================
# Purpose: Converted from Jupyter notebook.
#24 - Model 5 VADER (Lexicon Based)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

analyzer = SentimentIntensityAnalyzer()

def vader_predict(text):
    score= analyzer.polarity_scores(str(text))
    if score['compound']>=0.05:
        return 'positive'
    elif score['compound']<=-0.05:
        return 'negative'
    else:
        return 'neutral'
    
vader_pred=df['full_text'].apply(vader_predict)
vader_pred_test = vader_pred.iloc[X_test.index]

print("VADER Accuracy:", accuracy_score(y_test, vader_pred_test))

# ==================== CODE CELL 26 ====================
# Purpose: Converted from Jupyter notebook.
#25 - VADER Classification Report
print(classification_report(y_test, vader_pred_test))

# ==================== CODE CELL 27 ====================
# Purpose: Converted from Jupyter notebook.
#26 - VADER Confusion Matrix (visual)
cm_vader = confusion_matrix(y_test, vader_pred_test, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_vader, annot=True, fmt='d', cmap='YlOrBr',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix — VADER')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 28 ====================
# Purpose: Converted from Jupyter notebook.
#27 - Model 6 XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train_tfidf, y_train_encoded)

xgb_pred_encoded = xgb_model.predict(X_test_tfidf)
xgb_pred = le.inverse_transform(xgb_pred_encoded)

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

# ==================== CODE CELL 29 ====================
# Purpose: Converted from Jupyter notebook.
#28 - XGBoost Classification Report
print(classification_report(y_test, xgb_pred))

# ==================== CODE CELL 30 ====================
# Purpose: Converted from Jupyter notebook.
#29 - XGBoost Confusion Matrix (visual)
cm_xgb = confusion_matrix(y_test, xgb_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='copper',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix — XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 31 ====================
# Purpose: Converted from Jupyter notebook.
#30 - model 7 ANN(Artificial Neural Network)
from sklearn.neural_network import MLPClassifier

ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), 
                          max_iter=500, 
                          random_state=42,
                          activation='relu',
                          solver='adam')

ann_model.fit(X_train_tfidf, y_train)
ann_pred = ann_model.predict(X_test_tfidf)

print("ANN Accuracy:", accuracy_score(y_test, ann_pred))

# ==================== CODE CELL 32 ====================
# Purpose: Converted from Jupyter notebook.
#31 - ANN Classification Report
print(classification_report(y_test, ann_pred))

# ==================== CODE CELL 33 ====================
# Purpose: Converted from Jupyter notebook.
#32 — ANN Confusion Matrix (visual)
cm_ann = confusion_matrix(y_test, ann_pred, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='RdPu',
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix — ANN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==================== CODE CELL 34 ====================
# Purpose: Converted from Jupyter notebook.
#33 Final Comparison — All 7 Models
all_models = ['Logistic\nRegression', 'Naive\nBayes', 'Random\nForest', 
              'SVM', 'VADER', 'XGBoost', 'ANN']
all_accuracies = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, vader_pred_test),
    accuracy_score(y_test, xgb_pred),
    accuracy_score(y_test, ann_pred)
]

plt.figure(figsize=(12,6))
bars = plt.bar(all_models, all_accuracies,
               color=['#3498db','#e74c3c','#2ecc71','#9b59b6',
                      '#e67e22','#1abc9c','#e91e63'])
plt.title('All Models — Accuracy Comparison', fontsize=14)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.60, 0.85)

for bar, acc in zip(bars, all_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.002,
             f'{acc:.2%}',
             ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ==================== CODE CELL 35 ====================
# Purpose: Converted from Jupyter notebook.


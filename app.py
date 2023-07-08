
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
import nltk, random
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
#nltk.download('punkt')
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from flask import Flask, request, jsonify, render_template
import pickle

# Import dataset
url = 'https://raw.githubusercontent.com/taoki2/C964/main/data.csv'
df=pd.read_csv(url)

# Convert rating to int data type
df.rating = df.rating.astype(int)

# Create label column based on rating
df['label'] = np.where(df['rating'] >= 4, 1, 0)

# Check for missing values
df.isnull().sum()

# Descriptive statistics of the dataset
df.describe()

# Plot distribution of ratings
sns.displot(df, x='rating', discrete=True)

# Plot distribution of review sentiment
sns.displot(df, x='label', discrete=True, bins=[0,1])
X = df.drop(['rating', 'label'], axis=1)
y = df.drop(['rating', 'review'], axis=1)

# Balance classes
print(X.shape, y.shape)
under_sampler = RandomUnderSampler(random_state=222)
X, y = under_sampler.fit_resample(X, y)
print(X.shape, y.shape)

# Combine X and y to create a new dataframe
df = pd.concat([X,y], axis=1)
df.info()
pd.set_option('display.max_colwidth', None)
df.head(1)

# Remove the "READ MORE" string in the reviews
df['review'] = df['review'].str.replace('READ MORE','')

# Remove special characters
def clean_text(text):
    review = df['review']
    list_char = []
    for rev in review:
        for char in rev:
            if char not in list_char:
                list_char.append(char)
    special_char = []
    pattern = re.compile('[^a-zA-Z]')
    for char in list_char:
        if pattern.match(char):
            special_char.append(char)
    special_char.remove(' ')
    remove_punc = [char for char in text if char not in special_char]
    remove_punc = ''.join(remove_punc)
    return remove_punc.lower()

'''
# Clean text in reviews
df['review'] = df['review'].apply(clean_text)
df.head(10)

# Split the data into training and testing sets
X = df['review']
y = np.array(df[['label']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Perform text vectorization
tfidf_vec = TfidfVectorizer(min_df = 10, token_pattern = r'[a-zA-Z]+')
X_train_bow = tfidf_vec.fit_transform(X_train)
X_test_bow = tfidf_vec.transform(X_test)
print(X_train_bow.shape, X_test_bow.shape)


# Build the support vector machine (SVM) model
model_svm = svm.SVC(C=8.0, kernel='linear')
model_svm.fit(X_train_bow, y_train.ravel())

# Cross validation scores
model_svm_acc = cross_val_score(estimator=model_svm, X=X_train_bow, y=y_train, cv=5, n_jobs=-1)
print(model_svm_acc)

# Initial model's accuracy
print(model_svm.score(X_test_bow, y_test))

# Perform GridSearch to tune the model's hyperparameters
parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=10, n_jobs=-1)
clf.fit(X_train_bow, y_train.ravel())
print('Best parameters: ', clf.best_params_)


# Build new model with parameters identified from GridSearch
model_svm = svm.SVC(C=1.0, kernel='rbf')
model_svm.fit(X_train_bow, y_train.ravel())

# New model's accuracy
print(model_svm.score(X_test_bow, y_test))

# Confusion matrix
y_pred = model_svm.predict(X_test_bow)
cm = confusion_matrix(y_test, y_pred, normalize='all')
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# Save the model
pickle.dump(model_svm, open('model_svm.pkl', 'wb'))

'''

# Flask app
app = Flask(__name__)

def load_model():
    global loaded_model
    loaded_model = pickle.load(open('model_svm.pkl', 'rb'))

load_model()

@app.route('/')
def home():
    return render_template('index.html')

def predict_text(text):
    text = [text]
    predict_bow = tfidf_vec.transform(text)
    prediction = loaded_model.predict(predict_bow)
    return str(prediction)

@app.route('/',methods=['POST'])
def post_form():
    text = request.form['text'].lower()
    result = predict_text(text)
    if result == "[0]":
        output = "Negative"
    else:
        output = "Positive"
    return render_template('index.html', variable=output)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
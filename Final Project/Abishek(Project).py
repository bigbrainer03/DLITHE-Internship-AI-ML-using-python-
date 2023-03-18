# **************************
# ********** QS 1 **********
# **************************

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# READ THE DATA FROM THE DATASET PROVIDED
data=pd.read_csv('C:/Users/hp/Downloads/breast-cancer.csv')

# ADD THE LABELS COLUMN TO THE DATASET
data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

# THIS IS TO CONVERT THE STRING VALUES TO FLOAT FOR FITTING THE MODELS
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['age'] = le.fit_transform(data['age'])
data['menopause'] = le.fit_transform(data['menopause'])
data['tumor-size'] = le.fit_transform(data['tumor-size'])
data['node-caps'] = le.fit_transform(data['node-caps'])
data['breast'] = le.fit_transform(data['breast'])
data['breast-quad'] = le.fit_transform(data['breast-quad'])
data['irradiat'] = le.fit_transform(data['irradiat'])
data['inv-nodes'] = le.fit_transform(data['inv-nodes'])

# PRE-PROCESS THE DATA AND SPLIT IT INTO TRAINING AND TESTING SETS
data=data.dropna()
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# IMPLEMENT THE VARIOUS MODELS
logistic_regression=LogisticRegression(random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
naive_bayes=GaussianNB()

# TRAINING THE MODELS
logistic_regression.fit(X_train, y_train)
knn.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# TESTING THE MODELS
y_pred_log_reg = logistic_regression.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_nb = naive_bayes.predict(X_test)

# CREATING A CONFUSION MATRIX TO COMPARE THE MODELS
cm_logreg = confusion_matrix(y_test, y_pred_log_reg)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_nb = confusion_matrix(y_test, y_pred_nb)

# THIS GIVES A TABLE TO SUMMARIZE THE THREE MODELS USED
data = {
    'Model': ['Logistic Regression', 'K-Nearest Neighbors', 'Naive Bayes'],
    'True Positive': [cm_logreg[0][0], cm_knn[0][0], cm_nb[0][0]],
    'False Positive': [cm_logreg[0][1], cm_knn[0][1], cm_nb[0][1]],
    'True Negative': [cm_logreg[1][1], cm_knn[1][1], cm_nb[1][1]],
    'False Negative': [cm_logreg[1][0], cm_knn[1][0], cm_nb[1][0]],
    'Accuracy': [(cm_logreg[0][0] + cm_logreg[1][1])/len(y_test), 
                 (cm_knn[0][0] + cm_knn[1][1])/len(y_test),
                 (cm_nb[0][0] + cm_nb[1][1])/len(y_test)],
    'Precision': [cm_logreg[0][0]/(cm_logreg[0][0]+cm_logreg[0][1]), 
                  cm_knn[0][0]/(cm_knn[0][0]+cm_knn[0][1]), 
                  cm_nb[0][0]/(cm_nb[0][0]+cm_nb[0][1])],
    'Recall': [cm_logreg[0][0]/(cm_logreg[0][0]+cm_logreg[1][0]), 
               cm_knn[0][0]/(cm_knn[0][0]+cm_knn[1][0]), 
               cm_nb[0][0]/(cm_nb[0][0]+cm_nb[1][0])],
    'F1 Score': [(2*cm_logreg[0][0])/(2*cm_logreg[0][0]+cm_logreg[0][1]+cm_logreg[1][0]), 
                 (2*cm_knn[0][0])/(2*cm_knn[0][0]+cm_knn[0][1]+cm_knn[1][0]), 
                 (2*cm_nb[0][0])/(2*cm_nb[0][0]+cm_nb[0][1]+cm_nb[1][0])]
}
df = pd.DataFrame(data)

print(df)

# **************************
# ********** QS 2 ********** 
# **************************

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# TO ACCESS THE CONTENTS FROM THE LINK PROVIDED
link = 'https://monkeylearn.com/sentiment-analysis/'
response = requests.get(link)
soup = BeautifulSoup(response.content, 'html.parser')

# TO GET JUST THE PARAGRAPHS FROM THE WEBPAGE
text = ''
for paragraph in soup.find_all('p'):
    text += paragraph.get_text()

# USING NLTK (VADER LEXICON) WHICH IS USED SPECIFICALLY FOR SENTIMENT ANALYSIS
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text)
print("Text:\n ", text)

# TO DETERMINE THE POLARITY OF THE DATA
print("Sentiment:",end = ' ')
if sentiment['compound'] > 0:
    print('Positive')
elif sentiment['compound'] < 0:
    print('Negative')
else:
    print('Neutral')


# **************************
# ********** QS 3 **********
# **************************

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

data = pd.read_csv('C:/Users/hp/Downloads/CC GENERAL.csv')
print(data.head())
print(data.describe())

# TO CLEAN THE DATA AND DROP UNNECESSARY COLUMNS
data = data.drop(['CUST_ID'], axis = 1)
data = data.fillna(method='ffill')

# SCALING THE DATA
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

label_enc = LabelEncoder()
data_scaled[:, 4] = label_enc.fit_transform(data_scaled[:, 4])
data_scaled[:, 5] = label_enc.fit_transform(data_scaled[:, 5])

# PERFORMING THE CLUSTERING AND GETTING RESULTS ALONG WITH A PLOT TO VISUALIZE OUR RESULTS
kmeans = KMeans(n_clusters=5, max_iter=50)
kmeans.fit(data_scaled)
data['cluster'] = kmeans.labels_
data['cluster'].value_counts()
sns.scatterplot(x="PURCHASES", y="CASH_ADVANCE", hue="cluster", data=data)
plt.show()

# **************************
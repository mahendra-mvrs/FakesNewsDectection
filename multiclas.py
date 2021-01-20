import pandas as pd
import nltk
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
import re

nltk.download("stopwords")

df = pd.read_csv("C:/Users/Raghushyam/Desktop/data.csv")
df.head()

# preprosessing

corpus = []
for i in range(0, 3554):
    text = re.sub("[^a-zA-Z]", "  ", str(df['text'][i]))
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words("english"))]
    text = " ".join(text)
    corpus.append(text)
    
# x & y division    

X = corpus
y = df.iloc[:, 3].values

# test and tarining set 

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# feature extractions

cv = CountVectorizer()
cv.fit_transform(train_X)

#saving the model
import pickle
pickle.dump(cv, open('cv.pkl','wb'))

#loading model

multicv = pickle.load(open('cv.pkl', 'rb'))

Xtrain = cv.transform(train_X).toarray()
Xtest = cv.transform(test_X).toarray()

svm=SVC()
sgd=SGDClassifier(loss = "modified_huber", shuffle= True,random_state=101)

 # ----------------------------------------------------------------------------#

# gaussian model 
gpc= GaussianProcessClassifier()
gpc.fit(Xtrain,train_y)

# creating model for gpc
pickle.dump(gpc, open('gpc.pkl','wb'))

#loading model

gpc_model = pickle.load(open('gpc.pkl', 'rb'))

 # ----------------------------------------------------------------------------#

#sgd with baggign modle
sgd_bagging_classifier =BaggingClassifier(sgd,max_samples =0.2, max_features =1.0)
sgd_bagging_classifier.fit(Xtrain,train_y)

#creating model for sgd bagging.
pickle.dump(sgd_bagging_classifier, open('sgd_bagging.pkl','wb'))

#loading model

sgd_baggingclassifer = pickle.load(open('sgd_bagging.pkl', 'rb'))

 # ----------------------------------------------------------------------------#

#svm with bagging model
svm_bagging_classifier =BaggingClassifier(svm,max_samples =0.2, max_features =1.0)
svm_bagging_classifier.fit(Xtrain,train_y)
      
#creating model for svm bagging.
pickle.dump(svm_bagging_classifier, open('svm_bagging.pkl','wb'))

#loading model

svm_baggingclassifier = pickle.load(open('svm_bagging.pkl', 'rb'))

 # ----------------------------------------------------------------------------#


multi_classifier = VotingClassifier( estimators = [('svm_bagging_classifier',svm_bagging_classifier),('sgd_bagging_classifier',sgd_bagging_classifier),('gpc',gpc)],voting ='hard')
multi_classifier.fit(Xtrain,train_y)

#saving the model
import pickle
pickle.dump(multi_classifier, open('multiclas.pkl','wb'))

#loading model

multiclas = pickle.load(open('multiclas.pkl', 'rb'))

 # ----------------------------------------------------------------------------#

# testing 
df = pd.read_excel("C:/Users/Raghushyam/Desktop/dataset .xlsx")
texts = [df["text"][200]]
cvpred = multicv.transform(texts).toarray()
predictions = gpc_model.predict(cvpred)
for text, predicted in zip(texts, predictions):
    print('"{}"'.format(text))
    print("  - Predicted as: '{}'".format([predicted]))
    print("")


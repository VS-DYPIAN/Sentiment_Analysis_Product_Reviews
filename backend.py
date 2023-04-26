import streamlit as st
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
import time
from nltk.corpus import stopwords
import seaborn as sns
import numpy as np
import string
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk import word_tokenize
import imblearn
from imblearn.over_sampling import RandomOverSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline



nltk.download("wordnet")
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stopwords = stopwords.words('english')

models = ("BernoulliNB",
          "Logistic Regression",
          "GradientBoostingClassifier",
          "LinearSVC",
          'Distilbert Pipeline')

def load_dataset():
    path = "data/amazon_reviews.csv"
    return pd.read_csv(path)



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def model_evaluate(model,X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    classification_dict = classification_report(y_test, y_pred,output_dict = True)
    try:
        Pr_train = model.predict_proba(X_train)[:,1]
        Pr_test = model.predict_proba(X_test)[:,1]
        
        train_AUC = roc_auc_score(y_train,Pr_train)
        test_AUC = roc_auc_score(y_test,Pr_test)
    except AttributeError as e:
        train_AUC = None
        test_AUC = None
    return classification_dict,train_AUC,test_AUC
        #   "Train AUC",roc_auc_score(y_train,Pr_train),
        #   "Test AUC", roc_auc_score(y_test,Pr_test))

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word,pos = get_wordnet_pos(tag)) \
                 for word,tag in words_and_tags]



def preprocessing(message):
    Test_punc_removed = [word for word in message if word not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords and word.lower().isalpha() and word is not None]
    Test_punc_removed_join_clean = ' '.join(Test_punc_removed_join_clean)
    return Test_punc_removed_join_clean





def feature_engineering(data_df):
    X = data_df['verified_reviews']
    y = data_df['feedback']
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X.values.reshape(-1, 1), y)
    X_res = np.ravel(X_res)
    X_res = pd.Series(X_res)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                        stratify=y_res, 
                                                        test_size=0.25, random_state = 245)
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test

@st.cache_resource
def confusion_matrix_func(model_name,_X_train, _X_test, _y_train, _y_test):
    with st.spinner('Generating Confusion Matrix:'):
        time.sleep(1)
        model = model_name()
        model.fit(_X_train, _y_train)
        predicted = model.predict(_X_test)
        cm = confusion_matrix(_y_test,predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)

        # set axis labels and chart title
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
    
        
    st.success('Confusion Matrix generated!')

    # display the chart in Streamlit
    return st.pyplot(fig)


@st.cache_data
def reset_feature():
    path = "data/amazon_reviews.csv"
    df = pd.read_csv(path)
    df['verified_reviews'].replace(' ', np.nan, inplace=True)
    df.dropna(subset=['verified_reviews'], inplace=True)
    X = df['verified_reviews']
    y = df['feedback']
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X.values.reshape(-1, 1), y)
    X_res = np.ravel(X_res)
    X_res = pd.Series(X_res)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                    stratify=y_res, 
                                                    test_size=0.25, random_state = 245)
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=False)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test,vectorizer


# @st.cache_resource
def load_transformer():
    sentiment_classifier = pipeline(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    task="sentiment-analysis",
    top_k=None,
)
    return sentiment_classifier
    


def predict_pipeline(model,X_train, X_test, y_train, y_test):
    y_pred = model(X_test)
    classification_dict = classification_report(y_test, y_pred,output_dict = True)
    try:
        Pr_train = model.predict_proba(X_train)[:,1]
        Pr_test = model.predict_proba(X_test)[:,1]
        
        train_AUC = roc_auc_score(y_train,Pr_train)
        test_AUC = roc_auc_score(y_test,Pr_test)
    except AttributeError as e:
        train_AUC = None
        test_AUC = None
    return classification_dict,train_AUC,test_AUC



def predict_model(model_name,user_input,X_train,X_test,y_train,y_test,vectorizer,alpha=None,C=None,max_feat=None,n_estim=None,n_jobs=None,max_iterations=None,max_lr=None):
        if model_name == BernoulliNB :
            model = model_name(alpha=alpha)
            model.fit(X_train, y_train)
            result,train_AUC,test_AUC = model_evaluate(model,X_train, X_test, y_train, y_test)
            test = pd.Series(user_input)
            # vectorizer = backend.load_vectorizer()
            pred_test = vectorizer.transform(test)
            predictions = model.predict(pred_test)
            


            # loaded_pipe = backend.load_transformer()
            # predictions = loaded_pipe.predict(user_input)
            pred_to_label = {0: 'Negative', 1: 'Positive'}

            # Make a list of user_text with sentiment.
            data = []
            for t, pred in zip(user_input, predictions):
                data.append((t, pred, pred_to_label[pred]))
        
        elif model_name == LogisticRegression:
            model = model_name(C = C,max_iter = max_lr,n_jobs = n_jobs)
            model.fit(X_train, y_train)
            result,train_AUC,test_AUC= model_evaluate(model,X_train, X_test, y_train, y_test)
            test = pd.Series(user_input)
            pred_test = vectorizer.transform(test)
            predictions = model.predict(pred_test)
            pred_to_label = {0: 'Negative', 1: 'Positive'}

            # Make a list of user_text with sentiment.
            data = []
            for t, pred in zip(user_input, predictions):
                data.append((t, pred, pred_to_label[pred]))
        elif model_name == LinearSVC:
            model = model_name(C = C,max_iter = max_iterations)
            model.fit(X_train, y_train)
            result,train_AUC,test_AUC= model_evaluate(model,X_train, X_test, y_train, y_test)
            test = pd.Series(user_input)
            pred_test = vectorizer.transform(test)
            predictions = model.predict(pred_test)
            pred_to_label = {0: 'Negative', 1: 'Positive'}

            # Make a list of user_text with sentiment.
            data = []
            for t, pred in zip(user_input, predictions):
                data.append((t, pred, pred_to_label[pred]))
        else:
            model = model_name(n_estimators=n_estim,max_features=max_feat)
            model.fit(X_train, y_train)
            result,train_AUC,test_AUC= model_evaluate(model,X_train, X_test, y_train, y_test)
            test = pd.Series(user_input)
            pred_test = vectorizer.transform(test)
            predictions = model.predict(pred_test)
            pred_to_label = {0: 'Negative', 1: 'Positive'}

            # Make a list of user_text with sentiment.
            data = []
            for t, pred in zip(user_input, predictions):
                data.append((t, pred, pred_to_label[pred]))
        
        return data,result,train_AUC,test_AUC







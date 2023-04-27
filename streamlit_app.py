#import libraries
import time
from wordcloud import WordCloud
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import plotly.graph_objects as go
import backend






# Function to load reviews data
@st.cache_data
def load_reviews():
    """
    Loads the amazon reviews dataset
    """
    return backend.load_dataset()

# Load reviews data
df = backend.load_dataset()

# Create streamlit app header
st.header('Sentiment Analysis App')

# Sidebar menu options
Menu_Options = st.sidebar.selectbox('Required Task',('Data Visualization','Prediction'))

# Data visualization menu options
if Menu_Options == 'Data Visualization':
    st.write("Choose any of the options on the side bar to perform Data Visualization")
    Visualization_options = st.sidebar.selectbox('Data Visualization',('Word Cloud','Label Analysis','Confusion Matrix'))

    # Word cloud option
    if Visualization_options == 'Word Cloud':
        TOTAL = df['verified_reviews'].tolist()
        TOTAL = ' '.join(TOTAL)
        good_reviews = df[df['feedback'] == 1]
        bad_reviews = df[df['feedback'] == 0]
        good = good_reviews['verified_reviews'].tolist()
        bad = bad_reviews['verified_reviews'].tolist()
        GOOD_STR = ' '.join(good)
        BAD_STR = ' '.join(bad)
        emotion = st.sidebar.selectbox('Choose Sentiment',('Postive','Negative','Total'))

        # Positive sentiment word cloud
        if emotion == 'Postive':
            fig,ax = plt.subplots()
            plt.imshow(WordCloud().generate(GOOD_STR),interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

        # Negative sentiment word cloud    
        if emotion == 'Negative':
            fig,ax = plt.subplots()
            plt.imshow(WordCloud().generate(BAD_STR),interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

        # Total word cloud    
        if emotion == 'Total':
            fig,ax = plt.subplots()
            plt.imshow(WordCloud().generate(TOTAL),interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

    # Labels Analysis option       
    if Visualization_options == 'Label Analysis':
        df['variation'] = df['variation'].str.strip()
        tuple_variations = tuple(set(df['variation']))
        type_select = st.selectbox("Pick a model type",tuple_variations)
        type_df = df[df['variation'] == type_select]

        # group by feedback and count the number of occurrences
        feedback_counts = type_df['feedback'].value_counts()

        # create a bar chart using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Negative Reviews'], y=[feedback_counts.get(0, 0)], name='Negative', marker=dict(color='red')))
        fig.add_trace(go.Bar(x=['Positive Reviews'], y=[feedback_counts.get(1, 0)], name='Positive', marker=dict(color='green')))

        # add chart title and axis LABELS
        fig.update_layout(title='Feedback counts for Charcoal Fabric', xaxis_title='Feedback', yaxis_title='Count')

        # display the chart in Streamlit
        st.plotly_chart(fig)

    # Confusion Matrix option  
    if Visualization_options == 'Confusion Matrix':
        visual_models_cm = st.sidebar.selectbox('Prediction using Models',('BernoulliNB','Logistic Regression','GradientBoostingClassifier','LinearSVC'))
        plot_cm  = st.sidebar.button('Plot Confusion Matrix')

        if plot_cm:
            X_train, X_test, y_test, y_train = backend.feature_engineering(df)
            model_list = [BernoulliNB,LogisticRegression,GradientBoostingClassifier,LinearSVC]

            if visual_models_cm == 'Logistic Regression':
                ModelName = model_list[1]
                backend.confusion_matrix_func(ModelName,X_train, X_test, y_test, y_train)
                
            if visual_models_cm == 'GradientBoostingClassifier':
                ModelName = model_list[2]
                backend.confusion_matrix_func(ModelName,X_train, X_test, y_test, y_train)
                
            if visual_models_cm == 'LinearSVC':
                ModelName = model_list[3]
                backend.confusion_matrix_func(ModelName,X_train, X_test, y_test, y_train)

                
            if visual_models_cm == 'BernoulliNB':
                ModelName = model_list[0]
                backend.confusion_matrix_func(ModelName,X_train, X_test, y_test, y_train)
            
        else:
            pass

# After the prediction Button is pressed, what happens next
if Menu_Options == 'Prediction':
    st.write("Analyzing various reviews on an amazon product and predicting the sentiment. You can choose any model and tune the parameters for the best performance.")
    Models_toggle = st.sidebar.selectbox('Prediction using Models',('BernoulliNB','Logistic Regression','GradientBoostingClassifier','LinearSVC', 'Distilbert Pipeline'))
    
    user_input = st.text_input('Enter the review','Enter text')
    user_input = [user_input]
    X_train, X_test, y_train, y_test,vectorizer = backend.reset_feature()

    # Logistic Regression Model
    if Models_toggle == backend.models[1]:
        predict_sentiment  = st.sidebar.button('Predict Sentiment')
        C = st.slider('C',0.0, 10.0, (1.0))
        max_lr = st.slider('Maximum Features',0, 10000, (1000))
        n_jobs =  st.slider('Number of Jobs',-2, 2, (-1))

        if predict_sentiment:
            with st.spinner('Evaluating model performance and sentiment analysis'):
                PROGRESS_TEXT = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=PROGRESS_TEXT)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1, text=PROGRESS_TEXT)

                data,result,train_AUC,test_AUC = backend.predict_model(LogisticRegression,user_input,X_train,X_test,y_train, y_test,vectorizer,C=C,max_lr = max_lr, n_jobs = n_jobs)
                LABELS = 'Sentiment'
                values = data[0][2]
                st.metric(label=LABELS, value=values, delta=None)
                
                st.subheader("Model Performance")
                metric_data = pd.DataFrame(result)
                st.table(metric_data)
                st.write(f"ROC_AUC_Score (Train) {train_AUC}")
                st.write(f"ROC_AUC_Score (Test) {test_AUC}")
            st.success('Loaded')

        else:
            pass

    # Gradient Boosting Classifier Model
    if Models_toggle == backend.models[2]:
        predict_sentiment  = st.sidebar.button('Predict Sentiment')
        n_estim = st.slider('Number of Estimators',1, 100, (100))
        max_feat = st.slider('Maximum Features',1, 100, (1))

        if predict_sentiment:
            data,result,train_AUC,test_AUC = backend.predict_model(GradientBoostingClassifier,user_input,X_train,X_test,y_train, y_test,vectorizer,n_estim=n_estim,max_feat = max_feat)
            LABELS = 'Sentiment'
            values = data[0][2]
            st.metric(label=LABELS, value=values, delta=None)
            
            st.subheader("Model Performance")
            metric_data = pd.DataFrame(result)
            st.table(metric_data)
            st.write(f"ROC_AUC_Score (Train) {train_AUC}")
            st.write(f"ROC_AUC_Score (Test) {test_AUC}")
        else:
            pass

    #LinearSVC    
    if Models_toggle == backend.models[3]:
        predict_sentiment  = st.sidebar.button('Predict Sentiment')
        C = st.slider('C',1, 10, (1))
        max_iterations = st.slider('Maximum Number of Iterations',1000, 20000, (1000))
        if predict_sentiment:
            data,result,train_AUC,test_AUC = backend.predict_model(LinearSVC,user_input,X_train,X_test,y_train, y_test,vectorizer,C = C,max_iterations = max_iterations)
            LABELS = 'Sentiment'
            values = data[0][2]
            st.metric(label=LABELS, value=values, delta=None)
            st.subheader("Model Performance")
            metric_data = pd.DataFrame(result)
            st.table(metric_data)
            st.write(f"ROC_AUC_Score (Train) {train_AUC}")
            st.write(f"ROC_AUC_Score (Test) {test_AUC}")
        else:
            pass

    # BernoulliNB Model
    if Models_toggle == backend.models[0]:
        predict_sentiment  = st.sidebar.button('Predict Sentiment')
        lm = st.slider('Alpha',0, 10, (1))
        if predict_sentiment:
            data,result,train_AUC,test_AUC = backend.predict_model(BernoulliNB,user_input,X_train,X_test,y_train, y_test,vectorizer,alpha=lm)
           
            LABELS = 'Sentiment'
            values = data[0][2]
            st.metric(label=LABELS, value=values, delta=None)
            st.subheader("Model Performance")
            metric_data = pd.DataFrame(result)
            st.table(metric_data)
            st.write(f"ROC_AUC_Score (Train) {train_AUC}")
            st.write(f"ROC_AUC_Score (Test) {test_AUC}")
        else:
            pass

    #Distillbert Model    
    if Models_toggle == backend.models[4]:  
        predict_sentiment  = st.sidebar.button('Predict Sentiment')
        transformer = backend.load_transformer()

        if predict_sentiment:
            data = transformer(user_input)
            LABELS = 'Sentiment'
            values = max([label for label in data[0]], key=lambda x: x['score'])['label']
            st.metric(label=LABELS, value=values, delta=None)



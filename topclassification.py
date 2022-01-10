import time
import stqdm
import streamlit as st
import pandas as pd
import modelling
import plotly.express as px
import numpy as np
from PIL import Image
import seaborn as sns

import transformers

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from bert_sklearn import BertClassifier
from bert_sklearn import load_model
import analysis


def classify():
    st.title('Classify topics using BertClassifier')

    st.write("On this page, you can upload your dataset and the application will classify the assigned"
             " topics for you.")

    st.write("First, please upload your dataset below: ")
    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    global df
    if file is not None:
        try:
            df = pd.read_csv(file, sep=",")
        except Exception as e:
            print(e)
            df = pd.read_excel(file)
        #st.write(df.head(20))  # displays the table of data

        task = st.sidebar.radio(
            'Select the task',
            ['Data Analysis', 'Modelling and Results', 'Top N Topics'], help = 'Please upload a dataset that has Keyword column present'
        )

        if task == 'Data Analysis':
            st.write("Let's conduct some analysis on your dataset!")
            if 'Segment' in df.columns:
                df = df.rename({'Segment': 'Topic'}, axis = 1)
            st.write(df.head(20))
            st.write(analysis.get_analysis_values(df, 'Topic', 'Keyword'))
        if task == 'Modelling and Results':
            epochs = st.sidebar.number_input('Number of Epochs:', min_value=1, max_value=10,step=1)
            st.sidebar.number_input('Test Size:', 0.2)
            st.write("Now, let's perform some modelling to our data values and prepare them "
                     "to be fed into our classifier.")
            if 'Segment' in df.columns:
                df = df.rename({'Segment': 'Topic'}, axis=1)

            label_encoder = LabelEncoder()
            df['Topic'] = label_encoder.fit_transform(df['Topic'])
            global topic
            topic = label_encoder.inverse_transform(df['Topic'])
            st.write('Below, you can see the transformed data. It has transformed our topic values from string to an integer making it easier for our classifier to understand them.')
            st.write(df.head())

            seed = 42

            # Train-Test Split
            X = (np.array(df['Keyword']))
            y = (np.array(df['Topic']))


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            st.write(f'With our test_size as `{0.2}`. The sizes of our test and train become:')
            st.write(f'**Train Dataset Shape**: {X_train.shape}'
                     f'** Test Dataset Shape**: {X_test.shape}')


            split_train = pd.DataFrame({'X_train': X_train, 'y_train': y_train})
            split_test = pd.DataFrame({'X_test': X_test, 'y_test': y_test})
            split_test['y_test'] = label_encoder.inverse_transform(y_test)
            split_train['y_train'] = label_encoder.inverse_transform(y_train)
            # split_test.head(10)

            st.write('### Distribution of Topics after Splitting')
            testfig = px.histogram(split_test, x='y_test', title="Distribution of Topics in Test Set")
            st.plotly_chart(testfig)

            trainfig = px.histogram(split_train,x= 'y_train', title = "Distribution of Topics in Train Set")
            st.plotly_chart(trainfig)


            if st.button('Hey, click me to train your keywords.'):
                st.info('Initiating Process. Sit back and grab a coffee, while I learn topics from your keywords.')
                modelling._update_progress("Fitting the model", 100)
                model = BertClassifier()
                model.epochs = epochs
                model.validation_fraction = 0.05
                model.learning_rate = 2e-5
                model.max_seq_length = 128

                # Fit the model
                history = model.fit(X_train, y_train)

                modelling._update_progress("Generating Predictions", 100)
                # Make Predictions
                y_pred = model.predict(X_test)
                # Predictions on the train set
                y_pred_train = model.predict(X_train)

                modelling._update_progress("Generating Results", 100)
                st.success('Done! You can read the results below.')

                st.write(f'Mean Squared Error {round(mean_squared_error(y_test, y_pred),2)}')
                st.write(f'Train Set Accuracy Score: {round(accuracy_score(y_train, y_pred_train),2)}')
                st.write(f'Test Set Accuracy Score {round(accuracy_score(y_pred, y_test),2)}')

                st.write(modelling.plot_confusion_matrix(model, y_test, y_pred, 15, 10, "Confusion Matrix of Test Set", np.unique(topic)))
                st.write(modelling.plot_confusion_matrix(model, y_train, y_pred_train, 15, 10, "Confusion Matrix of Train Set", np.unique(topic)))

                # Inverse Labelling of the test labels
                y_pred = label_encoder.inverse_transform(y_pred)
                y_test = label_encoder.inverse_transform(y_test)

                # Inverse Labelling of the train labels
                y_pred_train = label_encoder.inverse_transform(y_pred_train)
                y_train = label_encoder.inverse_transform(y_train)

                testdf = pd.DataFrame({'Keyword': X_test, 'predicted_topics': y_pred, 'Topic': y_test})
                traindf = pd.DataFrame({'Keyword': X_train, 'predicted_topics': y_pred_train, 'Topic': y_train})

                # Concatenating test and train dfs along rows
                result_df = pd.concat([traindf, testdf], axis=0)
                st.write('**Your final Resultant Dataset**')
                st.write(result_df.head(20))
                csv = modelling.convert_df(result_df)

                st.download_button(label='Download your results as CSV',
                                   data = csv,
                                   file_name = 'result_df.csv',
                                   mime = 'text/csv',
                                   )
                st.write('Analysing the topics that were differently classified. You can use this dataframe to get the top N Prediction'
                         ' in the next task. ')
                # Analysing the results
                result_df['condition'] = (result_df['predicted_topics'] != result_df['Topic'])
                result_df_cond = result_df[result_df.condition]
                st.write(result_df_cond.head(20))
                csv = modelling.convert_df(result_df_cond)

                st.download_button(label='Download your results as CSV for top N predictions',
                                   data=csv,
                                   file_name='result_df_condition.csv',
                                   mime='text/csv',
                                   )
        elif task == 'Top N Topics':
            epochs = st.sidebar.number_input('Number of Epochs:', min_value=1, max_value=10, step=1)
            st.sidebar.number_input('Test Size:', 0.2)
            st.write('On this page, you can get the top two topics that were predicted by the model for all of the keywords that were misclassified.'
                         ' You can also get top n topics for all the keywords.')
            if st.button('Start Predicting'):
                if 'Segment' in df.columns:
                    df = df.rename({'Segment': 'Topic'}, axis=1)
                seed = 42

                # Train-Test Split
                X = (np.array(df['Keyword']))
                y = (np.array(df['Topic']))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                st.write(f'With our test_size as `{0.2}`. The sizes of our test and train become:')
                st.write(f'**Train Dataset Shape**: {X_train.shape}'
                         f'** Test Dataset Shape**: {X_test.shape}')

                st.info('Initiating Process. Sit back, grab a coffee! Instantiating and producing results.')
                modelling._update_progress("Fitting the model", 100)
                model = BertClassifier()
                model.epochs = epochs
                model.validation_fraction = 0.05
                model.learning_rate = 2e-5
                model.max_seq_length = 128

                # Fit the model
                history = model.fit(X_train, y_train)

                topn_topics = modelling.return_top_n_pred_prob_df(2, model, df['Keyword'], 'topics')
                st.success('Success! You can view your results in the below dataset.')
                # Forming the column topic as a list to inverse transform
                topics_0 = topn_topics['topics_0'].tolist()
                label_encoder = LabelEncoder()
                df['Topic'] = label_encoder.fit_transform(df['Topic'])
                topics_0 = label_encoder.inverse_transform(topics_0)

                topics_1 = topn_topics['topics_1'].tolist()
                topics_1 = label_encoder.inverse_transform(topics_1)

                # Assigning the new converted topic names to the column
                topn_topics['topics_0'] = topics_0
                topn_topics['topics_1'] = topics_1

                # merge it with the original data to get languages
                topn_topics = pd.merge(topn_topics, df, left_index=True, right_index=True)
                topn_topics = topn_topics[['keywords', 'topics_0', 'topics_0_prob', 'topics_1', 'topics_1_prob']]

                st.write(topn_topics.head(20))
                csv = modelling.convert_df(topn_topics)

                st.download_button(label='Download your results as CSV for top N topics',
                                   data=csv,
                                   file_name='topntopics.csv',
                                   mime='text/csv',
                                   )
                

                





















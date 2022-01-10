import streamlit as st
import pandas as pd
from PIL import Image
import analysis
import modelling
import plotly.express as px
from bertopic import BERTopic
from ast import literal_eval
import re

def app():
    st.title('Discovering new topics using BERTopic()')
    st.write("On this page, you can upload your dataset and the application will discover potential"
             " new topics for you.")
    st.write("First, please upload your dataset below: ")
    file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    global df
    if file is not None:
        try:
            df = pd.read_csv(file, sep=",")
            st.success('Dataframe successfully uploaded!')
        except Exception as e:
            print(e)
            df = pd.read_excel(file)
            st.success('Dataframe successfully uploaded!')

        task_chosen = st.sidebar.radio(
            'Select the task',
            ['Data Analysis', 'Modelling and Results', 'Similar Topics'], help = 'Please upload phrases or keywords that have a maximum count of 100 for better results.'
    )
        if task_chosen == 'Data Analysis':
            df = df.iloc[:, :5]
            df = df.rename_axis('Index').reset_index()
            st.write(df.head(10))
            global column
            column = st.sidebar.text_input(label='Choose the column name to perform analysis on:')

            if 'col1'  in column:
                df['col1'] = df.topic_selection.apply(lambda x: literal_eval(str(x)))
                df = df['col1']
                # df2.loc[0] = np.array(['français'])
                df = df.explode('col1')
                df = pd.DataFrame(df)

            if 'col2' in column:
                df['col2'] = df.topics.apply(lambda x: literal_eval(str(x)))
                df = df['col2']
                # df2.loc[0] = np.array(['français'])
                df = df.explode('col2')
                df = pd.DataFrame(df)



            if st.button('Start analyzing keyphrases!'):
                st.write(analysis.b2b_analysis(df, column))
        if task_chosen == 'Modelling and Results':

             if 'col1'  in column:
                df['col1'] = df.topic_selection.apply(lambda x: literal_eval(str(x)))
                df = df['col1']
                # df2.loc[0] = np.array(['français'])
                df = df.explode('col1')
                df = pd.DataFrame(df)

            if 'col2' in column:
                df['col2'] = df.topics.apply(lambda x: literal_eval(str(x)))
                df = df['col2']
                # df2.loc[0] = np.array(['français'])
                df = df.explode('col2')
                df = pd.DataFrame(df)

            df = df.rename_axis('Index').reset_index()
            st.write(df.head(20))
            docs = list(df.loc[:, column].values)

            st.write(f'The first five keywords out of your dataset for discovering topics are for column `{column}`')
            st.write(docs[:5])
            st.write(f'The total number of keywords which will be trained are `{len(docs)}`')

            if st.button('Hey, click me to discover topics from your keywords.'):
                st.info('Initiating Process. Sit back and grab a coffee, while I discover potential topic names from your keywords.')
                global model
                model = BERTopic(embedding_model='sentence-transformers/LaBSE', language="multilingual",
                                   calculate_probabilities=True, verbose=True)
                topics, probs = model.fit_transform(docs)

                st.success("Success! I have discovered potential new topics for you.")

                input_topics_freq_0 = model.get_topic_info()

                st.write('You can view some information on your topics by the visualizations formed below.')
                topicsmade = px.bar(input_topics_freq_0, x='Topic', y='Count', title='Distribution of Input Topic Generated')
                st.plotly_chart(topicsmade)


                wordscore = model.visualize_barchart(topics=[0,1])
                st.plotly_chart(wordscore)

                intertopic_distance_map = model.visualize_topics()
                st.plotly_chart(intertopic_distance_map)

                with st.expander(label='Not all topics being displayed? Find out why.'):
                    st.write("""
                    The technique used to visualise the clusters of the topics generated is by using UMAP which reduces the dimensions of' \
                    'the embeddings and for making the visual pleasant for the user, only the most embeddings that share a similar x and y ' \
                    'range are displayed. Its totally normal to not display all topics. For better understanding of the topics generated, you' \
                    'can download the result dataframe formed below. 
                    """)

                most_similar_dict = dict(zip(df['Index'], df[column]))
                grouped_topics = modelling.get_topic_val(model, topics)
                res_df = modelling.make_result_df(grouped_topics, most_similar_dict)
                global result_df
                result_df = modelling.make_final_dataframe(model, res_df)
                result_df = result_df.astype(str)
                st.write('Your resultant dataset: ')
                st.dataframe(result_df)

                csv = modelling.convert_df(result_df)

                st.download_button(label='Download your results as CSV for new discovered topics',
                                   data=csv,
                                   file_name='topicsdiscovered.csv',
                                   mime='text/csv',
                                   )


        elif task_chosen == 'Similar Topics':
            st.write('On this page, you can find topics that are similar enough to be grouped together or be used as sub-topics.')

            c_df = modelling.get_class_similarity_score(model)
            st.write('The top 10 topics that may be grouped together are:')
            similar_df = c_df.sort_values(['c_similarity_score'], ascending = [False]).head(10)
            st.dataframe(similar_df)

            topicnum = st.sidebar.number_input('Topic Number:', min_value=-1, max_value=400)
            st.write('To view the top 10 keywords in for each topic, you can use enter the topic number on the left.')
            st.write(f'Top keywords in topic **{topicnum}** are:'
                     f'{model.get_topic(topicnum)}')
            csv = modelling.convert_df(similar_df)

            st.download_button(label='Download your results as CSV for similar topics',
                               data=csv,
                               file_name='similartopics.csv',
                               mime='text/csv',
                               )





















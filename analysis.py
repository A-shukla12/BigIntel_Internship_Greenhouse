import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

buffer = io.StringIO()


def get_analysis_values(dataframe, columnname,column):
    st.write(f'Shape of your dataframe is `{dataframe.shape}`')
    #st.write(dataframe.describe())
    #st.write(dataframe[columnname].value_counts())
    fig = px.histogram(dataframe, x=columnname, title = "Distribution of Topics")
    st.plotly_chart(fig)
    st.write('Below, you can see the top keywords in the entire dataset.')
    wordcloud2 = WordCloud().generate(' '.join(dataframe[column]))
    wordfig = plt.figure(figsize=(10, 8), facecolor=None)
    plt.imshow(wordcloud2)
    plt.axis("off")
    st.pyplot(wordfig)

def b2b_analysis(dataframe, column):
    st.write(f'Shape of your dataframe is `{dataframe.shape}`')
    # buffer = io.StringIO()
    #dataframe.info(buf = buffer)
    #s = buffer.getvalue()
    #with open('df_describe.txt', 'w',
      #  encoding = 'utf-8') as f:
     #   f.write(s)
    #st.write(s) # todo: think about making this work
    # st.write(dataframe.describe())
    st.write(f'Below, you can see the top keyphrases in the specified column: {column}.')
    wordcloud2 = WordCloud().generate(' '.join(dataframe[column]))
    cloudfig = plt.figure(figsize=(10, 8), facecolor=None)
    plt.imshow(wordcloud2)
    plt.axis("off")
    st.pyplot(cloudfig)


def minimum_label_count(dataframe, columnname, min_counts):
    label_count = dataframe[columnname].value_counts()
    filtered_topics = label_count[label_count <= min_counts].index
    topics_not_in_filtered_topics = label_count[label_count > min_counts].index
    if len(topics_not_in_filtered_topics) > 0:
        print(
            f'The following topics do not meet the observations threshold {min_counts} and will be removed {list(filtered_topics)}')
        df = dataframe[~dataframe[columnname].isin(filtered_topics).values]
        if not list(filtered_topics):
            print('Enough observations for classification :)')

    print(f'New Shape of the Dataframe {df.shape}')
    fig = px.histogram(df, x=columnname, title="Distribution of Topics After Minimum Values")
    fig.update_traces(marker_color='mediumpurple')
    fig.show(renderer='colab')

    return df


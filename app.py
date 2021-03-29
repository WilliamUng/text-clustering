import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import tensorflow_hub as hub
import umap.umap_ as umap
import hdbscan

st.set_page_config(page_title='Altair Gallery', layout='wide', initial_sidebar_state = 'auto')
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>', unsafe_allow_html=True)

st.sidebar.title('Text Clustering')

SEED = 42

@st.cache(allow_output_mutation=True)
def get_USE_embedding_model():
    module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'
    USE_embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')

    return USE_embed

@st.cache(allow_output_mutation=True)
def compute_sentence_embeddings(text):
    embeddings = USE_embed(text)['outputs'].numpy()
    return embeddings

USE_embed = get_USE_embedding_model()

df_csv = pd.read_csv('train.csv')

toy_data = [
    ["My flight got cancelled and I didn't get a refund.", "travel"],
    ["The pilot said the delays were due to ice on the wings.", "travel"],
    ["The airport was closed due to a terrorist threat.", "travel"],
    ["The plane coudn't take off, so the airline booked us into the Marriott nearby.", "travel"],
    ["Room service was friendly and professional, I will definitely be back!", "hotel"],
    ["Hotel was having a huge function and I had no room to enjoy the facilities.", "hotel"],
    ["I was charged 10$ for a water in the mini fridge, ridiculous!!!", "hotel"],
    ["The soccer and basketball events were badly organised.", "activities"],
    ["I wish that they would offer surfing in the itinerary, the weather was perfect for it.", "activities"],
    ["I swim at 8 AM every day to train for the competition", "activities"],
    ["Lets get a together an plan a giant ski trip in France", "activities"],
    ["Today is gonna be the day that we're gonna throw it back to you.", "other"],
    ["I wish the duty free stores had more liquor options", "travel"],
    ["There was no more room at the gate, so I was forced to stand up for 30 minutes", "travel"],
    ["The airport security held me up for a petty reason and wasted my time","travel"],
    ["I had a great experience at the Aspire Lounge, I really enjoyed the food", "travel"],
    ["I was once again unable to enter the lounge, I was turned down due to capacity","travel"],
    ["Flights prices during the holiday are way to high, this is outrageous.", "travel"],
    ["Prices on this website seem to change every 10 seconds, I don't like it.", "travel"],
    ["I had a hard time finding the lounge, I am thankful for the Priority Pass navigation system", "travel"],
    ["This hotel was very full over the weekend, I wish they had more space", "hotel"],
    ["I was able to check into my room in 5 minutes, super easy", "hotel"],
    ["This hotel was full of corporate types, they ruined my holiday.", "hotel"],
    ["I loved my UCPA ski holiday, the food was great, and I learned lots of snowboarding tricks", "activities"],
    ["Next time I go to the beach, I will definitely try to surf", "activities"]
]

#df = pd.DataFrame(toy_data, columns = ['review','category'])
df = pd.DataFrame(list(zip(df_csv['tweet'])), columns=['review'])
train_embeddings = compute_sentence_embeddings(df.review[:500])

col_umap, col_hdbscan = st.beta_columns(2)
st.title('')
col_umap1, col_hdbscan1 = st.beta_columns(2)

##################### UMAP #####################

col_umap.subheader('UMAP Dimensionality Reduction')

umap_expander = col_umap.beta_expander('Parameters')
n_neighbours = umap_expander.slider('n_neighbours', 2, 100, 15)

clusterable_embedding = umap.UMAP(
    n_neighbors=n_neighbours,
    min_dist=0.0,
    n_components=2,
    random_state=10,
).fit_transform(train_embeddings)

umap = pd.DataFrame(list(zip(df.review, clusterable_embedding[:, 0], clusterable_embedding[:, 1])), columns=['review', 'x', 'y'])

c = alt.Chart(umap).mark_point(size=60).encode(
    x='x',
    y='y',
    #color='Origin',
    tooltip=['x', 'y']
).interactive().properties(
    height=400
)

col_umap1.altair_chart(c, use_container_width=True)

##################### HDBSCAN #####################

col_hdbscan.subheader('HDBSCAN Clustering')

hdbscan_expander = col_hdbscan.beta_expander('Parameters')
min_samples_slider = hdbscan_expander.slider('min_samples', 1, 100, 3)
min_cluster_slider = hdbscan_expander.slider('min_cluster_size', 2, 100, 4)

labels = hdbscan.HDBSCAN(
    min_samples=min_samples_slider,
    min_cluster_size=min_cluster_slider,
).fit_predict(clusterable_embedding)

umap['labels'] = labels

d = alt.Chart(umap).mark_point(size=60).encode(
    x='x',
    y='y',
    color=alt.Color('labels:N', legend=alt.Legend(title="Clusters by color"), scale=alt.Scale(scheme='tableau10')),
    tooltip=['review', 'labels']
).interactive().properties(
    height=400
)

col_hdbscan1.altair_chart(d, use_container_width=True)
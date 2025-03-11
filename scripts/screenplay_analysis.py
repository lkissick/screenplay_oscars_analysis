import pandas as pd
import os
from textblob import TextBlob
from wordcloud import WordCloud
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import base64
from io import BytesIO
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

nlp = spacy.load("en_core_web_sm")

# Read CSVs
awards_metadata = pd.read_csv("./movie_scripts_corpus/movie_metadata/screenplay_awards.csv")
metadata = pd.read_csv("./movie_scripts_corpus/movie_metadata/movie_meta_data.csv")

# Get IDs for screenplays in screenplay_awards.csv
id_original_lst = []
id_adapted_lst = []
id_no_oscar_lst = []
for awards_index, awards_row in awards_metadata.iterrows():
    id = awards_row['movie'].split('_')[1]
    if awards_row['Academy Awards original screenplay'] == '+':
        for md_index, md_row in metadata.iterrows():
            if str(md_row['imdbid']) in str(id):
                id_original_lst.append(id)
    elif awards_row['Academy Awards adapted screenplay'] == '+':
        for md_index, md_row in metadata.iterrows():
            if str(md_row['imdbid']) in str(id):
                id_adapted_lst.append(id)
    else:
        for md_index, md_row in metadata.iterrows():
            if str(md_row['imdbid']) in str(id):
                id_no_oscar_lst.append(id)

# Read screenplay function based on ID number
def read_screenplay(id_number):
    for file in os.scandir('./preprocessed_screenplays'):
        if id_number in file.name:
            with open(file.path, "r", encoding="utf8") as outfile:
                content = outfile.read()
            return content

# Sentiment analysis function
def get_sentiment(screenplay):
    blob = TextBlob(screenplay)
    return blob.sentiment.polarity

# Create dictionary with keys as titles and values as a dictionary containing year, sentiment, and award
screenplay_dct = {}
for id in id_original_lst:
    for index, row in metadata.iterrows():
        if str(row['imdbid']) in str(id):
            screenplay_dct[row['title']] = {'year': row['year'], 'sentiment': get_sentiment(read_screenplay(id)), 'award': 'Best Original Screenplay'}
for id in id_adapted_lst:
    for index, row in metadata.iterrows():
        if str(row['imdbid']) in str(id):
            screenplay_dct[row['title']] = {'year': row['year'], 'sentiment': get_sentiment(read_screenplay(id)), 'award': 'Best Adapted Screenplay'}
for id in id_no_oscar_lst:
    for index, row in metadata.iterrows():
        if str(row['imdbid']) in str(id):
            screenplay_dct[row['title']] = {'year': row['year'], 'sentiment': get_sentiment(read_screenplay(id)), 'award': 'No Oscar'}

# Creating data frame
df = pd.DataFrame.from_dict(screenplay_dct, orient='index')

# Filter out films with 'No Oscar' in the award column
df_filtered = df[df['award'] != 'No Oscar']

# Remove films with year = -1
df_filtered = df_filtered[df_filtered['year'] != -1]

# Remove specific films by title
films_to_remove = ['The Lost Weekend']
df_filtered = df_filtered[~df_filtered.index.isin(films_to_remove)]

# Create a lookup dictionary for title to ID
title_to_id = {row['title']: str(row['imdbid']) for index, row in metadata.iterrows()}

# Combine all screenplays into a list of documents
def get_documents_for_category(ids):
    documents = []
    for id in ids:
        content = read_screenplay(id)
        if content:
            documents.append(content)
    return documents

# Get documents for Oscar-winning and non-Oscar-winning films
oscar_documents = get_documents_for_category(id_original_lst + id_adapted_lst)
non_oscar_documents = get_documents_for_category(id_no_oscar_lst)

# Combine all documents for topic modeling
all_documents = oscar_documents + non_oscar_documents

# Preprocess the documents using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features=1000)
tfidf_matrix = vectorizer.fit_transform(all_documents)

# Apply NMF
num_topics = 5  # Number of topics to extract
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

# Get the top words for each topic
def get_top_words_for_topics(model, feature_names, n_top_words=10):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return top_words

feature_names = vectorizer.get_feature_names_out()
top_words = get_top_words_for_topics(nmf_model, feature_names)

# Assign topics to documents
document_topics = nmf_matrix.argmax(axis=1)

# Add topic assignments to the documents
oscar_topics = document_topics[:len(oscar_documents)]
non_oscar_topics = document_topics[len(oscar_documents):]

# Calculate proportional topic distribution
total_oscar_films = len(oscar_documents)
total_non_oscar_films = len(non_oscar_documents)

topic_distribution = pd.DataFrame({
    'Topic': list(range(num_topics)),
    'Oscar-Winning': [sum(oscar_topics == i) / total_oscar_films * 100 for i in range(num_topics)],
    'Non-Oscar-Winning': [sum(non_oscar_topics == i) / total_non_oscar_films * 100 for i in range(num_topics)]
})

# Create Dash app
app = dash.Dash(__name__)

# Define Oscars theme colors
oscars_colors = {
    'background': '#1a1a1a',  # Dark grey
    'text': '#ffffff',        # White
    'gold': '#ffd700',        # Gold
    'yellow': '#ffff00',      # Yellow
    'silver': '#c0c0c0',      # Silver
    'grey': '#808080'         # Grey
}

# Add this to the end of your app.layout
app.layout = html.Div(style={'backgroundColor': oscars_colors['background'], 'padding': '20px'}, children=[
    # Dashboard Title
    html.H1(
        "Textual Analysis of Oscar-Winning Films 🏆",
        style={
            'textAlign': 'center',
            'color': oscars_colors['gold'],
            'fontFamily': 'Georgia, serif',
            'marginBottom': '20px'
        }
    ),

    html.Div([
        # Left column: Scatterplot
        html.Div([
            dcc.Graph(
                id='sentiment-scatterplot',
                figure=px.scatter(
                    df_filtered.reset_index(),
                    x='year',
                    y='sentiment',
                    hover_data=['index'],
                    color='award',
                    title='Sentiment of Oscar-Winning Screenplays by Year of Release',
                    color_discrete_map={
                        'Best Original Screenplay': oscars_colors['gold'],
                        'Best Adapted Screenplay': oscars_colors['silver']
                    },
                    labels={'year': 'Year of Release', 'sentiment': 'Sentiment Polarity'}
                ).update_layout(
                    plot_bgcolor=oscars_colors['background'],
                    paper_bgcolor=oscars_colors['background'],
                    font={'color': oscars_colors['text']},
                    title_font={'size': 18, 'color': oscars_colors['gold']}
                )
            )
        ], style={'width': '60%', 'display': 'inline-block'}),

        # Right column: Word Cloud
        html.Div([
            html.H3(
                id='word-cloud-title',
                children='Word Cloud',
                style={
                    'textAlign': 'center',
                    'color': oscars_colors['gold'],
                    'fontFamily': 'Arial, sans-serif'
                }
            ),
            html.Img(id='word-cloud', src='', style={'width': '100%', 'height': 'auto'})
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    # Third visualization: Topic Distribution with Hover
    html.Div([
        dcc.Graph(
            id='topic-distribution-bar-chart',
            figure=px.bar(
                topic_distribution.melt(id_vars='Topic', var_name='Category', value_name='Percentage'),
                x='Topic',
                y='Percentage',
                color='Category',
                barmode='group',
                title='Topic Distribution: Oscar-Winning vs. Non-Oscar-Winning Films',
                labels={'Percentage': 'Percentage of Films (%)', 'Topic': 'Topic'},
                color_discrete_map={
                    'Oscar-Winning': oscars_colors['gold'],
                    'Non-Oscar-Winning': oscars_colors['silver']
                }
            ).update_layout(
                plot_bgcolor=oscars_colors['background'],
                paper_bgcolor=oscars_colors['background'],
                font={'color': oscars_colors['text']},
                title_font={'size': 18, 'color': oscars_colors['gold']}
            )
        )
    ], style={'margin-top': '20px'}),

    # Source text at the bottom
    html.Div(
        "Source: https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus/data",
        style={
            'textAlign': 'center',
            'color': oscars_colors['text'],
            'fontFamily': 'Arial, sans-serif',
            'marginTop': '20px',
            'fontSize': '14px'
        }
    )
])

# Callback to update word cloud title
@app.callback(
    Output('word-cloud-title', 'children'),
    Input('sentiment-scatterplot', 'hoverData')
)
def update_word_cloud_title(hover_data):
    if hover_data is None:
        return 'Word Cloud'
    title = hover_data['points'][0]['customdata'][0]
    return f'Word Cloud: {title} 🎬'

# Callback to generate word cloud on hover
@app.callback(
    Output('word-cloud', 'src'),
    Input('sentiment-scatterplot', 'hoverData')
)
def update_word_cloud(hover_data):
    if hover_data is None:
        return ''
    title = hover_data['points'][0]['customdata'][0]
    if title not in title_to_id:
        print(f"Title '{title}' not found in metadata.")
        return ''
    id = title_to_id[title]
    content = read_screenplay(id)
    if content is None:
        print(f"No screenplay content found for ID {id}.")
        return ''
    wordcloud = WordCloud(width=800, height=400, background_color=oscars_colors['background']).generate(content)
    img_buffer = BytesIO()
    wordcloud.to_image().save(img_buffer, format='PNG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

# Callback to show top words on hover
@app.callback(
    Output('topic-distribution-bar-chart', 'figure'),
    Input('topic-distribution-bar-chart', 'hoverData')
)
def update_topic_hover(hover_data):
    if hover_data is None:
        # Return the original figure if no data is hovered
        return px.bar(
            topic_distribution.melt(id_vars='Topic', var_name='Category', value_name='Percentage'),
            x='Topic',
            y='Percentage',
            color='Category',
            barmode='group',
            title='Topic Distribution: Oscar-Winning vs. Non-Oscar-Winning Films 🏆',
            labels={'Percentage': 'Percentage of Films (%)', 'Topic': 'Topic'},
            color_discrete_map={
                'Oscar-Winning': oscars_colors['gold'],
                'Non-Oscar-Winning': oscars_colors['silver']
            }
        ).update_layout(
            plot_bgcolor=oscars_colors['background'],
            paper_bgcolor=oscars_colors['background'],
            font={'color': oscars_colors['text']},
            title_font={'size': 18, 'color': oscars_colors['gold']}
        )

    # Get the hovered topic
    topic_idx = hover_data['points'][0]['x']
    top_words_str = ', '.join(top_words[topic_idx])

    # Update the figure title to show the top words
    updated_figure = px.bar(
        topic_distribution.melt(id_vars='Topic', var_name='Category', value_name='Percentage'),
        x='Topic',
        y='Percentage',
        color='Category',
        barmode='group',
        title=f'Topic {topic_idx + 1}: {top_words_str}',
        labels={'Percentage': 'Percentage of Films (%)', 'Topic': 'Topic'},
        color_discrete_map={
            'Oscar-Winning': oscars_colors['gold'],
            'Non-Oscar-Winning': oscars_colors['silver']
        }
    ).update_layout(
        plot_bgcolor=oscars_colors['background'],
        paper_bgcolor=oscars_colors['background'],
        font={'color': oscars_colors['text']},
        title_font={'size': 18, 'color': oscars_colors['gold']}
    )
    return updated_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
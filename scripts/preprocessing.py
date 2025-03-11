import pandas as pd
import os
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Read CSVs
awards_metadata = pd.read_csv("./movie_scripts_corpus/movie_metadata/screenplay_awards.csv")
metadata = pd.read_csv("./movie_scripts_corpus/movie_metadata/movie_meta_data.csv")

# Extract IDs from awards metadata
id_lst_awards = []
for awards_index, awards_row in awards_metadata.iterrows():
    id = awards_row['movie'].split('_')[1]
    id_lst_awards.append(id)

# Define a custom list of words to remove
custom_stopwords = {"look", "int", "ext", "script", "continued", "revise", "final", "gon", "dorothy"}

def read_screenplay(id_number):
    for file in os.scandir('./movie_scripts_corpus/screenplay_data/data/raw_text_lemmas/raw_text_lemmas'):
        if id_number in file.name:
            with open(file.path, "r", encoding="utf8") as outfile:
                content = outfile.read()
            doc = nlp(content)
            filtered_words = [
                token.text.lower() for token in doc
                if not token.is_stop  # Exclude stopwords
                and not token.ent_type_  # Exclude named entities (e.g., PERSON, GPE)
                and token.pos_ in {"NOUN", "VERB", "ADJ"}  # Include only nouns, verbs, and adjectives
                and token.text.lower() not in custom_stopwords  # Exclude custom stopwords
            ]
            filtered_words = ' '.join(filtered_words)
            return filtered_words, file.name

# Process and save screenplays
count = 1
for id in id_lst_awards:
    processed_text, file_name = read_screenplay(id)
    if ' s ' in file_name:
        file_name = file_name.replace(' s', "'s")
    with open(f'./preprocessed_screenplays/{file_name}', 'w', encoding='utf8') as outfile:
        outfile.write(processed_text)
        print(f'{count}/462')
        count += 1
# Oscar Screenplay Analysis
This project analyzes the sentiments of oscar-winning films for both best adapted and original screenplay, their most common words, and topics compared between oscar-winning films and non-oscar-winning films.

## Installation
1. Clone the repository:
    git clone https://github.com/your-username/oscar-screenplay-analysis.git
2. Navigate to the project directory:
    cd oscar-screenplay-analysis
3. Install the required Python libraries:
    pip install -r requirements.txt

## Data Source
Download and place the data from the following link into the data/ directory:
     https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus/data

## Running the Scripts
You must run the scripts in the following order:
1. Run the preprocessing scripts to process the necessary scripts and place them into a new directory:
    python scripts/preprocessing.py
2. Run the screenplay analysis script to analysis the data and create your dash app:
    python scripts/screenplay_analysis.py
3. Open the dash app in your browser:
    http://127.0.0.1:8050/

## Dependencies
1. Python 3.x
2. Libraries listed in requirements.txt

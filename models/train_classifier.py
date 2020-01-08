import sys
import sklearn
print(sklearn.__version__)
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt','wordnet','stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """Load the cleaned data from DB filepath"""
    db_file = 'sqlite:///' + database_filepath
    engine = create_engine(db_file)
    df = pd.read_sql_table('Disasters', con=engine)  
    print(df.head())
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    INPUT-text
    OUTPUT-Tokenize and transform input text.Return tranformed tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Take out all punctuations and english stop words while tokenizing
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    new_tokens =[]
    for w in tokens:
        if w not in stop_words: 
            new_tokens.append(w)
   # Lemmatize the txt tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in new_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Build and return final model with pipeline and Classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate final model and display the metrics
    INPUT
    model -- estimator-object
    X_test -- Dataframe to use for predictions
    y_test -- Dataframe with actual values
    category_names-- List of target class names
    OUTPUT
    Metrics-Precision,F1-Score, Recall and Support
    """
    y_pred = model.predict(X_test)
    metrics_df = pd.DataFrame(columns=['Target Name', 'F1 Score', 'Precision', 'Recall'])
    i = 0
    for cat in category_names:
        precision, recall, f1_score, support = precision_recall_fscore_support(Y_test[cat], y_pred[:,i], average='weighted')
        metrics_df.set_value(i+1, 'Target Name', cat)
        metrics_df.set_value(i+1, 'F1 Score', f1_score)
        metrics_df.set_value(i+1, 'Precision', precision)
        metrics_df.set_value(i+1, 'Recall', recall)
        i=i+1
    print('Overall avg F1-Score:', metrics_df['F1 Score'].mean())
    print('Overall avg Precision:', metrics_df['Precision'].mean())
    print('Overall avg Recall:', metrics_df['Recall'].mean())


def save_model(model, model_filepath):
    """Pickle  the model to a file"""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
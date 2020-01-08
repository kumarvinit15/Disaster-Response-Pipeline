import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Load data and create dataframe from filepaths
    INPUT:
    messages_filepath -- string or link to data file
    categories_filepath -- string or ink to data file
    OUTPUT
    df - pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    messages.head()
    categories = pd.read_csv(categories_filepath)
    categories.head()
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Process and clean data in the dataframe (df) and transform 'categories' part
    INPUT
    df -- pandas DataFrame
    OUTPUT
    df -- cleaned pandas DataFrame
    """
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.loc[0]
    category_colnames = []
    for txt in row:
        category_colnames.append(txt[:-2])
    print('Column names:', category_colnames)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    # check number of duplicates
    df.duplicated().sum()
    print('No. of duplicate rows:', df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2]
    print('No. of duplicate rows:', df.duplicated().sum())
    return df


def save_data(df, database_filename):
    """
    Saves DataFrame (df) to database path
    INPUT
    df-pandas dataframe
    database_filename-Name of the database file
    """
    db_file = 'sqlite:///' + database_filename
    engine = create_engine(db_file)
    df.to_sql('Disasters', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
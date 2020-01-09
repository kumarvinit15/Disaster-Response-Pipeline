# Disaster Response Pipeline
## Table of Contents
  1. Project Description   
  2. Python Libraries   
  3. File Description  
  4. Implementation  
  5. Running Scripts  
  6. Results  
  7. Visuals from Web page 
  8. Licensing, Authors, and Acknowledgements  
  
## Project Description
The objective of the Disaster Response Pipeline project is to create a web application that can help emergency workers to classify incoming messages using machine learning into specific categories to speedup aid and help to needed people.
The dataset for this project is provided by Figure Eight which consists of about 27,000 messages with 36 categories.
## Python Libraries
Following python libraries are used for this project:  
`pandas,numpy,sqlalchemy,matplotlib,plotly,re,NLTK,NLTK [punkt, wordnet, stopwords],sklearn,pickle,joblib,json,flask`
## File Description
Project files has been organized into data, models and app folders as described below:  
1. data:  
    * `DisasterResponse.db`: SQLite database containing messages and categories data  
    * `disaster_categories.csv`: dataset including all the categories  
    * `disaster_messages.csv`: dataset including all the messages  
    * `process_data.py`: ETL pipeline scripts to read, clean, and save data into a database  
2. models:  
    * `train_classifier.py`: machine learning pipeline to train and pickle a machine learning classifier  
    * `classifier.pkl`: output of the machine learning classifier  
3. app:  
    * `run.py`: Flask file to run the web application  
    * `templates`: contains html files for the web application  
## Implementation
1. ETL Pipeline
    * Merge two dataset (`messages` and `caregories`) csv files
    * Modify the `categories.csv` file and split each category into a separate column
    * Remove duplicates rows from combined dataset
    * Generate a SQLlite database `DisasterResponse.db` with transformed dataset  
2. Machine Learning Pipeline
   - Text Preprocessing  
      * Replace any URLs with string
      * Tokenize text
      * Remove special characters
      * lemmatize and convert text to lower case
      * remove stop words
   - Build ML Pipeline
      * Create ML Pipeline with `countevectorizer` and `tfidtransformer`
      * Use multioutput classifier with `Random Forest` Model
      * Split the dataset into train and test data
      * Train and test the dataset with ML Pipeline
      * Generate predictions with ML pipeline and display `F1 Score`,`precison` and `recall` metrics
3. Improve ML Model
    * Preform `GirdSearchCV`
    * Find best parameters and update the pipeline with new parameters
4. Export ML Model as .pkl File
## Running Python Scripts
Run the following commands in the project's root directory to set up your database and model.
  * To run ETL pipeline that cleans data and stores in database python-`data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
  * To run ML pipeline that trains classifier and saves model- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
  * Run the following command in the app's directory to run the web app- `python run.py`
  * open another Terminal Window in IDE workspace and type- `env|grep WORK`
  * In a new web browser window, type- 'https://SPACEID-3001.SPACEDOMAIN' and replace `SPACEID` with `SPACEID` form previous step.
  * Launch the web broswer which should bring the web application.
## Results
- ETL pipeline-To clean,transform and load clean data into a SQLite database.
- ML pipeline-To create a best ML classifier which outputs multiclasses
- Flask app- A Web page to show data visualization and classify any message that users would enter on the web page.
## Visuals from Web page
![Classification](https://github.com/kumarvinit15/Disaster-Response-Pipeline/blob/master/images/web%20page.PNG)
![Classification](https://github.com/kumarvinit15/Disaster-Response-Pipeline/blob/master/images/distr%20of%20messages%20genre.PNG)
![Classification](https://github.com/kumarvinit15/Disaster-Response-Pipeline/blob/master/images/message%20per%20category.PNG)
![Classification](https://github.com/kumarvinit15/Disaster-Response-Pipeline/blob/master/images/heatmap.PNG)
## Licensing, Authors, and Acknowledgements
We acknowledge and thank the effort of FigureEight for providing the dataset to be used by this project.
We also acknowledge and thank our Udacity team for providing a good starter code and easy to understand instructions towards completion of this project.

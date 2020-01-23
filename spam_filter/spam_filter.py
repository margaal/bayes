# we need pandas to get data from csv file
from pandas import read_csv
# we need train_test_split to split our dataset in two parts
from sklearn.model_selection import train_test_split
# we need nltk module to get stopwords list to preproccess the content of document
import nltk
# remember to download stopwords if you don't have it:
# nltk.download('stopwords')
from nltk.corpus import stopwords
# we need CountVectorizer class to transform our train text to Bag of Words(matrix of token counts)
from sklearn.feature_extraction.text import CountVectorizer
# TfidfTransformer is important because it will give us "tf-idf(Term frequency times inverse document-frequency)"
# which is a statistical measure used to evaluate the importance of a word for a document in a collection or a corpus
from sklearn.feature_extraction.text import TfidfTransformer
# Naive Bayes algorithm we will use, MultinomialNB because 
from sklearn.naive_bayes import MultinomialNB
# Pipeline is important to store a pipeline of workflow. This will allow us to keep the configuration of our model so that we can reuse it
from sklearn.pipeline import Pipeline
# we need these modules to evaluate our model
from sklearn.metrics import classification_report,confusion_matrix


# 1. Load data
dataset = read_csv('../resources/datasets/spamraw.csv', encoding='utf-8')

# 2. Become familiar with the data by exploring it
dataset.head()
dataset.describe()
###
# the following instructions in this step are not necessary, I did it to learn certains features of pandas
def normalize_class(s):
    return 0 if s.strip()=='ham' else 1
dataset = dataset.rename(columns={'type':'class'})
dataset['class'] = dataset['class'].apply(normalize_class) # from this level I will use 0 for "ham" and 1 for "spam"
dataset['length'] = dataset['text'].apply(len)
dataset.hist(column='length', by='class', bins=50)
# you can delete dataset column 'length' because, we don't need it anymore
dataset.drop(columns=['length'])


# 3. Prepare data for Model by divide it in two parts, train and test data

# I divide a dataset by taking 20% for test and rest for modele's training
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], dataset['class'], test_size=.2, random_state=1)


# 4. Create Pipeline by
# Building "Bag of Words", normalize it by using TfidfTransformer and use Multinomial Naive Bayes to classifier our data
spam_filter_pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer='word', stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())])
    
spam_filter_pipeline.fit(x_train, y_train)
predictions = spam_filter_pipeline.predict(x_test)

# 5. Model evaluation
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
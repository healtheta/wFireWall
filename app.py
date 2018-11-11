import re;
import string;
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

def pre_process_clean(data):
	clnsms_words = [];
	stemmer = PorterStemmer();
	for sms in data:
		sms = sms.lower()	
		sms = re.sub(r"http\S+", "", sms)
		sms = re.sub('\d+','',sms)	
		sms = re.sub(r'['+string.punctuation+']', r'', sms)
		sms = ' '.join( [w for w in sms.split() if len(w)>1] )
		words = [stemmer.stem(word) for word in sms.split() if word not in stopwords.words('english')]
		clnsms_words.append(words)
		
	cln_sms_set = [];
	for row in clnsms_words:
		sms = '';
		for word in row:
			sms = sms + ' ' + word;
			
		cln_sms_set.append(sms);
		
	return cln_sms_set
	
def pre_prep_labels(labels):
	encoder = LabelEncoder()
	encoder.fit(['ham','spam'])
	return list(encoder.transform(labels))
	
def extract_features(e):
	c_vect = CountVectorizer()
	c_vect_ngram = CountVectorizer(ngram_range = (2,3))
	
	c_vect.fit(e)
	c_vect_ngram.fit(e)
	
	e_count = c_vect.transform(e)
	e_ngram = c_vect_ngram.transform(e)
	
	return [e_count, e_ngram]
	
def do_prediction(classifier , x_train, x_test, y_train):
	classifier.fit(x_train,y_train)
	predict = classifier.predict(x_test)
	return predict	
	
def calc_accuracy(predict, y_test):
	accuracy = accuracy_score(predict, y_test)
	return accuracy
	
def calc_confusion_mtrx(predict, y_test):
	conf_mat = confusion_matrix(y_test, predict)
	return conf_mat
	
def calc_precision(conf_matrix):
	precision = conf_matrix[1,1]/(conf_matrix[1,1] + conf_matrix[0,1])
	return precision
	
def calc_recall(conf_matrix):
	recall = conf_matrix[1,1]/(conf_matrix[1,1] + conf_matrix[1,0])
	return recall
	
def save_pipeline(pipeline):
	joblib.dump(pipeline, 'lib/firewall.model')
	
def load_pipeline():
	pipeline = joblib.load('lib/firewall.model');
	return pipeline;
	
def train_and_test(data, printme = False):
	#prepare x, y dataset
	x = pre_process_clean(data.v2)
	y = pre_prep_labels(data.v1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 40)
	
	count_vect = CountVectorizer()
	count_vect_ngram = CountVectorizer(ngram_range = (2,3))
	
	count_vect.fit(x_train)
	
	x_train_count = count_vect.transform(x_train)
	x_test_count = count_vect.transform(x_test)
	
	LR = LogisticRegression()
	DTC = DecisionTreeClassifier()
	RFC = RandomForestClassifier()
	NBMC = MultinomialNB()
	KNNC = KNeighborsClassifier()
	classifiers =   {'Logistic Regression' : LR, 'Decision Tree': DTC, 'Random Forest': RFC, 'Naive Bayes' : NBMC, 'KNN': KNNC}	
	
	summary  = []
	prediction = []
	for name, classifier in classifiers.items():
		predict = do_prediction(classifier,x_train_count,x_test_count,y_train)
		accuracy = calc_accuracy(predict,y_test)
		confusion_matrix = calc_confusion_mtrx(predict,y_test)
		precision = calc_precision(confusion_matrix)
		recall = calc_recall(confusion_matrix)
		prediction.append(predict)
		summary.append((name,[accuracy, precision, recall]))
	
	result_summary =  pd.DataFrame.from_items(summary,orient='index', columns=['Accuracy', 'Precision', 'Recall'])
	
	if(printme):
		print(result_summary)	
		return null
	else:
		return [NBMC , count_vect]

	

def main():
	
	data = pd.read_csv('data/spam.csv', encoding = 'latin-1')
	data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)
	
	#train and run tests on a prospective model
	model, vector = train_and_test(data, False)	#returns the most precise accurate recall model

	#create and save the pipline as a library
	pipeline = Pipeline([('vectorizer', vector), ('nbmc', model)])
	save_pipeline(pipeline)
	
	
	#let's load the pipeline and deploy in in an enviornment
	sample_sms = ["Wola you have won 10000$ in cash prizes" , "mom i will be home" , " dialog arathana, contact us if you need Cash Prize" , "Call us if you want loan schemes", "Darez Sale starts on the 11th november! Enjoy further 10% savings with COMBANK debit cards. Visit http://bit.lz.eu. Maximum Discount of Rs 500"]
	sample_sms = pre_process_clean(sample_sms)
	
	#load the pipeline
	pipeline = load_pipeline()		
	predict = pipeline.predict(sample_sms)
	print(predict)
	
main()
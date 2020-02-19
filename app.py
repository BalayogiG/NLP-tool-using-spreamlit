# Packages
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from PIL import Image
import streamlit as st 
import pandas as pd
import warnings
import spacy
import time
import sys

# To ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Data loading for Sentiment analysis
file = pd.read_excel('dataset/imdb_sentiment.xlsx')
train = list(zip(file['sentence'],file['sentiment_value']))
cl = NaiveBayesClassifier(train)

# loading spacy langauge data
nlp = spacy.load('en_core_web_sm')


# Algorithms
def Sentiment_analysis(sentence):
	sentiment_type = cl.classify(sentence)
	return sentiment_type
	

def Word_similarity(words):
	tokens = nlp(words)
	t1, t2 = tokens[0], tokens[1]
	return "Similarity :" + str(t1.similarity(t2))

def Parts_of_speech_tagging(sentence):
	doc = nlp(sentence)
	d = pd.DataFrame()
	for token in doc:
	    temp = pd.DataFrame({'Text': [token.text], 'Lemma': [token.lemma_], 'pos':
	        [token.pos_], 'Tag':[token.tag_], 'Dep':[token.dep_], 'Shape':[token.shape_], 'Is_Alpha':[token.is_alpha], 'Is_Stop':[token.is_stop]})
	    d = pd.concat([d,temp])
	st.write(d)
	
# Main driver 
image = Image.open('images/nlp.jpg')
st.image(image, caption='Natural Language Processing',use_column_width=True)
st.title("Natural Language Processing Tools")

df = pd.DataFrame()
sentence = ''
sentiment_type = ''

df['tools'] = ['Select tool','Sentiment analysis','Word similarity','Parts-of-speech tagging']

option = st.sidebar.selectbox(
    'Select the Tool',
     df['tools'])

if option == 'Select tool':
	option = ''

if option == 'Sentiment analysis':
	st.write('Simple Sentiment analysis using NaiveBayesClassifier')
	sentence = st.text_input("Enter a sentence: ")
	if sentence != '':
		st.write(Sentiment_analysis(sentence))


if option == 'Word similarity':
	st.write('Simple word similarity identifier using Spacy')
	words = st.text_input("Enter words with space:")
	if words != '':
		st.write(Word_similarity(words))

if option == 'Parts-of-speech tagging':
	st.write('Simple Parts_of_speech_tagging using Spacy')
	sentence = st.text_input("Enter sentence: ")
	if sentence != '':
		Parts_of_speech_tagging(sentence)



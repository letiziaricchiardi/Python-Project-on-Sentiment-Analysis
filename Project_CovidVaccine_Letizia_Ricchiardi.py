#!/usr/bin/env python
# coding: utf-8

# # Covid Vaccine

# Since 2020 Covid changed our way of living. It has already been a year since scientists found a vaccine that could help fighting against the virus and that could actually bring our lives back to normal. 
# Since November 2021 a new form of the virus, Omicron, is spreading and it is necessary to be really careful. A new dose of the Vaccine is now on the market. People need to take this third intake since the last two are not strong enough to protect themselves from the new variant.
# 
# I decided to analyse people thoughts and reactions to the vaccine, since many do not trust it and do not want to be vaccinated. In order to perform this analysis I studied Tweets accessing the Tweepy library.
# 
# First of all I cleaned the data. Secondly, I performed some descriptive analysis. I visualized the most frequent words in the tweets and computed their frequencies, I plotted some graphs of my results. Third, I performed some sentiment analysis using Naive Bayes classifier to classify tweets in positive or negative. Finally, I performed some sentiment analysis with TextBlob library (to determine whether a piece of writing is positive, negative, or neutral) and I computed the polarity and subjectivity of a sentence. 

# In[1]:


# I import all the libraries that I need in order to perform the analysis

import tweepy
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytz
import time
import warnings
import datetime
from datetime import date
from tweepy import OAuthHandler
from dotenv import load_dotenv


# In[2]:


# Get the credentials to acces the Tweets

import os

consumer_key = os.environ["CONSUMER_KEY"]
consumer_secret_key = os.environ["CONSUMER_SECRET_KEY"]
access_token = os.environ["ACCESS_TOKEN"]
access_secret_token = os.environ["ACCESS_SECRET_TOKEN"]

consumer_key = consumer_key
consumer_secret_key = consumer_secret_key
access_token = access_token
access_secret_token = access_secret_token
auth = OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_secret_token)
api = tweepy.API(auth)


# In[4]:


# Create a list of tweets with the word "CovidVaccine" in order to study people opinion on this matter.

searched_tweets = []
categorie=["text"]
csvWriter.writerow(categorie)
  
public_tweets_list = tweepy.Cursor(
    api.search_tweets,
    q="CovidVaccine -filter:retweets",
    result_type="mixed",
    tweet_mode='extended',
    lang='en',
).items()

for tweet in public_tweets_list:
    text = tweet._json["full_text"]
    print(text)

    favourite_count = tweet.favorite_count
    retweet_count = tweet.retweet_count
    created_at = tweet.created_at
    
    line = {'tweet' : text, 'favourite_count' : int(favourite_count), 'retweet_count' : int(retweet_count), 'created_at' : created_at}
    searched_tweets.append(line)
csvFile.close()  


# In[2]:


import pickle

# read the file and loading it into a new variable
with open("CovidVaccine_tweets.pkl", "rb") as f:
    searched_tweets = pickle.load(f)

print(searched_tweets)


# In[3]:


# I generate a DataFrame with my tweets
df = pd.DataFrame(searched_tweets)
df.head(10)


# In[4]:


df.shape


# In[5]:


# Import all libraries I need for performing both descriptive analysis and sentiment analysis

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
import numpy as np
import re, string, random
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from wordcloud import WordCloud
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples, stopwords
from nltk import FreqDist, classify, NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from wordcloud import WordCloud
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
import preprocessor as p
nltk.download('punkt')                      # list of punctuation
nltk.download('averaged_perceptron_tagger') # pretrained part of speech tagger (English)
nltk.download('stopwords')                  # list of stopwords
nltk.download('wordnet')                    # english semantic database


# # CLEANING DATA

# All fresh data should be processed before the consumption.

# In[6]:


# First step: clean raw tweets with tweet-preprocessor library
# I print a tweet example to check 

tweet_ex = '@ROWPublicHealth Thx to the excellent and caring staff at Pinebush Clinic for helping my anxious little one get her 2nd dose! #CovidVaccine 💉😬👍😁🤗 https://t.co/5erEcv6c4h'
cleaned_tweet = p.clean(tweet_ex)
print(cleaned_tweet)


# In[7]:


# Second step: tokenization
# I just want lower letters 

tokenized_text = sent_tokenize(cleaned_tweet.lower())
print(tokenized_text)


# In[8]:


# Word tokenization

cleaned_tweet_token=word_tokenize(cleaned_tweet.lower())
print(cleaned_tweet_token)
print(type(cleaned_tweet_token))


# In[9]:


# Third step: non-alphabetic characters

cleaned_tweet_token_no_punct = [w for w in cleaned_tweet_token if w.isalpha()]
print(cleaned_tweet_token_no_punct)


# In[10]:


# Fourth step: Stopwords
# List of stopwords 

stop_words=set(stopwords.words("english"))
print(stop_words)


# In[11]:


# Removing stopwords from my tweet example 

cleaned_tweet_token_no_punct_no_stop=[]
for w in cleaned_tweet_token_no_punct:
    if w not in stop_words:
        cleaned_tweet_token_no_punct_no_stop.append(w)

# I print the tweet before and after in order to see the differences

print("Tokenized Sentence:",cleaned_tweet_token_no_punct)
print("\n\nFilterd Sentence:",cleaned_tweet_token_no_punct_no_stop)


# In[12]:


# Fifth step: POS tagging and Lemmatization

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
tagged_words= nltk.pos_tag(cleaned_tweet_token_no_punct_no_stop)
print(tagged_words)

tagged_words = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), tagged_words)

lem = WordNetLemmatizer()
lem_words=[]
for word, tag in tagged_words:
        print('word: ', word, ' tag',tag)
        if tag is None:
            #if there is no available tag, append the token as is
            lem_words.append(word)
        else:        
           # else use the tag to lemmatize the token
            lem_words.append(lem.lemmatize(word, tag))
print('lemmtize words: ', lem_words)


# In[13]:


# Print the final result: the tweet example is now cleaned and ready to be analyzed

print("Filterd Sentence:",cleaned_tweet_token_no_punct_no_stop)
print("\nLemmatize Sentence:",lem_words)


# In[14]:


# I have to apply the same procedure to all my tweets

def clean_tweet(row):
  tweet = row['tweet']
  tweet_cleaned = ''
  try:
     tweet_cleaned = p.clean(tweet)
  except:
    print(tweet)
  return tweet_cleaned

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
def lemmatize_words(word_to_lemmatize):
  lem = WordNetLemmatizer()
  #detect type of words (adverb,noun,etc)
  part_of_speech_tag = pos_tag([word_to_lemmatize])
  pos_to_lemmatize = nltk_pos_tagger(part_of_speech_tag[0][1])
  if pos_to_lemmatize is None:
    lm_word = lem.lemmatize(word_to_lemmatize)
  else:
    lm_word = lem.lemmatize(word_to_lemmatize,pos=pos_to_lemmatize)
  return lm_word

def process_text(row):
    twt = row['tweet_cleaned']
    stop_w = stopwords.words('english')
    a = set(stop_w)
    tokens = nltk.word_tokenize(twt.lower())
    token_words = [w for w in tokens if w.isalpha()]
    stop_words = [w for w in token_words if not w in a]
    meaningful_words = [lemmatize_words(word) for word in stop_words]
    return meaningful_words 


# In[15]:


# I summarize my results in the DataFrame adding two columns `tweet_cleaned` and `tweet_preprocessed`

df['tweet_cleaned'] = df.apply(clean_tweet,axis=1)
df['tweet_preprocessed'] = df.apply(process_text,axis=1)
df.head(10)


# In[ ]:





# # Descriptive statistics

# In[16]:


# I want to visualize most frequent words:

words_list = []
for index,tr in df.iterrows():
  for x in list(tr['tweet_preprocessed']):
    words_list.append(x)

freq_dist = FreqDist(words_list)
print(type(freq_dist))
print(freq_dist.most_common(50))


# In[22]:


#I get a first plot
freq_dist.plot(50,cumulative=False)
plt.show()


# In[17]:


#I add two columns to my DataFrame collecting the words and their frequencies
freq_words = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])

#I remove the search key words 
freq_words_cln = freq_words.loc[(freq_words['Word'] != 'CovidVaccine')] 

#I sort the 50 values with most common frequency
top_50_freq_words_cln = freq_words_cln.sort_values(by=['Frequency'],ascending=False)[:50]

print(top_50_freq_words_cln)


# In[18]:


#I can plot a bar chart of my results
frequency_word_plot = px.bar(top_50_freq_words_cln, 
             x='Frequency',
             y='Word',
             orientation = 'h', 
             template = 'plotly_white',
             color = 'Frequency',
             range_color=[50, 500])
frequency_word_plot.show()


# In[44]:


#I want to create a WordCloud 
list_completed = ' '.join([str(elem) for elem in words_list])
wordcloud = WordCloud(
    background_color = 'white', 
    width = 5000, height = 3000, 
    random_state = 0,
    max_words= 200,
    collocations=False,
).generate(list_completed)

plt.figure(figsize=(20,20), facecolor = None)
plt.title(
    'Covid Vaccine, people talking',
    fontweight = 'bold',
    fontsize=50
)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# # Sentiment Analysis with TextBlob

# To determine whether a piece of writing is positive, negative, or neutral.

# In[19]:


from textblob import TextBlob


# In[24]:


# Create a function to get the Subjectivity

def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to compute Subjectivity Analysis

def getSubjectivityAnalysis(score):
  if score < 0.50:
    return 'Objective'
  elif score == 0.50:
    return 'Neutral'
  elif score > 0.50:
    return 'Subjective'


# In[25]:


# Create a function to get the Polarity

def getPolarity(text):
   return TextBlob(text).sentiment.polarity

# Create a function to compute Polarity Analysis

def getPolarityAnalysis(score):
 if score < 0:
  return 'Negative'
 elif score == 0:
  return 'Neutral'
 elif score > 0:
  return 'Positive'


# In[26]:


# Add in DataFrame colums indicating Subjectivity and Subjectivity Analysis
df['subjectivity'] = df['tweet_cleaned'].apply(getSubjectivity)
df['subjectivity_analysis'] = df['subjectivity'].apply(getSubjectivityAnalysis)

# Add in DataFrame colums indicating Polarity and Polarity Analysis
df['polarity'] = df['tweet_cleaned'].apply(getPolarity)
df['polarity_analysis'] = df['polarity'].apply(getPolarityAnalysis)
df.head(10)


# In[27]:


# Create a function to allocate the sentiment in a specific quadrant

def assignQuadrant(pol,subj):
    return pol + " and " + subj

# Add info in DataFrame
df['type'] = np.vectorize(assignQuadrant)(df['polarity_analysis'], df['subjectivity_analysis'])
df.head(10)


# In[28]:


# Count of Subjectivity results
frequency_sub = df.groupby(['subjectivity_analysis']).size().reset_index(name='cnt')
frequency_sub.head()


# In[29]:


# Count of Polarity results
frequency_pol = df.groupby(['polarity_analysis']).size().reset_index(name='cnt')
frequency_pol.head()


# In[36]:


# Pie-charts

donut_cnt = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
donut_cnt.add_trace(go.Pie(labels=frequency_pol.polarity_analysis, values=frequency_pol.cnt,marker_colors=px.colors.sequential.Plasma_r,textinfo='label+percent',title = 'Polarity'),1, 1)
donut_cnt.add_trace(go.Pie(labels=frequency_sub.subjectivity_analysis, values=frequency_sub.cnt,marker_colors=px.colors.sequential.Plasma,textinfo='label+percent',title = 'Subjectivity'),1, 2)

donut_cnt.update_traces(hole=.5, hoverinfo="label+percent")

donut_cnt.update_layout(title_text = 'Size of Polarity results and Subjectivity results')
donut_cnt.update_layout(showlegend=False)

donut_cnt.show()


# In[30]:


# I want to perform quadrant analysis, first I remove `neutral` tweets

quadrant = df.loc[(df['polarity_analysis'] != 'Neutral') & (df['subjectivity_analysis'] != 'Neutral')]


# In[48]:


# I want to perform quadrant analysis

colorscales = px.colors.named_colorscales()
quadrant_scatter_plot = px.scatter(quadrant,x="polarity",y="subjectivity",color="type",hover_data=['tweet','favourite_count','retweet_count'],template = 'plotly_white',
                 title="Scatter Plot focused on subjectivity and polarity (neutral excluded)")
quadrant_scatter_plot.show()


# In[ ]:





# # Sentiment Analysis using Naïve Bayes
# 
# The Naive Bayes classifier in NLTK is used to classify the tweets

# In[37]:


# Load the text fields of the positive and negative tweets

all_positive_tweets= twitter_samples.strings('positive_tweets.json')
all_negative_tweets= twitter_samples.strings('negative_tweets.json')


# In[38]:


# Details of the sample dataset

print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of positive tweets: ', len(all_negative_tweets))


# In[39]:


# Print an example of a positive tweet

print("Positive tweet example:")
print(all_positive_tweets[0])

# Print an example of a negative tweet

print("\nNegative tweet example:")
print(all_negative_tweets[0])


# In[40]:


# First step: clean the data

# Remove hyperlinks, Twitter marks and styles
def remove_hyperlinks_marks_styles(tweet):
    
    new_tweet = re.sub(r'^RT[\s]+', '', tweet)                   # remove the old styles retweet text 'RT'

    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', new_tweet)   # remove the hyperlinks

    new_tweet = re.sub(r'#', '', new_tweet)                      # remove hashtags
    
    return new_tweet

#Tokenize the string
tokenizer= TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
def tokenize_tweet(tweet):
    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens

#Remove stopwords and punctuation
stopwords_english = stopwords.words('english')
punctuation = string.punctuation

def remove_stopwords_punctuation(tweet_tokens):
    
    tweet_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
        word not in string.punctuation):
            tweet_clean.append(word)
    return tweet_clean

#Stemming (convert a word to its most general form, or stem)
stemmer = PorterStemmer() 

def get_stem(tweet_clean):
    
    tweets_stem=[]
    
    for word in tweet_clean:
        stem_word= stemmer.stem(word)
        tweets_stem.append(stem_word)
        
    return tweets_stem


# In[41]:


# Combine all preprocessed techniques

def process_tweet(tweet):
    processed_tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens= tokenize_tweet(processed_tweet)
    tweets_clean= remove_stopwords_punctuation(tweet_tokens)
    tweets_stem= get_stem(tweets_clean)
    
    return tweets_stem


# In[42]:


# Apply function `process_tweet` to all tweets

processed_positive_tweets = [process_tweet(tw) for tw in all_positive_tweets]
processed_negative_tweets = [process_tweet(tw) for tw in all_negative_tweets]


# In[57]:


# I print all my positive tweets cleaned

processed_positive_tweets


# In[58]:


# I print all my negative tweets cleaned

processed_negative_tweets


# In[43]:


# To apply Naive Bayes, first define function in order to transform list of lists in list of dictionaries

def list_to_dict(clean_tweet_list):
    all_tweet_list=[]
    tweet_dict={}
    for tweet in clean_tweet_list:
        
        for token in tweet:
            tweet_dict[token]=True         
        
        all_tweet_list.append(tweet_dict)
        tweet_dict={}
    return all_tweet_list


# In[44]:


# Apply function to tweets

positive_tokens_for_model = list_to_dict(processed_positive_tweets)
negative_tokens_for_model = list_to_dict(processed_negative_tweets)
 
positive_tokens_for_model


# In[45]:


# Adapt inputs (tweet training data) in order to be able to perform NaiveBayesClassifier
# I need a tuple for each element in my dictionary => generate a list of tuples as (dict,'sentiment')

positive_dataset = [(tweet_dict, 'Positive') for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, 'Negative') for tweet_dict in negative_tokens_for_model]

# Create a unique dataset

dataset = positive_dataset + negative_dataset


# In[46]:


# Split training data in training and testing data (80%-20%) to see how NaiveBayesClassifier performs

random.shuffle(dataset)

train_data = dataset[:8000]
test_data = dataset[8000:]


# In[47]:


# Training of the model with `NaiveBayesClassifier.train` 

classifier = NaiveBayesClassifier.train(train_data)


# In[48]:


# Show most informative features

print(classifier.show_most_informative_features(20))


# In[49]:


# Compute accuracy on testset to see how the model would perform if applied to original dataset

print("Accuracy is:", classify.accuracy(classifier, test_data))


# In[50]:


# Now I can make some predictions
# First I need the list of all the tweets already cleaned and preprocessed

tweet_prediction = df["tweet_preprocessed"].values.tolist()
print(tweet_prediction)


# In[51]:


# I apply `list_to_dict` function in order to transform a list of lists in a list of dictionaries

custom_tokens_for_model= list_to_dict(tweet_prediction)
custom_tokens_for_model


# In[52]:


# For each dictionary in `custom_tokens_for_model` list, I apply the classifier

predictions = []
for tw in custom_tokens_for_model:
    predictions.append(classifier.classify(tw))


# In[53]:


for (tweet, pred) in zip(tweet_prediction, predictions):
    print(pred, " - ", " ".join(tweet))


# In[54]:


df['sentiment'] = predictions
df


# In[55]:


# Count number of positive results

frequency_sentiment = df.groupby(['sentiment']).size().reset_index(name='cnt')
frequency_sentiment.head()


# In[56]:


# Plot the results

donut_cnt = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
donut_cnt.add_trace(go.Pie(labels=frequency_sentiment.sentiment, values=frequency_sentiment.cnt,marker_colors=px.colors.sequential.Plasma_r,textinfo='label+percent',title = 'Sentiment Analysis'),1, 1)

donut_cnt.update_traces(hole=.5, hoverinfo="label+percent")

donut_cnt.update_layout(title_text = 'Size of Sentiment Analysis results')
donut_cnt.update_layout(showlegend=False)

donut_cnt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# Implementation with frequency Dictionary


# # Implementation: create the Frequency Dictionary
# 
# Probability of each tokenized word being in a positive or negative tweet:
# 
#  1. preprocess the data;
#  2. compute a Frequency Dictionary (how many times a word is seen in a positive or negative tweet).
# 
# To compute the probability: (# times “neg/pos word” is in the tweet) / (tot # positive/negative count)

# In[ ]:


# I need to split my data in a Training Set

train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg

train_y = np.append(np.ones((len(train_pos))), np.zeros((len(train_neg))))


# In[ ]:


#Create the Frequency Dictionary function

def create_frequency(tweets, ys):            #The parameters "tweets" contains a list of tweets
    freq_d={}                                #The parameter "ys" contains the corresponding y-value
                                            
#Use Python dictionary to store the data

    for tweet, y in zip(tweets, ys): 
        for word in process_tweet(tweet):
            #define the key, which is the word and label tuple
            pair = (word, y)  
            if pair in freq_d:
                freq_d[pair]+=1
            else:
                freq_d[pair]=freq_d.get(pair,1)
    return freq_d


# In[ ]:


#Testing the function

tweets=['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys=[1,0,0,0,0]

freq_d= create_frequency(tweets,ys)
print(freq_d)


# In[ ]:


#Build the frequency dictionary using the actual training set 

freqs= create_frequency(train_x, train_y)


# In[ ]:


# Compute probability of each word being in a positive or negative tweet


# In[ ]:


#Train the model using Naive Bayes
#Define the "Train Naive Bayes" function

def train_naive_bayes(freqs,train_x,train_y):
    logliklihood = {}
    logprior = 0

    #count the tot number of positive and negative words for all the tweets    
    unique_words= set([pair[0] for pair in freqs.keys()])    # calculate V, number of unique words in the dictionary
    V = len(unique_words)
    
    N_pos = 0
    N_neg = 0                           # Calculate N_pos, N_neg
    for pair in freqs.keys():                   
        if pair[1]>0:
            N_pos += freqs[pair]                # N_pos : total number of positive words for all tweets       
        else:
            N_neg += freqs[pair]                # N_neg : total number of negative words for all tweets  

    #Compute total number of tweets, positive and negative tweets
    D = train_y.shape[0]                               
    D_pos = sum(train_y)
    D_neg = D - sum(train_y)

    #Calculate the logprior (number of positive tweets /number of negative tweets)
    logprior = np.log(D_pos) - np.log(D_neg)
    
    #For each unique word
    for word in unique_words:
        
        #get the positive and negative frequency of the word
        freqs_pos = freqs.get((word, 1),0)
        freqs_neg = freqs.get((word, 0),0)

        #calculte the probability of each word being positive and negative
        p_w_pos = (freqs_pos+1)/(N_pos+V)
        p_w_neg = (freqs_neg+1)/(N_neg+V)
        #Add the "+1" in the numerator for additive smoothing
        
        #Calculate the loglikelihood of the word 
        logliklihood[word] = np.log(p_w_pos/p_w_neg)
    
    return logprior, logliklihood


# In[ ]:


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
print(logprior)                 #probability of the tweets given being positive or negative
print(len(loglikelihood))       #number of unique words, I have probability of each of these being in a positive tweet


# In[ ]:


# Now: make predictions!


# Naive Bayes Predict function to make predictions on tweets.
#   1. The function takes in the tweet, logprior, loglikelihood.
#   2. It returns the probability that the tweet belongs to the positive or negative class.
#   3. For each tweet, sum up loglikelihoods of each word in the tweet.
#   4. Also add the logprior to this sum to get the predicted sentiment of that tweet.

# In[ ]:


#Output is how confident it is for a tweet to be positive

def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    # initialize probability to zero
    p = 0                              #keeps track of the confidence of the tweet
    # add the logprior (whenever the data is not perfectly balanced, the logprior will be a non-zero value)
    p+=logprior                        
    for word in word_l:
        
        #check if the word exists in the loglikelihood dictionary
        if word in loglikelihood: 
            # add the log likelihood of that word to the probability
            p+=loglikelihood[word]
    
    return p


# In[ ]:


tweets=[tweet_ex]
for tweet in tweets:
    p=naive_bayes_predict(tweet, logprior, loglikelihood)
print(f'{tweet}->{p:.2f}')


# In[ ]:





# In[ ]:





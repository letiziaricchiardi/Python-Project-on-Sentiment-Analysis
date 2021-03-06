#!/usr/bin/env python
# coding: utf-8

# # Covid Vaccine

# Since 2020 Covid changed our way of living. It has already been a year since scientists found a vaccine that could help fighting against the virus and that could actually bring our lives back to normal. 
# Since November 2021 a new form of the virus, Omicron, is spreading and it is necessary to be really careful. A new dose of the Vaccine is now on the market. People need to take this third intake since the last two are not strong enough to protect themselves from the new variant.
# 
# I decided to analyse people thoughts and reactions to the vaccine, since many do not trust it and do not want to be vaccinated. In order to perform this analysis I studied Tweets accessing the Tweepy library.
# 
# The first section of this project presents the cleaning of the data. The second section shows some descriptive analysis: which are the most frequent words in the tweets and their frequencies. The section also presents some graphs. Third, sentiment analysis was performed with the TextBlob library in order to determine whether a piece of writing is positive, negative, or neutral. Finally, sentiment analysis was performed using the Naïve Bayes Classifier that classifies tweets in positive or negative.

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


# In[ ]:


# Create a list of tweets with the word "CovidVaccine" in order to study people opinion on this matter
# In order to perform the analysis I always studied the same tweets, so I am not running this coding, but just showing how it should be done
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


# In[3]:


# I import tweets previously saved to perform always the analysis on the same ones
import pickle

# read the file and loading it into a new variable
with open("CovidVaccine_tweets.pkl", "rb") as f:
    searched_tweets = pickle.load(f)

print(searched_tweets)


# In[4]:


# I generate a DataFrame with my tweets
df = pd.DataFrame(searched_tweets)
df.head(10)


# In[5]:


df.shape


# In[6]:


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

# In[7]:


# First step: clean raw tweets with tweet-preprocessor library
# I print a tweet example to check if everything works out
tweet_ex = '@ROWPublicHealth Thx to the excellent and caring staff at Pinebush Clinic for helping my anxious little one get her 2nd dose! #CovidVaccine 💉😬👍😁🤗 https://t.co/5erEcv6c4h'
cleaned_tweet = p.clean(tweet_ex)
print(cleaned_tweet)


# In[8]:


# Second step: tokenization
# I just want lower letters 
tokenized_text = sent_tokenize(cleaned_tweet.lower())

# Word tokenization
cleaned_tweet_token=word_tokenize(cleaned_tweet.lower())
print(cleaned_tweet_token)
print(type(cleaned_tweet_token))


# In[10]:


# Third step: non-alphabetic characters
cleaned_tweet_token_no_punct = [w for w in cleaned_tweet_token if w.isalpha()]
print(cleaned_tweet_token_no_punct)


# In[11]:


# Fourth step: Stopwords
# List of stopwords 
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[12]:


# Removing stopwords from my tweet example 
cleaned_tweet_token_no_punct_no_stop=[]
for w in cleaned_tweet_token_no_punct:
    if w not in stop_words:
        cleaned_tweet_token_no_punct_no_stop.append(w)

# I print the tweet before and after in order to see the differences
print("Tokenized Sentence:",cleaned_tweet_token_no_punct)
print("\n\nFilterd Sentence:",cleaned_tweet_token_no_punct_no_stop)


# In[13]:


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


# In[14]:


# Print the final result: the tweet example is now cleaned and ready to be analyzed
print("Filterd Sentence:",cleaned_tweet_token_no_punct_no_stop)
print("\nLemmatize Sentence:",lem_words)


# In[15]:


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


# In[16]:


# I summarize my results in the DataFrame adding two columns `tweet_cleaned` and `tweet_preprocessed`
df['tweet_cleaned'] = df.apply(clean_tweet,axis=1)
df['tweet_preprocessed'] = df.apply(process_text,axis=1)
df.head(10)


# # Descriptive statistics

# In[17]:


# I want to visualize most frequent words:
words_list = []
for index,tr in df.iterrows():
  for x in list(tr['tweet_preprocessed']):
    words_list.append(x)

freq_dist = FreqDist(words_list)
print(type(freq_dist))
print(freq_dist.most_common(50))


# In[68]:


#I get a first plot
freq_dist.plot(50,cumulative=False)
plt.show()


# In[20]:


#I add two columns to my DataFrame collecting the words and their frequencies
freq_words = pd.DataFrame(list(freq_dist.items()), columns = ["Word","Frequency"])

#I remove the search key words 
freq_words_cln = freq_words.loc[(freq_words['Word'] != 'CovidVaccine')] 

#I sort the 50 values with most common frequency
top_50_freq_words_cln = freq_words_cln.sort_values(by=['Frequency'],ascending=False)[:50]

print(top_50_freq_words_cln)


# In[21]:


#I can plot a bar chart of my results
frequency_word_plot = px.bar(top_50_freq_words_cln, 
             x='Frequency',
             y='Word',
             orientation = 'h', 
             template = 'plotly_white',
             color = 'Frequency',
             range_color=[50, 500])
frequency_word_plot.show()


# In[22]:


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

# In[23]:


from textblob import TextBlob


# In[34]:


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


# In[35]:


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


# In[36]:


# Add in DataFrame colums indicating Subjectivity and Subjectivity Analysis
df['subjectivity'] = df['tweet_cleaned'].apply(getSubjectivity)
df['subjectivity_analysis'] = df['subjectivity'].apply(getSubjectivityAnalysis)

# Add in DataFrame colums indicating Polarity and Polarity Analysis
df['polarity'] = df['tweet_cleaned'].apply(getPolarity)
df['polarity_analysis'] = df['polarity'].apply(getPolarityAnalysis)
df.head(10)


# In[37]:


# Create a function to allocate the sentiment in a specific quadrant
def assignQuadrant(pol,subj):
    return pol + " and " + subj

# Add info in DataFrame
df['type'] = np.vectorize(assignQuadrant)(df['polarity_analysis'], df['subjectivity_analysis'])
df.head(10)


# In[38]:


# Count of Subjectivity results
frequency_sub = df.groupby(['subjectivity_analysis']).size().reset_index(name='cnt')
frequency_sub.head()


# In[39]:


# Count of Polarity results
frequency_pol = df.groupby(['polarity_analysis']).size().reset_index(name='cnt')
frequency_pol.head()


# In[40]:


# Pie-charts
donut_cnt = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
donut_cnt.add_trace(go.Pie(labels=frequency_pol.polarity_analysis, values=frequency_pol.cnt,marker_colors=px.colors.sequential.Plasma_r,textinfo='label+percent',title = 'Polarity'),1, 1)
donut_cnt.add_trace(go.Pie(labels=frequency_sub.subjectivity_analysis, values=frequency_sub.cnt,marker_colors=px.colors.sequential.Plasma,textinfo='label+percent',title = 'Subjectivity'),1, 2)

donut_cnt.update_traces(hole=.5, hoverinfo="label+percent")

donut_cnt.update_layout(title_text = 'Size of Polarity results and Subjectivity results')
donut_cnt.update_layout(showlegend=False)

donut_cnt.show()


# In[41]:


# I want to perform quadrant analysis, first I remove `neutral` tweets
quadrant = df.loc[(df['polarity_analysis'] != 'Neutral') & (df['subjectivity_analysis'] != 'Neutral')]


# In[42]:


# I want to perform quadrant analysis
colorscales = px.colors.named_colorscales()
quadrant_scatter_plot = px.scatter(quadrant,x="polarity",y="subjectivity",color="type",hover_data=['tweet','favourite_count','retweet_count'],template = 'plotly_white',
                 title="Scatter Plot focused on subjectivity and polarity (neutral excluded)")
quadrant_scatter_plot.show()


# # Sentiment Analysis using Naïve Bayes
# 
# The Naive Bayes classifier in NLTK is used to classify the tweets

# In[46]:


# Load the text fields of the positive and negative tweets
all_positive_tweets= twitter_samples.strings('positive_tweets.json')
all_negative_tweets= twitter_samples.strings('negative_tweets.json')


# In[47]:


# Details of the sample dataset
print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of positive tweets: ', len(all_negative_tweets))


# In[48]:


# Print an example of a positive tweet
print("Positive tweet example:")
print(all_positive_tweets[0])

# Print an example of a negative tweet
print("\nNegative tweet example:")
print(all_negative_tweets[0])


# In[49]:


# First step: clean the data (use a different approach than before)
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


# In[50]:


# Combine all preprocessed techniques
def process_tweet(tweet):
    processed_tweet = remove_hyperlinks_marks_styles(tweet)
    tweet_tokens= tokenize_tweet(processed_tweet)
    tweets_clean= remove_stopwords_punctuation(tweet_tokens)
    tweets_stem= get_stem(tweets_clean)
    
    return tweets_stem


# In[51]:


# Apply function `process_tweet` to all tweets
processed_positive_tweets = [process_tweet(tw) for tw in all_positive_tweets]
processed_negative_tweets = [process_tweet(tw) for tw in all_negative_tweets]


# In[52]:


# I print all my positive tweets cleaned
processed_positive_tweets


# In[53]:


# I print all my negative tweets cleaned
processed_negative_tweets


# In[54]:


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


# In[55]:


# Apply function to tweets
positive_tokens_for_model = list_to_dict(processed_positive_tweets)
negative_tokens_for_model = list_to_dict(processed_negative_tweets)
 
positive_tokens_for_model


# In[56]:


# Adapt inputs (tweet training data) in order to be able to perform NaiveBayesClassifier
# I need a tuple for each element in my dictionary => generate a list of tuples as (dict,'sentiment')
positive_dataset = [(tweet_dict, 'Positive') for tweet_dict in positive_tokens_for_model]
negative_dataset = [(tweet_dict, 'Negative') for tweet_dict in negative_tokens_for_model]

# Create a unique dataset
dataset = positive_dataset + negative_dataset


# In[57]:


# Split training data in training and testing data (80%-20%) to see how NaiveBayesClassifier performs
random.shuffle(dataset)
train_data = dataset[:8000]
test_data = dataset[8000:]


# In[58]:


# Training of the model with `NaiveBayesClassifier.train` 
classifier = NaiveBayesClassifier.train(train_data)


# In[59]:


# Show most informative features
print(classifier.show_most_informative_features(20))


# In[60]:


# Compute accuracy on testset to see how the model would perform if applied to original dataset
print("Accuracy is:", classify.accuracy(classifier, test_data))


# In[61]:


# Now I can make some predictions
# First I need the list of all the tweets already cleaned and preprocessed
tweet_prediction = df["tweet_preprocessed"].values.tolist()
print(tweet_prediction)


# In[62]:


# I apply `list_to_dict` function in order to transform a list of lists in a list of dictionaries
custom_tokens_for_model= list_to_dict(tweet_prediction)
custom_tokens_for_model


# In[63]:


# For each dictionary in `custom_tokens_for_model` list, I apply the classifier
predictions = []
for tw in custom_tokens_for_model:
    predictions.append(classifier.classify(tw))


# In[64]:


for (tweet, pred) in zip(tweet_prediction, predictions):
    print(pred, " - ", " ".join(tweet))


# In[65]:


df['sentiment'] = predictions
df


# In[66]:


# Count number of positive results
frequency_sentiment = df.groupby(['sentiment']).size().reset_index(name='cnt')
frequency_sentiment.head()


# In[67]:


# Plot the results
donut_cnt = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
donut_cnt.add_trace(go.Pie(labels=frequency_sentiment.sentiment, values=frequency_sentiment.cnt,marker_colors=px.colors.sequential.Plasma_r,textinfo='label+percent',title = 'Sentiment Analysis'),1, 1)
donut_cnt.update_traces(hole=.5, hoverinfo="label+percent")
donut_cnt.update_layout(title_text = 'Size of Sentiment Analysis results')
donut_cnt.update_layout(showlegend=False)
donut_cnt.show()


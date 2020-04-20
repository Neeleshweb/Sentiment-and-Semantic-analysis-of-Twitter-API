#!/usr/bin/env python
# coding: utf-8

# In[248]:


import Credentials
from tweepy import API
import csv
from collections import Counter
from tweepy import Cursor
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import heapq 
import nltk
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from sklearn.feature_extraction.text import CountVectorizer
import re
import numpy as np
import pandas as pd

class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth=OAuthHandler(Credentials.consumer_key,Credentials.consumer_secret)
        auth.set_access_token(Credentials.access_token,Credentials.access_token_secret)
        return auth

class Streamthetweet():
    def get_tweets(self,savitinfile,list_of_tweets):
        listener=Listenthetweets(saveitinfile)
        stream=Stream(auth,listener)
        stream.filter(track=list_of_tweets)
        
        
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets
        
        
class TweetAnalyzer():
    
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        return df
        
class Listenthetweets(StreamListener):
    def __init__(self, saveitinfile):
        self.saveitinfile = saveitinfile
    def on_data(self,data):
        try:
            print(data)
            
            with open(self.saveitinfile,'w') as tf:
                tf.write(data)
                
            return True
        except BaseException as ee:
            print("error %s" % str(ee))
        return True
    def on_error(self,status):
        print(status)
        
if __name__ == '__main__':
    df2=pd.read_csv(r'G:\DALHOUSIE Term 1\Data management and ware dalhousie\Polarity2.csv')
    positive=df2[df2['Polarity']>=2].Word
    negative=df2[df2['Polarity']<=-2].Word
    neutral=df2[(df2['Polarity']>=1)&(df2['Polarity']>=-1)].Word
    poswords=positive.to_list()
    negwords=negative.to_list()
    neutralwords=neutral.to_list()

    list_of_tweets=["Canada", "University", "Dalhousie University", "Halifax",
"Canada Education"]
    saveitinfile="totalcollectedtweets.csv"
    twitter_client = TwitterClient()
    api = twitter_client.get_twitter_client_api()
    tweets = api.home_timeline(count=3000)
    authenticateit=TwitterAuthenticator()
    authenticateit.authenticate_twitter_app()
    tweet_analyzer = TweetAnalyzer()
    df = tweet_analyzer.tweets_to_data_frame(tweets)
    print(df.head(10))
    dataset=df['Tweets'].to_list()
    
    
    def unique_tweets(tweet_polarity):
        final_tweets = []
        for i in tweet_polarity:
            if i not in final_tweets:
                final_tweets.append(i)
        return final_tweets
    
    
    
    v=CountVectorizer()
    nltk.download('stopwords')
    nltk.download('punkt')
    postweet=[]
    negtweet=[]
    neutratweets=[]
    nosentimenttweet=[]
    final_df=pd.DataFrame(columns=["Tweet","tweets","Match","Polarity"])
    print(final_df.head())
    for i in range(len(dataset[:2000])):
        
        dataset[i] = dataset[i].lower() 
        dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        dataset[i] = re.sub(r'https\s+', ' ', dataset[i])
        dataset[i] = re.sub(r't co\s+', ' ', dataset[i])
        tweet1=[dataset[i]]
        
        print(tweet1)
    
        bagofwords = {}
        
        for data in tweet1:
            words = nltk.word_tokenize(data)
            for word in words:

                if word not in bagofwords.keys():
                    bagofwords[word] = 1
                else:
                    bagofwords[word] += 1
        
        
        
        for data in tweet1:
            words = nltk.word_tokenize(data)
            for word in words:
                if word in poswords:
                    postweet.append(tweet1)
                    final_df=final_df.append({"Tweet":i+1,"tweets":tweet1,"Match":word,"Polarity":'positive'},ignore_index=True)
                    print(word)
                elif word in negwords:
                    negtweet.append(tweet1)
                    final_df=final_df.append({"Tweet":i+1,"tweets":tweet1,"Match":word,"Polarity":'negative'},ignore_index=True)
                    print(word)
                elif word in neutralwords:
                    neutratweets.append(tweet1)
                    final_df=final_df.append({"Tweet":i+1,"tweets":tweet1,"Match":word,"Polarity":'neutral'},ignore_index=True)
                    print(word)
                
        print(bagofwords)
        
        positive_tweets=unique_tweets(postweet)
        
        negative_tweets=unique_tweets(negtweet)
        neutral_tweets=unique_tweets(neutratweets)
        negative_word_tweets=final_df[final_df['Polarity']=='negative'].Match
        list_of_negativewords=negative_word_tweets.to_list()
        print(list_of_negativewords)
        pos_word_tweets=final_df[final_df['Polarity']=='positive'].Match
        list_of_positivewords=pos_word_tweets.to_list()
        print(list_of_positivewords)
        neutral_word_tweets=final_df[final_df['Polarity']=='neutral'].Match
        list_of_neutralwords=neutral_word_tweets.to_list()
        print(list_of_neutralwords)

    
    word_could_dict=Counter(list_of_positivewords)
    print(word_could_dict)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(15,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    word_could_dict=Counter(list_of_negativewords)
    print(word_could_dict)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(15,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    word_could_dict=Counter(list_of_neutralwords)
    print(word_could_dict)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(15,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
    
    
        

        
        
        

        

        
    


        
    
    
    


# In[247]:


final_df


# In[255]:


positive_tweets


# In[256]:


neutral_tweets


# In[257]:


negative_tweets


# In[249]:


import json
import requests
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import math
 

headers = {'Authorization': '1a144a80798743fb9771c38a2287619d'}
 

top_headlines_url = 'https://newsapi.org/v2/top-headlines'

everything_news_url = 'https://newsapi.org/v2/everything'

sources_url = 'https://newsapi.org/v2/sources'
 
headlines_payload = {'category': 'education', 'country': 'Canada'}
everything_payload = {'q': 'Dalhousie University', 'language': 'en', 'sortBy': 'popularity','pageSize': 100}
sources_payload = {'q': 'Halifax','category': 'general', 'language': 'en', 'country': 'Canada'}
 

response = requests.get(url=everything_news_url, headers=headers, params=everything_payload)
 

pretty_json_output = json.dumps(response.json(), indent=4)
print(pretty_json_output)

 

response_json_string = json.dumps(response.json())
 

response_dict = json.loads(response_json_string)
print(response_dict)
 

articles_list = response_dict['articles']
 

df = pd.read_json(json.dumps(articles_list))
df.to_csv('G:/DALHOUSIE Term 1/Data management and ware dalhousie/newsarticles.csv')
df3=pd.read_csv('G:/DALHOUSIE Term 1/Data management and ware dalhousie/newsarticles.csv')
df3=df3[["title","description","content"]]
listofcontents=df3['content'].to_list()




# In[250]:


cleanedList = [x for x in listofcontents if str(x) != 'nan']


# In[251]:




    

subs="Canada"
subs2="business"
subs3="Dalhousie University"
subs4="University"
subs5="Halifax"
res =[i for i in cleanedList if subs in i]
res2=[j for j in cleanedList if subs2 in j]
res3=[k for k in cleanedList if subs3 in k]
res4=[l for l in cleanedList if subs4 in l]
res5=[m for m in cleanedList if subs5 in m]
final_df2=pd.DataFrame(columns=["Searchquery","noofdocsmatchquery","ratio","Log10(N/df)"])
final_df2=final_df2.append({"Searchquery":'Canada',"noofdocsmatchquery":len(res),"ratio":100/(len(res)),
                            "Log10(N/df)":math.log10(100/len(res))},ignore_index=True)
final_df2=final_df2.append({"Searchquery":'business',"noofdocsmatchquery":len(res2),"ratio":100/(len(res2)),
                            "Log10(N/df)":math.log10(100/len(res2))},ignore_index=True)
final_df2=final_df2.append({"Searchquery":'Dalhousie University',"noofdocsmatchquery":len(res3),"ratio":100/(len(res3)),
                            "Log10(N/df)":math.log10(100/len(res3))},ignore_index=True)
final_df2=final_df2.append({"Searchquery":'University',"noofdocsmatchquery":len(res4),"ratio":100/(len(res4)),
                            "Log10(N/df)":math.log10(100/len(res4))},ignore_index=True)
final_df2=final_df2.append({"Searchquery":'Halifax',"noofdocsmatchquery":len(res5),"ratio":100/(len(res5)),
                            "Log10(N/df)":math.log10(100/len(res5))},ignore_index=True)
print(final_df2.head())

 
        
    


# In[162]:


cleanedList


# In[253]:


final_df6=pd.DataFrame(columns=["articlenumber","totalwords","frequency","articlecontent"])
for z in range(len(cleanedList)):
    substr1="Canada"
    tokens=nltk.word_tokenize(cleanedList[z])
    print(tokens)
    res=[i for i in tokens if substr1 in i]
    print(res)
    print(len(tokens));print(len(res))
    final_df6=final_df6.append({"articlenumber":'Article#'+str(z),"totalwords":len(tokens),
                                "frequency":len(res),"articlecontent":cleanedList[z]},ignore_index=True)
final_df6['relativefrequency']=final_df6['frequency']/final_df6['totalwords']   
final_df7=final_df6[final_df6['frequency']>0]
print(final_df7.loc[final_df7['relativefrequency'].astype(float).idxmax(),'articlecontent'])



# In[254]:


final_df7


# In[ ]:





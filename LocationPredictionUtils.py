import overpy
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import logging
from time import sleep
from random import randint
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from ast import literal_eval
from sentence_transformers import SentenceTransformer, util
import requests
import base64
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sklearn.metrics import accuracy_score
import geopy.distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import model_selection

# In[20]:


class DBConnection:
    def __init__(self,host='localhost', port=27017, username=None, password=None):
        self.username=username
        self.password=password
        self.port=port
        self.host=host
        self.conn=self._connect_mongo()
        
        
    def _connect_mongo(self):
        """ A util for making a connection to mongo """

        if self.username and self.password:
            mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (self.username, self.password, self.host, self.port)
            conn = MongoClient(mongo_uri)
        else:
            conn = MongoClient(self.host, self.port)

        return conn

    def read_mongo(self,db_name, collection, query={}, no_id=True):
        """ Read from Mongo and Store into DataFrame """

        # Connect to MongoDB
        db = self.conn[db_name]

        # Make a query to the specific DB and Collection
        cursor = db[collection].find(query)

        # Expand the cursor and construct the DataFrame
        df =  pd.DataFrame(list(cursor))

        # Delete the _id
        if no_id:
            del df['_id']

        return df
    
    
    

    


# In[21]:


class POICrawler:
    
    def __init__(self):    
        print('POICrawler: __init__')
        
        self.api = overpy.Overpass()
        #Intialization
        self.names=[]
        self.lat=[]
        self.lon=[]
        self.amenity=[]
        self.highway=[]
        self.place=[]
        self.city=[]
        self.street=[]
        self.landuse=[]
        self.shop=[]
        self.operator=[]
        self.public_transport=[]
        self.railway=[]
        self.bus=[]
        self.roads=[]
        self.village=[]
        self.municipality=[]
        self.city=[]
        self.country=[]
        self.geolocator = Nominatim(user_agent="geoapiExercises")
        
    def extractPOI_info(self,nodes):
        for node in nodes:        
            tags=node.tags
            keys=tags.keys()
            if 'name' in tags:
                self.names.append(tags.get("name"))
                self.lat.append(node.lat)
                self.lon.append(node.lon)
                
                location=self.reverse_geocode(self.geolocator,str(node.lat)+","+str(node.lon))
                address_keys=location.raw['address'].keys()
                if "road" in address_keys:
                    self.roads.append(location.raw['address']['road'])
                else:
                    self.roads.append(np.nan)

                if "village" in address_keys:
                    self.village.append(location.raw['address']['village'])
                else:
                    self.village.append(np.nan)

                if "municipality" in address_keys:
                    self.municipality.append(location.raw['address']['municipality'])
                else:
                    self.municipality.append(np.nan)

                if "city" in address_keys:
                    self.city.append(location.raw['address']['city'])
                else:
                    self.city.append(np.nan)

                if "country" in address_keys:
                    self.country.append(location.raw['address']['country'])
                else:
                    self.country.append(np.nan)


                
                if 'amenity' in keys:
                    self.amenity.append(tags.get("amenity"))
                else:
                    self.amenity.append(np.nan)

                if 'highway' in keys:
                    self.highway.append(tags.get("highway"))
                else:
                    self.highway.append(np.nan)

                if 'place' in keys:
                    self.place.append(tags.get("place"))
                else:
                    self.place.append(np.nan)

                if 'addr:city' in keys:
                    self.city.append(tags.get("addr:city"))
                else:
                    self.city.append(np.nan)

                if 'addr:street' in keys:
                    self.street.append(tags.get("addr:street"))
                else:
                    self.street.append(np.nan)

                if 'landuse' in keys:
                    self.landuse.append(tags.get("landuse"))
                else:
                    self.landuse.append(np.nan)

                if 'shop' in keys:
                    self.shop.append(tags.get("shop"))
                else:
                    self.shop.append(np.nan)

                if 'operator' in keys:
                    self.operator.append(tags.get("operator"))
                else:
                    self.operator.append(np.nan)

                if 'public_transport' in keys:
                    self.public_transport.append(tags.get("public_transport"))
                else:
                    self.public_transport.append(np.nan)

                if 'railway' in keys:
                    self.railway.append(tags.get("railway"))
                else:
                    self.railway.append(np.nan)

                if 'bus' in keys:
                    if tags.get("bus")=='yes':
                        self.bus.append(True)
                    else:
                        self.bus.append(False)
                else:
                    self.bus.append(False)

    def find_POIs(self,queries):
        for query in queries:
            self.result = self.api.query(query)
            self.extractPOI_info(self.result.nodes)
        

    def savePOIs(self,path,columns=["name","lat","lon","amenity","highway","place","city","street","landuse","shop","operator","public_transport","railway","bus_stop","roads","village","municipality","city","country"]):
        df=pd.DataFrame(data=zip(self.names,self.lat,self.lon,self.amenity,self.highway,self.place,self.city,self.street,self.landuse,self.shop,self.operator,self.public_transport,self.railway,self.bus,self.roads,self.village,self.municipality,self.city,self.country),columns=["name","lat","lon","amenity","highway","place","city","street","landuse","shop","operator","public_transport","railway","bus_stop","roads","village","municipality","city","country"])
        df=df[columns]
        df.to_csv(path,index=False)
        
    def reverse_geocode(self,geolocator, latlon, sleep_sec=10):
        try:
            return geolocator.reverse(latlon)
        except GeocoderTimedOut:
            logging.info('TIMED OUT: GeocoderTimedOut: Retrying...')
            sleep(randint(1*100,sleep_sec*100)/100)
            return reverse_geocode(geolocator, latlon, sleep_sec)
        except GeocoderServiceError as e:
            logging.info('CONNECTION REFUSED: GeocoderServiceError encountered.')
            logging.error(e)
            return None
        except Exception as e:
            logging.info('ERROR: Terminating due to exception {}'.format(e))
            return None


# In[22]:


class MapSplit:
    def __init__(self):        
        self.grids=[]
        self.grids_dict={}
    
    def printAreaWidthHeight(self):
        print ("Height: ",geopy.distance.geodesic((self.cols[0],self.rows[0]), (self.cols[0],self.rows[1])).km)
        print ("Width: ",geopy.distance.geodesic((self.cols[0],self.rows[0]), (self.cols[1],self.rows[0])).km)
        print ("Area: ",(geopy.distance.geodesic((self.cols[0],self.rows[0]), (self.cols[1],self.rows[0])).km)*(geopy.distance.geodesic((self.cols[0],self.rows[0]), (self.cols[0],self.rows[1])).km))
        print ("Max Distance Error: ",geopy.distance.geodesic((self.cols[0],self.rows[0]), (self.cols[1],self.rows[1])).km/2)
        
    def split_into_grids(self,bottomLeft,bottomRight,topLeft,topRight,num):
        self.cols = np.linspace(bottomLeft[0], bottomRight[0], num=num)
        self.rows = np.linspace(bottomLeft[1], topLeft[1], num=num)
        x_=[]
        y_=[]
        for c in self.cols[0:-1]:
            for r in self.rows[0:-1]:
                x_.append(c+0.0000001)
                y_.append(r+0.0000001)
        
        x_ = np.searchsorted(self.cols, x_)
        y_ = np.searchsorted(self.rows,y_)
        
        for i in range(len(x_)):
            self.grids.append((x_[i],y_[i]))
        
        key=0
        for g in self.grids:
            self.grids_dict[g]=key
            key+=1
    
    def map_grid_nb_to_original_split(self,d1,d2):
        
        d3={}
        for key,value in d1.items():
            # we will put the nb in d1 as the key and the value in d2 as the value
            # example if grid nb 1 in grids_dict_POIS corresponds to grid nb 13 in grids_dict then we will have 1:13
            #in grids_dict_mapped
            if key in d2.keys():
                d3[value]=d2[key]
            else:
                d3[value]=np.nan
            
        return d3
        

class OperationsPOI(MapSplit):
    
    def __init__(self,data):
        MapSplit.__init__(self)
        self.data=data
        self.X=[]
        self.y=[]
        self.grids_POIs=[]
        self.labels=[]
        self.grids_dict_POIs={}
        self.grids_dict_mapped={}
        
    def plot(self,hover_data=["highway", "place","bus_stop"]):
        fig = px.scatter_mapbox(self.data, lat="lat", lon="lon", hover_name="name", hover_data=hover_data,
                        color_discrete_sequence=["fuchsia"], zoom=9, height=1000)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()

    
    
    def split(self,bottomLeft,bottomRigh,topLeft,topRight,num):
        self.split_into_grids(bottomLeft,bottomRigh,topLeft,topRight,num)
        
        if 'lon' in self.data.columns and 'lat' in self.data.columns:
            self.data['col'] = np.searchsorted(self.cols, self.data['lon'])
            self.data['row'] = np.searchsorted(self.rows, self.data['lat'])
        elif 'longitude' in self.data.columns and 'latitude' in self.data.columns:
            self.data['col'] = np.searchsorted(self.cols, self.data['longitude'])
            self.data['row'] = np.searchsorted(self.rows, self.data['latitude'])
            
        
#         grids=[]
        for i in range(len(self.data)):
            self.grids_POIs.append((self.data.iloc[i]['col'],self.data.iloc[i]['row']))
        
        self.data['grids']=self.grids_POIs
        
        key=0
        for g in set(self.grids_POIs):
#             print(g,key)
            self.grids_dict_POIs[g]=key
            key+=1
            
        self.grids_dict_mapped=self.map_grid_nb_to_original_split(self.grids_dict_POIs,self.grids_dict)
        
#         labels=[]
        for i in range(len(self.data)):
            self.labels.append(self.grids_dict_POIs[self.data.iloc[i]['grids']])
        
        self.data['POI_labels']=self.labels
    
    def save_data(self,path):
        self.data.to_csv(path,index=False)
        
    def prepare_data_sentence(self):
        df=self.data
        for i in range(len(df)):
            sample=""
            if (df.iloc[i]['name'] is np.nan) == False:
                sample+=df.iloc[i]['name']

            if (df.iloc[i]['roads'] is np.nan) == False:
                sample+=" "+df.iloc[i]['roads']


            if (df.iloc[i]['village'] is np.nan) == False:
                sample+=" "+df.iloc[i]['village']


            if (df.iloc[i]['municipality'] is np.nan) == False:
                sample+=" "+df.iloc[i]['municipality']

            if (df.iloc[i]['city'] is np.nan) == False:
                sample+=" "+df.iloc[i]['city']

            if (df.iloc[i]['country'] is np.nan) == False:
                sample+=" " +df.iloc[i]['country']

            self.X.append(sample)
            self.y.append(df.iloc[i]['POI_labels'])

        for i in range(len(df)):
            sample=""
            if (df.iloc[i]['name'] is np.nan) == False:
                sample+=df.iloc[i]['name']

            if (df.iloc[i]['city'] is np.nan) == False:
                sample+=" "+df.iloc[i]['city']

            if (df.iloc[i]['country'] is np.nan) == False:
                sample+=" " +df.iloc[i]['country']

            self.X.append(sample)
            self.y.append(df.iloc[i]['POI_labels'])

        for i in range(len(df)):
            sample=""
            if (df.iloc[i]['name'] is np.nan) == False:
                sample+=df.iloc[i]['name']

            self.X.append(sample)
            self.y.append(df.iloc[i]['POI_labels'])
            
    def split_data_into_train_test(self,train_path,test_path,test_size=0.33,random_state=42):
        if len(self.X)==0 and len(self.y)==0:
            self.prepare_data_sentence()
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,shuffle=True,test_size=test_size, random_state=random_state)
        prepared_data_train=pd.DataFrame(zip(X_train,y_train),columns=["X","y"])
        prepared_data_train.to_csv(train_path,index=False)
        prepared_data_test=pd.DataFrame(zip(X_test,y_test),columns=["X","y"])
        prepared_data_test.to_csv(test_path,index=False)


# In[23]:


class TextPreprocessing:
    
    def __init__(self):
        
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer() 

        self.emoticons_str = r"""
            (?:
                [:=;] # Eyes
                [oO\-]? # Nose (optional)
                [D\)\]\(\]/\\OpP] # Mouth
            )"""

        self.urls_str= r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

        self.regex_str = [
            self.emoticons_str,
            r'<[^>]+>', # HTML tags
            r'(?:@[\w_]+)', # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

            r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
            r'(?:[\w_]+)', # other words
            r'(?:\S)' # anything else
        ]

        self.tokens_re = re.compile(r'('+'|'.join(self.regex_str)+')', re.VERBOSE | re.IGNORECASE)
        self.emoticon_re = re.compile(r'^'+self.emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
        self.url_re = re.compile(r'^'+self.urls_str+'$', re.VERBOSE | re.IGNORECASE)
        self.stop_words = set(stopwords.words('english')+stopwords.words('french')+stopwords.words('Greek')+stopwords.words('arabic'))

    def remove_urls(self,string):
        t=[]
        for s in string.split():
            if self.url_re.search(s):
                continue
            else:
                t.append(s)
        return ' '.join(t)

    def tokenize(self,s):
        return self.tokens_re.findall(s)

    def preprocess(self,s, lowercase=False,remove_urlss=False):
        s=self.remove_urls(s)
        tokens = self.tokenize(s)
        stop_words = set(stopwords.words('english')+stopwords.words('french')+stopwords.words('Greek')+stopwords.words('arabic'))
        if lowercase:
            tokens = [token if self.emoticon_re.search(token) else token.lower() for token in tokens if token not in stop_words]

        if remove_urls:
            tokens = ['' if self.url_re.search(token) else token for token in tokens if token not in stop_words]
        tokens=' '.join(tokens)
        return tokens

    def joinTokens(tokens):
        s=''
        for i in range(len(tokens)):
            if i==len(tokens)-1:
                s+=tokens[i]
            else:
                s+=tokens[i]+' '
        return s

    # tweet = geo_tagged_df['text'].iloc[0]
    

    def preprocess(self,sentence):
        sentence=str(sentence)
        sentence = sentence.lower()
        sentence=sentence.replace('{html}',"") 
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', sentence)
        rem_url=re.sub(r'http\S+', '',cleantext)
        rem_num = re.sub('[0-9]+', '', rem_url)
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(rem_num)  
        filtered_words = [w for w in tokens if len(w) > 2 if not w in self.stop_words]
        stem_words=[self.stemmer.stem(w) for w in filtered_words]
        lemma_words=[self.lemmatizer.lemmatize(w) for w in stem_words]
        return " ".join(filtered_words)


# In[ ]:





# In[24]:


class TwitterConnection:
    
    def __init__(self,consumer_key,consumer_secret_key):
        self.consumer_key = consumer_key
        self.consumer_secret_key = consumer_secret_key
        #Reformat the keys and encode them
        self.key_secret = '{}:{}'.format(self.consumer_key, self.consumer_secret_key).encode('ascii')
        #Transform from bytes to bytes that can be printed
        self.b64_encoded_key = base64.b64encode(self.key_secret)
        #Transform from bytes back into Unicode
        self.b64_encoded_key = self.b64_encoded_key.decode('ascii')
        
        self.base_url = 'https://api.twitter.com/'
        self.auth_url = '{}oauth2/token'.format(self.base_url)
        self.auth_headers = {
            'Authorization': 'Basic {}'.format(self.b64_encoded_key),
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }
        
        self.auth_data = {
            'grant_type': 'client_credentials'
        }
        
        self.auth_resp = requests.post(self.auth_url, headers=self.auth_headers, data=self.auth_data)
        print(self.auth_resp.status_code)
        self.access_token = self.auth_resp.json()['access_token']

        
class TwitterQueries(TwitterConnection):
    
    def __init__(self,consumer_key,consumer_secret_key):
        TwitterConnection.__init__(self,consumer_key=consumer_key,consumer_secret_key=consumer_secret_key)
        self.geo_headers = {
            'Authorization': 'Bearer {}'.format(self.access_token)    
        }
        
    
    def get_place_info(self,place_id):
        geo_params = {
            'place_id': place_id
        }

        geo_url = 'https://api.twitter.com/1.1/geo/id/'+geo_params['place_id']+'.json'  
        geo_resp = requests.get(geo_url, headers=self.geo_headers, params=geo_params)
        
        geo_data = geo_resp.json()
        return geo_data





# In[25]:


class DataOperations:
    
        
    def splitting_by_day(self,df,value='created_at'):
        df=df.sort_values(by=[value])
        dfs=[]
        start=0
        end=1
        c=0
        for i in range(len(df)):
            if df.shift(-1).iloc[i][value].date()==df.iloc[i][value].date():
                end+=1

            else:
                c+=1
                dfs.append(df.iloc[start:end])
                if i==0:
                    start=1
                else:
                    start=end
                end+=1

        if start<len(df):
            dfs.append(df[start:-1])
        return dfs
    
    def most_frequent(self,List):
        counter = 0
        num = List[0]

        for i in List:
            curr_frequency = List.count(i)
            if(curr_frequency> counter):
                counter = curr_frequency
                num = i

        return num

    


# In[26]:



class TweetsOperations(MapSplit,TextPreprocessing,DataOperations):
    
    def __init__(self,data,NERModel="xlm-roberta-large-finetuned-conll03-english",EmbModel='sentence-transformers/paraphrase-xlm-r-multilingual-v1'):
        MapSplit.__init__(self)
        TextPreprocessing.__init__(self)
        DataOperations.__init__(self)
        self.data=data
        self.grids_tweets=[]
        self.grids_dict_tweets={}
        self.grids_dict_mapped={}
        self.entities=[]

        self.tokenizer_ner = AutoTokenizer.from_pretrained(NERModel)
        self.model_ner = AutoModelForTokenClassification.from_pretrained(NERModel)
        self.nlp = pipeline('ner', model=self.model_ner, tokenizer=self.tokenizer_ner, aggregation_strategy="simple")
        self.model_embedding = SentenceTransformer(EmbModel)
        self.locations=[]
        self.persons=[]
        self.misc=[]
        self.organizations=[]
        self.lat=[]
        self.long=[]
        self.labels=[]

        
    def preprocessText(self,columnName='text'):
        self.data['preprocessed_text']=self.data[columnName].map(self.preprocess)
        
    def split(self,bottomLeft,bottomRigh,topLeft,topRight,num):
        self.split_into_grids(bottomLeft,bottomRigh,topLeft,topRight,num)
            
        self.data['col'] = np.searchsorted(self.cols, self.data['longitude'])
        self.data['row'] = np.searchsorted(self.rows, self.data['latitude'])
        
#         grids=[]
        for i in range(len(self.data)):
            self.grids_tweets.append((self.data.iloc[i]['col'],self.data.iloc[i]['row']))
        
        self.data['grids']=self.grids_tweets
        
        key=0
        for g in set(self.grids_tweets):
#             print(g,key)
            if g==(num,num) or g==(0,num) or g==(num,0) or g==(0,0):
                self.grids_dict_tweets[g]=np.nan
            else:
                self.grids_dict_tweets[g]=key
                key+=1
            
        self.grids_dict_mapped=self.map_grid_nb_to_original_split(self.grids_dict_tweets,self.grids_dict)
        
#         labels=[]
        for i in range(len(self.data)):
            self.labels.append(self.grids_dict_tweets[self.data.iloc[i]['grids']])
        
        self.data['tweet_labels_'+str(num)]=self.labels
        
    def calculate_and_save_embedding(self,path,column='preprocessed_text',columnName='text'):
        if 'preprocessed_text' not in self.data.columns and column == 'preprocessed_text':
            self.preprocessText(columnName=columnName)
        
        sentences=self.data.dropna(subset=[column])[column].to_list()    
        emb_sentences=self.model_embedding.encode(sentences,convert_to_numpy=True)
        self.data['embeddings']=list(emb_sentences)
        np.save(path,emb_sentences)
        
    def applyNER(self,columnName='preprocessed_text'):
        if 'preprocessed_text' not in self.data.columns and columnName == 'preprocessed_text':
            self.preprocessText()
        for i in range(len(self.data)):
            self.entities.append(self.nlp(self.data[columnName].iloc[i]))
        
        self.data['entities']=self.entities
        self.locations,self.persons,self.misc,self.organizations=self.extract_entities()
        
        self.data['entity_locations']=self.locations
        self.data['entity_persons']=self.persons
        self.data['entity_organization']=self.organizations
        self.data['entity_MISC']=self.misc
        identified_locations,only_locations=self.get_locations()
        self.data['identified_locations']=identified_locations
        self.data['only_locations']=only_locations
    
    def save(self,path):
        self.data.to_csv(path,index=False)
    
    def extract_entities(self):
        loc=[]
        per=[]
        misc=[]
        org=[]
        for entity in self.entities:
            #make for each LOC ,PER, MISC, ORG a list and append to it the captured antities 
            l=[]
            p=[]
            m=[]
            o=[]
            for e in entity:
                if e['entity_group']=='LOC':
                    l.append(e['word'])
                elif e['entity_group']=='PER':
                    p.append(e['word'])
                elif e['entity_group']=='MISC':
                    m.append(e['word'])
                elif e['entity_group']=='ORG':
                    o.append(e['word'])
                else:
                    print(e['entity_group'],e['word'])
                    l.append(np.nan)
                    p.append(np.nan)
                    m.append(np.nan)
                    o.append(np.nan)

            if len(l)==0:
        #         l.append(np.nan)
                l=np.nan
            if len(p)==0:
        #         p.append(np.nan)
                p=np.nan
            if len(o)==0:
        #         o.append(np.nan)
                o=np.nan
            if len(m)==0:
        #         m.append(np.nan)
                m=np.nan


            loc.append(l)
            per.append(p)
            misc.append(m)
            org.append(o)
            
        return loc,per,misc,org
    
    def is_nan(self,x):
        return (x != x)
    def get_locations(self):
        identified_locations=[]
        df=self.data
        for i in range(len(self.data)):
            s=""
            l=""
            o=""
            if not self.is_nan(df.iloc[i]['entity_organization']):
                o=" ".join(df.iloc[i]['entity_organization'])

            if not self.is_nan(df.iloc[i]['entity_locations']):
                l=" ".join(df.iloc[i]['entity_locations'])
        #     print("l",l)

            if o=="" and l=="":
                s=np.nan
            else:                
                s=o+" "+l

            identified_locations.append(s)

        only_locations=[]
        for i in range(len(df)):
            s=""
            if not self.is_nan(df.iloc[i]['entity_locations']) and df.iloc[i]['entity_locations']!="['']":
                s=" ".join(df.iloc[i]['entity_locations'])
            else:                
                s=np.nan

            only_locations.append(s)
            

        return identified_locations,only_locations
    

    def find_long_lat(self):
        for i in range(len(self.data)):
            if 'geo' in self.data.columns and self.data['geo'].iloc[i] is not np.nan:
                if type(self.data['geo'].iloc[i])=='str':
                    if 'coordinates' in literal_eval(self.data['geo'].iloc[i]).keys():
                        self.long.append(literal_eval(self.data['geo'].iloc[i])['coordinates']['coordinates'][0])
                        self.lat.append(literal_eval(self.data['geo'].iloc[i])['coordinates']['coordinates'][1])
                    else:
                        self.lat.append(np.nan)
                        self.long.append(np.nan)            
                else:
                    if 'coordinates' in self.data['geo'].iloc[i].keys():
                        self.long.append(self.data['geo'].iloc[i]['coordinates']['coordinates'][0])
                        self.lat.append(self.data['geo'].iloc[i]['coordinates']['coordinates'][1])
                    else:
                        self.lat.append(np.nan)
                        self.long.append(np.nan)
            else:
                self.lat.append(np.nan)
                self.long.append(np.nan)
            
        self.data['latitude']=self.lat
        self.data['longitude']=self.long
        
    def find_sentence_similarity(self,only_with_labels=False,label_name='tweet_labels_10',embedding_name='embeddings',predicted_label_name='predicted_labels_10',probability_label_name='probability_labels_10'):
        if only_with_labels:
            df=self.data.dropna(subset=[label_name])
        else:
            df=self.data
        
        df['created_at']=pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
        self.dfs=self.splitting_by_day(df)
        for dff in self.dfs:
            l,p=self.get_label_semantic_coherence(dff,tweet_label=label_name,embedding_name=embedding_name)
            
            labels=[]
            probabilty=[]
            for key,value in l.items():
                if value is np.nan:
                    labels.append(np.nan)
                else:
                    labels.append(value[label_name])
            
            for key,value in p.items():
                if value is np.nan:
                    probabilty.append(np.nan)
                else:
                    probabilty.append(value['probability_'+label_name])
                
            dff[predicted_label_name]=labels
            dff[probability_label_name]=probabilty
        
        self.merged_df=pd.concat(self.dfs)
        
    
    def get_label_semantic_coherence(self,df,tweet_label='tweet_labels_10',embedding_name='embeddings'):
#         df=self.data
        all_labels={}
        all_probabilities={}
        for k in range(len(df)):
            labels={}
            probability={}

            cosine_scores= util.pytorch_cos_sim(df[embedding_name].to_list()[k],df[embedding_name].to_list())
            points=[(j,cosine_scores[i][j]) for i in range(len(cosine_scores)) for j in range(len(cosine_scores[i])) if cosine_scores[i][j]>=0.6 and j!=k]

            if len(points)>0:
                label=[]
                probability_=[]

                if len(points)==2:                
                    if points[0][1]>points[1][1]: 
                        points=[points[0]]
                    else:
                        points=[points[1]]


                for point,precision in points:
                    label.append(df.iloc[point][tweet_label])
                    
                prediction=self.most_frequent(label)
                
                for point,precision in points:
                    if df.iloc[point][tweet_label] == prediction:
                        probability_.append(precision)

                labels[tweet_label]=prediction
                
                probability['probability_'+tweet_label]=np.mean(probability_)
                
                all_labels[k]=labels
                all_probabilities[k]=probability
            else:
                all_labels[k]=np.nan
                all_probabilities[k]=np.nan

        return all_labels,all_probabilities


# In[ ]:




class MultiViewLearner:
    def __init__(self,name,path_to_POIs,path_to_tweets,test_size=0.33,random_state=42,columns=['original_predicted_label_5','original_detected_label_5','probability_labels_5','scores','original_tweet_label_5']):
        self.tweets_prediction=pd.read_csv(path_to_tweets)
        self.POIS_prediction=pd.read_csv(path_to_POIs)
        self.df=pd.merge(self.tweets_prediction,self.POIS_prediction,on=["id"])
        self.X,self.y=self.prepare_D_prime_dataset(columns=columns)
        self.X_train,self.X_test,self.y_train,self.y_test=self.split_train_test(test_size=test_size,random_state=random_state)
        self.name=name
        
    def prepare_D_prime_dataset(self,columns=['original_predicted_label_5','original_detected_label_5','probability_labels_5','scores','original_tweet_label_5']):
        X=[]
        y=[]
        for i in range(len(self.df)):
            X.append([self.df.iloc[i][columns[0]],self.df.iloc[i][columns[1]],self.df.iloc[i][columns[2]],self.df.iloc[i][columns[3]]])
            y.append(self.df.iloc[i][columns[4]])
        return X,y

    def split_train_test(self,test_size=0.33,random_state=42):
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,test_size=test_size,random_state=random_state)
        return X_train,X_test,y_train,y_test

    def train_and_validate(self,saving_path='./models/',save_model=False):
        self.clf = RandomForestClassifier()
        self.clf.fit(self.X_train, self.y_train)
        y_test_pred = self.clf.predict(self.X_test)
        ac=accuracy_score(y_true=self.y_test, y_pred=y_test_pred)

        print("the accuracy score of the testing data is : " + str(ac))
        if save_model==True:
            filename = saving_path+str(self.name)+'_model.sav'
            pickle.dump(self.clf, open(filename, 'wb'))
            
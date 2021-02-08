## Module containing funtions for sorting locations into buckets
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def TTWA_county_feature(df_train,df_TTWA,raw):
    
    TTWA_names = df_TTWA['TTWA Name']
    TTWA_names = TTWA_names.dropna()
    
    if raw:
        raw_location = df_train.LocationRaw
    else:
        raw_location = df_train.LocationNormalized
    
    # Get indices for TTWA
    indices = get_TTWA(TTWA_names, raw_location)
    
    df_train['TTWA'] = np.nan
    
    for i in range(0,len(TTWA_names)):
        df_train.TTWA.loc[indices[i]] = TTWA_names.iloc[i]
    
    # Now get indices for Counties
    url = 'https://raw.githubusercontent.com/andreafalzetti/uk-counties-list/master/uk-counties/uk-counties-list.csv'
    counties = pd.read_csv(url, header = None)
    counties.columns = ['Country', 'County']
    
    indices = get_TTWA(counties.County, raw_location)
    
    df_train['County'] = np.nan
    for i in range(0,len(counties)):
        df_train.County.loc[indices[i]] = counties.County.iloc[i]
        
    # Create one feature for TTWA and county. First TTWA, if no TTWA then fill in county
    df_Loc = df_train[['TTWA','County']]
    df_Loc['TTWA_County'] = df_Loc.TTWA
    df_Loc.TTWA_County[df_Loc.TTWA_County.isna()] = df_Loc.County[df_Loc.TTWA_County.isna()] 
    
    return df_Loc

# Function to find TTWA_names in raw locations
def get_TTWA(TTWA_names, raw_location):
    
    indices = []
    for i in range(0,len(TTWA_names)):
        indices.append(raw_location[raw_location.str.contains(TTWA_names.iloc[i])].index)
        
    return indices

#remove stop words
def remove_stop_words(bag, unique):
    stop_words = stopwords.words('english')
    stop_words.append('k')
    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(bag)
    
    words = []
    for word in word_tokens:
        words.append(word.lower())
    # Get unique words only
    if unique:
        words_set = set(words)
        words = list(words_set)
    
    no_stop_words = [w for w in words if not w in stop_words] 
    
    return no_stop_words


def lemmatize_words(no_stop_words):
    
    lemma = WordNetLemmatizer() 
    lemma_words = []

    for w in no_stop_words:
        lemma_words.append(lemma.lemmatize(w))
    
    return lemma_words

def get_max_length(X_train, X_test, X_val):
    lengths =[]
    for x in X_test:
        lengths.append(len(x))
    for x in X_train:
        lengths.append(len(x))
    for x in X_val:
        lengths.append(len(x))

    max_length = max(lengths)
    
    return max_length
## Module containing funtions for sorting locations into buckets
import numpy as np
import pandas as pd

def simple(x,y):
    return x+y

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
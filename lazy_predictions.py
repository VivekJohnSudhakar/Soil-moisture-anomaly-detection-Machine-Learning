#!/home/vsudhakar/project1/ncics/env/bin/python
import pandas as pd
import numpy as np
import pandas as pd
import  lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

def make_boolean(x):
    if x <500.0:
        return 1
    elif x > 500.0:
        return 0

def update_flag(row):
    if row['SPIKE'] == 1:
        return 1
    elif row['NOISE'] == 1 or row['DIURNALNOISE'] == 1 or row['FROZENRECOVERY']==1:
        return 2
    else:
        return 0

if __name__ == '__main__':
#print("hello world")
	df = pd.read_csv("/store/ronnieleeper/activeProjects/soilQC/SoilMoistureQC/data/acclima_soil_water-full_with_tipping_bucket_and_wetness-20230113.csv",dtype={'PRECIPITATION': 'float64'})
# Add column to easily detect normal / flagged data
	df['FLAG'] = df.NOPRCPRESPONSE + df.FROZENRECOVERY+ df.NOISE + df.FAILURE + df.STATIC+df.ERRATIC+ df.DIURNALNOISE+df.TOOHIGH+df.SCALING + df.ZERO+df.SPIKE
	df['UTC_START'] = pd.to_datetime(df['UTC_START'])
	df['PRECIPITATION'] = df['PRECIPITATION'].fillna(value= 0.0)
	df['PRECIPITATION'] = df['WETNESS_1'].apply(lambda x: make_boolean(x))
	df_modified = df.drop(['NOPRCPRESPONSE','ZERO','TOOHIGH','ERRATIC','FAILURE','SCALING','FLAG','TIPPING_BUCKET','WETNESS_2'],axis=1)
	df_modified['FLAG'] = df_modified.apply(lambda x: update_flag(x), axis=1)
	df_modified.drop(['FROZENRECOVERY','DIURNALNOISE','NOISE','SPIKE'],axis=1,inplace=True)
	df_modified.drop('WBAN',axis=1,inplace=True)
	df_modified.drop('WETNESS_1',axis=1,inplace=True)
	df_modified.drop('STATIC',axis=1,inplace=True)
	df_modified[['TEMPERATURE','VOLUMETRIC']] = df_modified[['TEMPERATURE','VOLUMETRIC','STATION_ID']].groupby('STATION_ID').transform(lambda x: (x - x.mean()) / x.std())
	df_modified.drop('UTC_START',axis=1,inplace=True)
	X = df_modified.drop(['FLAG'],axis=1)
	y = df_modified['FLAG']
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =42)
	clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
	models,predictions = clf.fit(X_train, X_test, y_train, y_test)
	print(models)

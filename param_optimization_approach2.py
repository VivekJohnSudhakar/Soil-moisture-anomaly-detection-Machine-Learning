#!/home/vsudhakar/project1/ncics/env/bin/python
import pandas as pd
import numpy as np
import pandas as pd
import  lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight

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
	#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =42)
	models = []
	accuracies =[]
	f_score_macro = []
	lbl = preprocessing.LabelEncoder()
	X['VARIABLE'] = lbl.fit_transform(X['VARIABLE'].astype(str))
	x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0,stratify= y)   
	parameters = {
              'learning_rate': [0.5,0.6, 0.65, 0.7, 0.8], 
              'max_depth': [ 25,30,40,55],
              'n_estimators': [300,500,750,1250, 1500, 1750],
              "reg_alpha"   : [0.3,0.4,0.5,0.6],
              "reg_lambda"  : [2],
              "gamma"       : [0.1,0.5,1]}
	xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
	grid_obj_xgb = RandomizedSearchCV(xgb_model,parameters, cv=5,n_iter=40,verbose=3,n_jobs=30)
	classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
	search = grid_obj_xgb.fit(x_train, y_train,sample_weight=classes_weights,verbose=4)
	print("best params: ",search.best_params_)
	best_model = search.best_estimator_
	y_pred = best_model.predict(x_test)
	print("F1 score macro:" , f1_score(y_test, y_pred, average='macro'))
	print("F1 score micro:" , f1_score(y_test, y_pred, average='micro'))
	print("F1 score none:" , f1_score(y_test, y_pred, average=None))
	labels = search.classes_
	conf_df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=labels, index=labels)
	conf_df.index.name = 'True labels'
	print(conf_df)

		
	#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
	#models,predictions = clf.fit(X_train, X_test, y_train, y_test)
	#print(models)

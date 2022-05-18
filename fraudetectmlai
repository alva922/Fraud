#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
%matplotlib inline
warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
from datetime import datetime, date
import math
from math import radians, sin, cos, acos, atan2
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
!pip install mpu --user
!pip install imbalanced-learn
#read train and test data from yourpath directory
data_train = pd.read_csv(r'yourpath\fraudTrain.csv')
data_test = pd.read_csv(r'yourpath\fraudTest.csv')
# Concatenation train+test dataset
data = pd.concat([data_train, data_test])
#drop unwanted columns
cols_to_delete = ['Unnamed: 0', 'cc_num', 'street', 'zip', 'trans_num', 'unix_time' ]
data.drop(cols_to_delete, axis = 1, inplace = True)
# Create a column customer name with first and last columns 
data['Customer_name'] = data['first']+" "+data['last']
data.drop(['first','last'], axis=1, inplace=True)
#Create a categorical column Population_group by binning the variable city_pop
data["Population_group"] = pd.cut(data["city_pop"], bins=list(range(0,3000001,500000)), labels = ["<5lac","5-10lac","10-15lac","15-20","20-25lac","25-30lac"])
data["Population_group"].value_counts()
#create a column age from dob variable
data['dob'] = pd.to_datetime(data['dob'])
def calculate_age(born):
today = date.today()
return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
data['age'] = data["dob"].apply(calculate_age)
#create a column age_group from the column age
data["age_group"] = pd.cut(data["age"], bins=[0,25,40,60,80,9999], labels = ["<25","25-40","40-60","60-80","80+"])
#distance between the customer and merchant location
R = 6373.0 # radius of the Earth
data['lat'] = data['lat'].astype('float')
data['long'] = data['long'].astype('float')
data['merch_lat'] = data['merch_lat'].astype('float')
data['merch_long'] = data['merch_long'].astype('float')#coordinates
data['lat'] = np.radians(data['lat'])
data['long'] = np.radians(data['long'])
data['merch_lat'] = np.radians(data['merch_lat'])
data['merch_long'] = np.radians(data['merch_long'])
data['dlon'] = data['merch_long'] - data['long'] #change in coordinates
data['dlat'] = data['merch_lat'] - data['lat']
a = np.sin(data['dlat'] / 2)**2 + np.cos(data['lat']) * np.cos(data['merch_lat']) * np.sin(data['dlon'] / 2)**2 #Haversine formula
c = 22np.arctan2(np.sqrt(a), np.sqrt(1 - a))
data['distance'] = R * c
#create a column dist_range_km from the column distance
data["dist_range_km"] = pd.cut(data["distance"], bins=[0,25,50,100,150,200,250,300,9999], labels = ["<25","25-50","50-100","100-150","150-200","200-250","250-300","300+"])
data.drop(['dlat', 'dlon'], axis=1, inplace=True)
data.drop(['dob','city_pop'], axis=1, inplace=True)
#create the transaction date and time column
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
#Extract year and month from trans_date_trans_time column
data['year'] = pd.DatetimeIndex(data['trans_date_trans_time']).year
data['month'] = pd.DatetimeIndex(data['trans_date_trans_time']).month
#Extract day of the week and transaction hour from trans_date_trans_time column
data['day_of_week'] = data['trans_date_trans_time'].dt.day_name()
data['transaction_hour'] = data['trans_date_trans_time'].dt.hour
#Feature Engineering
#drop unwanted columns
data.drop(["merchant", "city", "job"], axis=1, inplace=True)
#perform train-test data split
train,test = train_test_split(data,test_size=0.3,random_state=42, stratify=data.is_fraud)
y_train = train.pop("is_fraud")
X_train = train
y_test = test.pop("is_fraud")
X_test = test
X_train['transaction_hour']= X_train['transaction_hour'].astype(str)
X_train['month']= X_train['month'].astype(str)
X_test['transaction_hour']= X_test['transaction_hour'].astype(str)
X_test['month']= X_test['month'].astype(str)
cat_cols = ["category", "state", "month", "day_of_week", "transaction_hour", 'gender', 'Population_group','age_group', 'dist_range_km']
dummy = pd.get_dummies(X_train[cat_cols], drop_first=True)
X_train = pd.concat([X_train, dummy], axis=1)
X_train.drop(cat_cols, axis=1, inplace=True)
X_train.drop(['age','distance'], axis=1, inplace=True)
#scale the numerical variables of train data
scaler = MinMaxScaler()
scale_var = ["amt"]
X_train[scale_var] = scaler.fit_transform(X_train[scale_var])
dummy1 = pd.get_dummies(X_test[cat_cols], drop_first=True)
X_test = pd.concat([X_test, dummy1], axis=1)
X_test.drop(cat_cols, axis=1, inplace=True)
X_test.drop(['age','distance'], axis=1, inplace=True)
X_test[scale_var] = scaler.transform(X_test[scale_var]) #applying scaler transform
#check train data heatmap for correlation
plt.figure(figsize=(20,20))
sns.heatmap(X_train.corr())
plt.show()
#feature selection process by running random forest
rf = RandomForestClassifier(n_estimators = 25).fit(X_train, y_train)
imp_df = pd.DataFrame({
"Varname": X_train.columns,
"Imp": rf.feature_importances_
})
cols_for_model = ['amt', 'category_grocery_pos', 'transaction_hour_22', 'transaction_hour_23', 'category_gas_transport',
'age_group_60-80', 'gender_M', 'age_group_25-40', 'age_group_40-60', 'category_misc_net', 'dist_range_km_150-200',
'category_misc_pos', 'category_shopping_net', 'dist_range_km_100-150', 'day_of_week_Sunday', 'dist_range_km_200-250',
'category_shopping_pos', 'age_group_80+', 'day_of_week_Saturday']
X_train = X_train[cols_for_model]
X_test = X_test[cols_for_model]
#apply the SMOTE resampling
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=45, k_neighbors=5)
X_resampled_smt, y_resampled_smt = smt.fit_resample(X_train, y_train)
from imblearn.over_sampling import RandomOverSampler
over_sample = RandomOverSampler(sampling_strategy = 1)
X_resampled_os, y_resampled_os = over_sample.fit_resample(X_train, y_train)
len(X_resampled_os)
from xgboost import XGBClassifier
xgb_os = XGBClassifier()
xgb_os.fit(X_resampled_os, y_resampled_os)
y_pred_xgb_os = xgb_os.predict(X_test)

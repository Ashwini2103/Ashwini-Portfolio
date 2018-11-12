# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:57:14 2018

@author: Dad
"""

## Airbnb New Booking Dataset
''' Prediction of new Airbnb users booking their destination country.
Importing and Reading csv(.csv)files initally.
'''
import pandas as pd
files=['age_gender_bkts.csv','countries.csv','sessions.csv','train_users_2.csv','test_users.csv']
#%%
## Placing files into dictionary data
data={}
for i in files:
    d=pd.read_csv(r'E:\Airbnb Dataset\{0}'.format(i))
    data[i.replace('.csv','')]=d
#%%
## Displaying contents present in dictionary
for k,v in data.items():
    print('\n'+k+'\n') 
    print(v.head())
#%%
## Checking datatypes of key 'countries' in dictionary
data['countries'].dtypes
#%%
## Displaying contents in countries
data['countries'].head()
#%%
## Checking datatypes of key 'age_gender_bkts' in dictionary
data['age_gender_bkts'].dtypes
#%%
## Displaying contents in age_gender_bkts
data['age_gender_bkts'].head()
#%%
## Converting column 'year' in 'age_gender_bkts' to datatype 'int'
data['age_gender_bkts']['year']=data['age_gender_bkts']['year'].astype(int)
#%%
## Checking datatypes of key 'age_gender_bkts' in dictionary
data['age_gender_bkts'].dtypes
#%%
## Checking datatypes of key 'train_users_2' in dictionary
data['train_users_2'].dtypes
#%%
## Displaying contents in train_users_2
data['train_users_2'].head()
#%%
## Counting user_id's in 'train_users_2'
data['train_users_2']['id'].value_counts()
#%%
## Checking datatypes of key 'sessions' in dictionary
data['sessions'].dtypes
#%%
## Displaying contents in sessions
data['sessions'].head()
#%%
## Counting user_id's in 'sessions' 
data['sessions']['user_id'].value_counts()
#%%
## Creating a new dataframe 'sessions' and assigning data['sessions'] to it
sessions=data['sessions']
#%%
## Checking for null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
## Dropping rows containing null id's in dataframe 'sessions'
sessions=sessions[sessions.user_id.notnull()]
#%%
## Checking for further null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
## Null values in feature 'action' relate to feature 'action_type'
sessions[sessions.action.isnull()].action_type.value_counts()
#%%
## Changing null value to 'message' in feature 'action'
sessions[sessions.action.isnull()]='message'
#%%
## Removing user_id's having 'message' in dataframe 'sessions'
sessions=sessions[sessions.user_id!='message']
#%%
## Checking for further null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
## Null values in feature 'action_type' relate to feature 'action_detail'
sessions[sessions.action_type.isnull()].action_detail.value_counts()
#%%
''' Filling null values in column 'action_type' and 'action_detail' by using 
groupby 'user_id' and 'action' inside a function min_null_values_action_type_detail
'''
def min_null_values_action_type_detail(df,feature):
    
## Filling common values for each user and action
    new_df=pd.DataFrame(df.groupby(['user_id','action'])[feature].value_counts())
    new_df.rename(columns={feature:'Count'},inplace=True)
    new_df=new_df.reset_index()
    new_df_max=pd.DataFrame(new_df.groupby(['user_id','action'])['Count'].max())
    new_df_max=new_df_max.reset_index()
    ## Merging 2 dataframes
    new_df_max=new_df_max.merge(new_df,on=['user_id','action','Count'])
    del new_df_max['Count']
    ## Merge with main dataframe
    df=df.merge(new_df_max,left_on=['user_id','action'],right_on=['user_id','action'],how='left')
    return df
#%%
## Passing values while calling function
sessions=min_null_values_action_type_detail(sessions,'action_type')
#%%
sessions=min_null_values_action_type_detail(sessions,'action_detail')
#%%
## Replacing null values to cols 'action_type' and 'action_detail'
sessions.loc[sessions.action_type_x.isnull(),'action_type_x']=sessions.action_type_y
#%%
sessions.loc[sessions.action_detail_x.isnull(),'action_detail_x']=sessions.action_detail_y
#%%
## Assigning values of action_type_x and action_detail_x tp features action_type and action_detail
sessions['action_type']=sessions.action_type_x
sessions['action_detail']=sessions.action_detail_x
#%%
## Dropping few cols from Dataframe sessions
sessions=sessions.drop(['action_type_x','action_detail_x','action_type_y','action_detail_y'],axis=1)
#%%
## Checking for further null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
''' Filling null values in column 'action_type' and 'action_detail' by using 
groupby 'action' inside a function min_null_values_action_type_detail
'''
def min_null_values_action_type_detail(df,feature):
    new_df=pd.DataFrame(df.groupby(['action'])[feature].value_counts())
    new_df.rename(columns={feature:'Count'},inplace=True)
    new_df=new_df.reset_index()
    new_df_max=pd.DataFrame(new_df.groupby(['action'])['Count'].max())
    new_df_max=new_df_max.reset_index()
    ## Merging two dataframes
    new_df_max=new_df_max.merge(new_df,on=['action','Count'])
    del new_df_max['Count']
    ## Merge with main dataframe
    df=df.merge(new_df_max,left_on=['action'],right_on=['action'],how='left')
    return df
#%%
## Passing values to the function
sessions=min_null_values_action_type_detail(sessions,'action_type')
#%%
sessions=min_null_values_action_type_detail(sessions,'action_detail')
#%%
## Replacing null values to cols 'action_type' and 'action_detail'
sessions.loc[sessions.action_type_x.isnull(),'action_type_x']=sessions.action_type_y
#%%
sessions.loc[sessions.action_detail_x.isnull(),'action_detail_x']=sessions.action_detail_y
#%%
## Assigning values of action_type_x and action_detail_x to features action_type and action_detail
sessions['action_type']=sessions.action_type_x
sessions['action_detail']=sessions.action_detail_x
#%%
## Dropping few cols from Dataframe sessions
sessions=sessions.drop(['action_type_x','action_detail_x','action_type_y','action_detail_y'],axis=1)
#%%
## Checking for further null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
## We can see 'Track Page View' and 'Lookup' actions having null value in entire dataset
sessions.loc[sessions['action']=='lookup','action_type']='lookup'
sessions.loc[sessions['action']=='lookup','action_detail']='lookup'
#%%
sessions.loc[sessions['action']=='Track Page View','action_type']='Track Page View'
sessions.loc[sessions['action']=='Track Page View','action_detail']='Track Page View'
#%%
## Filling missing values using 'missing' in dataset
sessions.action_type=sessions.action_type.fillna("missing")
sessions.action_detail=sessions.action_detail.fillna("missing")
#%%
## Checking for further null values in features of dataframe 'sessions'
sessions.isnull().sum()
#%%
sessions.dtypes
#%%
## Changing datatype of feature 'secs_elapsed' to 'float'
sessions['secs_elapsed']=sessions['secs_elapsed'].astype(float)
#%%
## Filling null values in column 'secs_elapsed' by grouping 'action'
def sum_values_secs_elapsed(df,feature):
    nw_df=pd.DataFrame(df.groupby(['action'],as_index=False)['secs_elapsed'].median())
    df=df.merge(nw_df,left_on=['action'],right_on=['action'],how='left')
    return df
#%%
## Passing values to the function
sessions=sum_values_secs_elapsed(sessions,'secs_elapsed')
#%%
# Replacing null values using col secs_elapsed_y
sessions.loc[sessions.secs_elapsed_x.isnull(),'secs_elapsed_x']=sessions.secs_elapsed_y
#%%
## Assigning values of secs_elapsed_x to secs_elapsed
sessions['secs_elapsed']=sessions.secs_elapsed_x
#%%
## Dropping cols in sessions
sessions=sessions.drop(['secs_elapsed_x','secs_elapsed_y'],axis=1)
#%%
''' Creating a new dataframe new_session by grouping various cols ['user_id','action','device_type','action_type','action_detail']
and finding mean of 'secs_elapsed'
'''
new_session=pd.DataFrame(sessions.groupby(['user_id','action','device_type','action_type','action_detail'])['secs_elapsed'].mean())
#%%
## Resetting the index
new_session=new_session.reset_index()
#%%
## Counting values in feature 'device_type'
new_session['device_type'].value_counts()
#%%
## Seggregating device_types into respective categories by reducing categories in a list
apple_device=['Mac Desktop','iPhone','iPad Tablet','iPodtouch']
android_device=['Android Phone','Android App Unknown Phone/Tablet','Tablet']
windows_device=['Windows Desktop','Windows Phone']
other_device=['Linux Desktop','Chromebook','Blackberry','Opera Phone']
#%%
## Creating a dictionary by appending values of list
device_types={'apple_device':apple_device,
             'android_device':android_device,
             'windows_device':windows_device,
             'other_device':other_device}
#%%
## Creating columns for key cols and naming in form of 0 and 1
for device in device_types:
    new_session[device]=0
    new_session.loc[new_session.device_type.isin(device_types[device]),device]=1
#%%
## Dropping col 'device_type' from dataframe new_session
new_session=new_session.drop(['device_type'],axis=1)
#%%
## Finding out the duration the time user spends on a website
time_spent=pd.DataFrame(sessions.groupby(['user_id'])['secs_elapsed'].sum())
#%%
## Resetting index in dataframe time_spent
time_spent.reset_index()
#%%
## Merging time_spent dataframe with new_session dataframe
new_session=new_session.merge(time_spent,left_on='user_id',right_on='user_id',how='left')
#%%
new_session['duration']=new_session.secs_elapsed_y
#%%
## Dropping cols in new_session
new_session=new_session.drop(['secs_elapsed_x','secs_elapsed_y'],axis=1)
#%%
## Dropping duplicates in dataframe new_session
new_session=new_session.drop_duplicates()
#%%
## Checking null values in dataframe new_session
new_session.isnull().sum()
#%%
## Merging new_session with dataframe data[train_users_2]
train1=data['train_users_2'].merge(new_session,left_on=data['train_users_2']['id'],right_on=new_session['user_id'],how='inner')
#%%
## User_id not present in data['train_users_2']
train2=data['train_users_2'][train1 !='id']
#%%
## Concatenate 2 dataframes train1 and train2 into train
train=pd.concat([train1,train2])
#%%
## Merging new_session with dataframe data[test_users]
test1=data['test_users'].merge(new_session,left_on=data['test_users']['id'],right_on=new_session['user_id'],how='inner')
#%%
## User_id not present in data['test_users']
test2=data['test_users'][test1!='id']
#%%
test=pd.concat([test1,test2])
#%%
## Combining train and test dataframes together
df=pd.concat([train,test])
#%%
## Dropping user_id and key_0 similar to id
df=df.drop(['key_0','user_id','first_device_type'],axis=1)
#%%
## Checking for null values in dataframe df
df.isnull().sum()
#%%
## Replacing null values by 0 in list cols
cols=['android_device','apple_device','duration','windows_device','other_device']
df[cols]=df[cols].fillna(0)
#%%
## Finding out null values in dataframe df
df.isnull().sum()
#%%
## Replace null values using "missing" in list col1
col1=['action','action_type','action_detail']
df[col1]=df[col1].fillna("missing")
#%%
## Finding out null values in dataframe df
df.isnull().sum()
#%%
## Checking language col being null belong to which country_destination
df[df.language.isnull()].country_destination.value_counts()
#%%
def min_language_null_values(frame,feature):
    n_df=pd.DataFrame(frame.groupby(['country_destination'])['language'].value_counts())
    n_df.rename(columns={'language':'count'},inplace=True)
    n_df=n_df.reset_index()
    n_df_new=pd.DataFrame(n_df.groupby(['country_destination'])['count'].max())
    n_df_new=n_df_new.reset_index()
    ## Merging two dataframes
    n_df_new=n_df_new.merge(n_df,on=['country_destination','count'])
    ## Merging with main dataframe
    frame=frame.merge(n_df_new,left_on=['country_destination'],right_on=['country_destination'],how='left')
    return frame
#%%
## Passing values to function min_language_null_values
df=min_language_null_values(df,'language')
#%%
# Replacing null values using col language_y
df.loc[df.language_x.isnull(),'language_x']=df.language_y
#%%
## Assigning values of col language_x to col language
df['language']=df.language_x
#%%
## Dropping cols from dataframe
df=df.drop(['language_x','language_y','count'],axis=1)
#%%
## Checking for null values in dataframe df
df.isnull().sum()
#%%
## Finding out null values in col first_affiliate_tracked
df[df.first_affiliate_tracked.isnull()]['affiliate_channel'].value_counts()
#%%
## Filling null values in first_affiliate_tracked
def min_first_affiliate_null(frame,feature):
    nf_frame=pd.DataFrame(frame.groupby(['affiliate_channel','affiliate_provider'])['first_affiliate_tracked'].value_counts())
    nf_frame.rename(columns={'first_affiliate_tracked':'count'},inplace=True)
    nf_frame=nf_frame.reset_index()
    ## Finding max value in it
    nf_frame_max=pd.DataFrame(nf_frame.groupby(['affiliate_channel','affiliate_provider'])['count'].max())
    ## Resetting the index of dataframe nf_frame_max
    nf_frame_max.reset_index()
    ## Merging two dataframes
    nf_frame_max=nf_frame_max.merge(nf_frame,on=['affiliate_channel','affiliate_provider','count'])
    ## Merging with main dataframe
    frame=frame.merge(nf_frame_max,left_on=['affiliate_channel','affiliate_provider'],right_on=['affiliate_channel','affiliate_provider'],how='left')
    return frame
#%%
df=min_first_affiliate_null(df,'first_affiliate_tracked')
#%%
## Replacing null values in 'first_affiliate_tracked_x' 
df.loc[df.first_affiliate_tracked_x.isnull(),'first_affiliate_tracked_x']=df.first_affiliate_tracked_y
#%%
## Assigning values of 'first_affiliate_tracked_x' to col 'first_affiliate_tracked'
df['first_affiliate_tracked']=df.first_affiliate_tracked_x
#%%
## Dropping cols from dataframe df
df=df.drop(['first_affiliate_tracked_x','first_affiliate_tracked_y'],axis=1)
#%%
## Checking for null values in dataframe df
df.isnull().sum()
#%%
## Filling null values in col 'date_first_booking'
df['date_first_booking']=df['date_first_booking'].fillna(df['date_first_booking'].mode()[0])
#%%
## Checking for null values in dataframe df
df.isnull().sum()
#%%
## Converting feature 'timestamp_first_active' to 'datetime' datatype
df['timestamp_first_active']=df['timestamp_first_active'].astype(str)
#%%
from datetime import datetime
df['timestamp_first_active']=df['timestamp_first_active'].apply(lambda x:datetime.strptime(x,'%Y%m%d%H%M%S'))
#%%
## Converting timestamp_first_active into year,month and date:
df['timestamp_hour']=df['timestamp_first_active'].map(lambda x: x.hour)
df['timestamp_minute']=df['timestamp_first_active'].map(lambda x: x.minute)
df['timestamp_second']=df['timestamp_first_active'].map(lambda x: x.hour)
df['timestamp_year']=df['timestamp_first_active'].map(lambda x: x.year)
df['timestamp_month']=df['timestamp_first_active'].map(lambda x: x.month)
df['timestamp_day']=df['timestamp_first_active'].map(lambda x: x.day)
df['timestamp_weekday']=df['timestamp_first_active'].map(lambda x: x.weekday())
#%%
## Dropping column 'timestamp_first_active'
df=df.drop(['timestamp_first_active'],axis=1)
#%%
## Converting cols 'date_account_created' and 'date_first_booking' to 'datetime' datatype
df['date_account_created']=df['date_account_created'].astype(str)
df['date_first_booking']=df['date_first_booking'].astype(str)
#%%
df['date_account_created']=df['date_account_created'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
df['date_first_booking']=df['date_first_booking'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
#%%
## Feature 'time_to_first_booking' is a difference between 'date_account_created' and 'date_first_booking'
df['time_to_first_booking']=df['date_account_created']-df['date_first_booking']
#%%
## Univariate analysis of DataFrame df
df['country_destination'].value_counts(normalize=True).plot.bar(title='Destination_Country')
#%%
## Bi-variate analysis 
# 1. Action Feature
Action=pd.crosstab(df['action'],df['country_destination'])
#%%
Action.plot(kind='bar',figsize=(10,10),stacked=True)
#%%
## Using a cut-off value to reduce the categories in feature 'action'
cut_off=1378
other_actions=[]
for action,count in df.action.value_counts().iteritems():
    if count < cut_off:
        other_actions.append(action)
        
df.loc[df.action.isin(other_actions),"action"]="other_action"
#%%
df.action.value_counts()
#%%
# 2. Action_Type Feature
action_type=pd.crosstab(df['action_type'],df['country_destination'])
action_type.plot(kind='bar',figsize=(10,10),stacked=True)
#%%
## Using a cut-off value to reduce the categories in feature 'action_type'
other_action_type=[]
for action_type,count in df.action_type.value_counts().iteritems():
    if count < cut_off:
        other_action_type.append(action_type)
        
df.loc[df.action_type.isin(other_action_type),"action_type"]="other"
#%%
df.action_type.value_counts()
#%%
# 3. Action_Detail Feature
action_detail=pd.crosstab(df['action_detail'],df['country_destination'])
action_detail.plot(kind='bar',figsize=(10,10),stacked=True)
#%%
## Using a cut-off value to reduce the categories in feature 'action_detail'
other_action_detail=[]
for action_detail,count in df.action_detail.value_counts().iteritems():
    if count < cut_off:
        other_action_detail.append(action_detail)
        
df.loc[df.action_detail.isin(other_action_detail),"action_detail"]="other"
#%%
df.action_detail.value_counts()
#%%
# 4. affiliate_channel Feature
affiliate_channel=pd.crosstab(df['affiliate_channel'],df['country_destination'])
affiliate_channel.plot(kind='bar',figsize=(10,10),stacked=True)
#%%
# 5. affiliate_provider Feature
affiliate_provider=pd.crosstab(df['affiliate_provider'],df['country_destination'])
affiliate_provider.plot(kind='bar',figsize=(10,10),stacked=True)
#%%
## Using cut-off value to reduce the categories in feature 'affiliate_provider'
other_affiliate_providers=[]
for affiliate_provider,count in df.affiliate_provider.value_counts().iteritems():
    if count < cut_off:
        other_affiliate_providers.append(affiliate_provider)

df.loc[df.affiliate_provider.isin(other_affiliate_providers),"affiliate_provider"]="other"
#%%
df.affiliate_provider.value_counts()
#%%
# 6.first_browser Feature
first_browser=pd.crosstab(df['first_browser'],df['country_destination'])
#%%
## Create a new feature for mobile browsers
mobile_browsers=['Mobile Safari','Chrome Mobile','Android Browser','Mobile Firefox','IE Mobile']
df.loc[df.first_browser.isin(mobile_browsers),"first_browser"]="Mobile"
#%%
## Categorizing other_browsers using cut_off
other_browsers=[]
for browser,count in df.first_browser.value_counts().iteritems():
    if count < cut_off:
        other_browsers.append(browser)
        
df.loc[df.first_browser.isin(other_browsers),"first_browser"]="Other"
#%%
df.first_browser.value_counts()
#%%
# 7. gender Feature
gender=pd.crosstab(df['gender'],df['country_destination'])
#%%
## 8. signup_app Feature
signup_app=pd.crosstab(df['signup_app'],df['country_destination'])
#%%
## 9. signup_method Feature
signup_method=pd.crosstab(df['signup_method'],df['country_destination'])
#%%
## 10.language Feature
language=pd.crosstab(df['language'],df['country_destination'])
#%%
## Categorizing language into English , Non-English
other_lang=[]
for language,count in df.language.value_counts().iteritems():
    if count < 275:
        other_lang.append(language)

df.loc[df.language.isin(other_lang),"language"]="other"

#%%
df['language']=df.language.map(lambda x: 0 if x == 'en' else 1)
#%%
df['language'].value_counts()
#%%
## signup_flow field page from which the user choose to signup 0----> Home Page
df['signup_flow_simple']=df['signup_flow'].map(lambda x: 0 if x == 0 else 1)
#%%
## Analysis of feature 'first_booking'
df['year_first_booking']=df.date_first_booking.dt.year
df['month_first_booking']=df.date_first_booking.dt.month
df['weekday_first_booking']=df.date_first_booking.dt.weekday
#%%
## Value count in first_booking(yearly,monthly and weekday)
df['year_first_booking'].value_counts()
#%%
df['month_first_booking'].value_counts()
#%%
df['weekday_first_booking'].value_counts()
#%%
## Analysis of feature 'Account_created'
df['year_acct_created']=df.date_account_created.dt.year
df['month_acct_created']=df.date_account_created.dt.month
df['weekday_acct_created']=df.date_account_created.dt.weekday
#%%
## Value count in 'date_account_created'(yearly,monthly and weekday)
df['year_acct_created'].value_counts()
#%%
df['month_acct_created'].value_counts()
#%%
df['weekday_acct_created'].value_counts()
#%%
## Drop features which do not add more information in model building
df=df.drop(['date_first_booking','date_account_created','timestamp_hour','timestamp_minute','timestamp_second','timestamp_year','timestamp_month','timestamp_day','timestamp_weekday','year_acct_created','month_acct_created','weekday_acct_created','time_to_first_booking','age','signup_flow'],axis=1)
#%%
## Seperate target feature 'country_destination' from other input features
label=pd.DataFrame(df['country_destination'])
#%%
## Drop target feature 'country_destination' from input variables
df=df.drop(['country_destination'],axis=1)
#%%
## Displaying datatypes of df
df.dtypes
#%%
## Splitting data in terms of categorical and continous features
cat_features=[]
cont_features=[]
for feature in df.columns:
    if df[feature].dtype == float or df[feature].dtype == 'int64':
        cont_features.append(feature)
    elif df[feature].dtype == object:
        cat_features.append(feature)
#%%
## Date feature needs to be in categorical list
date_lst=['year_first_booking','month_first_booking','weekday_first_booking']
for feature in date_lst:
    if feature in cont_features:
        cont_features.remove(feature)
        cat_features.remove(feature)
#%%
## Create dummies for categorical features
for feature in cat_features:
    dummies=pd.get_dummies(df[feature],prefix=feature,drop_first=False)
    ## Drop non-important features
    df=df.drop(feature,axis=1)
    df=pd.concat([df,dummies],axis=1)
    print("{} is complete".format(feature))
#%%
#######################################################################
'''           Model Building Phase
'''
#######################################################################
'''Using LDA ( Linear Discriminant Analysis) for reducing features. 
Data fit is restricted to a limit , since it throws an out of memory error for more data.'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=3)
df_lda=lda.fit_transform(df[:650000],label[:650000])
#%%
## Applying K-Fold cross validation technique on data
from sklearn.model_selection import KFold
kf=KFold(n_splits=20,shuffle=True,random_state=42)
for train_index,val_index in kf.split(df_lda):
    x_train,x_val=df_lda[train_index],df_lda[val_index]
    y_train,y_val=label.iloc[train_index],label.iloc[val_index]
#%%
## Model Building using Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr=LogisticRegression(penalty='l2',C=0.01,multi_class='ovr',max_iter=300,solver='lbfgs',n_jobs=-1,random_state=42)
lr.fit(x_train,y_train)
y_lr_pred=lr.predict(x_val)
print("Accuracy score",metrics.accuracy_score(y_val,y_lr_pred))
#%%
# Model Building using Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dtc=DecisionTreeClassifier(max_depth=10,min_samples_split=90,min_samples_leaf=20,random_state=42)
dtc.fit(x_train,y_train)
y_dtc_predict=dtc.predict(x_val)
acc_dtc_score=metrics.accuracy_score(y_val,y_dtc_predict)
acc_dtc_score
#%%
## Model Building using RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
rfc=RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=90,n_jobs=-1,max_features=0.13,random_state=20,verbose=10)
rfc.fit(x_train,y_train)
y_rfc_pred=rfc.predict(x_val)
acc_rfc_score=metrics.accuracy_score(y_val,y_rfc_pred)
acc_rfc_score
    

        
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
import collections


def blight_model():
    
    train_all = pd.read_csv('train.csv',encoding = "ISO-8859-1")
    test_all = pd.read_csv('test.csv',encoding = "ISO-8859-1")
    
    test_id = np.array(test_all['ticket_id'])
    ######################## Preprocessing #######################################
    train1 = train_all[np.isfinite(train_all['compliance'])]
    #train1 = train_all.dropna(subset=['compliance'])

    #drop features that exist only in training set
    train2 = train1.drop(['payment_amount', 'balance_due','payment_date', 'payment_status', 'collection_status','compliance_detail'], axis=1)
    
    ##check if the all the entries of a feature are the same 
    ##e.g. admin_fee, state_fee, country(only USA in test)
    #arr = train2.loc[:,'country'].as_matrix().astype(str)
    #arr_test = test_all.loc[:,'country'].as_matrix().astype(str)
    ##print(arr.isnull().sum(),arr_test.isnull().sum())
    #print(np.intersect1d(arr,arr_test))
    ###check_arr = arr[0]*np.ones(len(train2))
    ###print(np.array_equal(arr,check_arr))
    #print(np.where(arr != arr[0])) # for both int and string data, check if it returns an empty array
    #print(collections.Counter(arr))
    #print(collections.Counter(arr_test))
        
    #drop extra uninformative features from both training and test set
    train = train2.drop(['inspector_name', 'violation_zip_code', 'violation_street_number','violation_street_name','violator_name',\
                    'mailing_address_str_number', 'mailing_address_str_name', 'city',\
                    'state', 'zip_code', 'non_us_str_code', 'country','ticket_issued_date', 'hearing_date',\
                    'admin_fee', 'state_fee', 'clean_up_cost', 'grafitti_status','violation_description'], axis=1)
    test = test_all.drop(['inspector_name', 'violation_zip_code', 'violation_street_number','violation_street_name','violator_name',\
                    'mailing_address_str_number', 'mailing_address_str_name', 'city',\
                    'state', 'zip_code', 'non_us_str_code', 'country','ticket_issued_date', 'hearing_date',\
                    'admin_fee', 'state_fee', 'clean_up_cost', 'grafitti_status','violation_description'], axis=1)
    ################################################################################################
    train = train.set_index('ticket_id')
    test = test.set_index('ticket_id')
    
    #y_train = train.iloc[:,-1]
    #x_train = train.iloc[:,:-1]
    ###################### Further preprocessing ##########################
#    for column_name in ['ticket_issued_date', 'hearing_date']:
#    
#        # test
#        day_time = pd.to_datetime(test[column_name])
#        test.drop(column_name, axis=1, inplace=True)
#        test[column_name+'_month'] = np.array(day_time.dt.month)
#        test[column_name+'_year'] = np.array(day_time.dt.year)
#        #test[column_name+'_day'] = np.array(day_time.dt.day)
#        #test[column_name+'_dayofweek'] = np.array(day_time.dt.dayofweek)
#    
#        # train
#        day_time = pd.to_datetime(train[column_name])
#        train.drop(column_name, axis=1, inplace=True)
#        train[column_name+'_month'] = np.array(day_time.dt.month)
#        train[column_name+'_year'] = np.array(day_time.dt.year)
#        #train[column_name+'_day'] = np.array(day_time.dt.day)
#        #train[column_name+'_dayofweek'] = np.array(day_time.dt.dayofweek)
#    
    ############ MDST starter script ###########################
    # Convert string columns to categorical
    cols = test.select_dtypes(exclude=['float', 'int']).columns
    print(cols)
    len_train = len(train)
    #print(train.loc[:,'violation_code'])
    temp_concat = pd.concat((train[cols], test[cols]), axis=0)
    # Some filtering on violation_code to make it more manageable
    temp_concat['violation_code'] = temp_concat['violation_code'].apply(lambda x: x.split(' ')[0])
    temp_concat['violation_code'] = temp_concat['violation_code'].apply(lambda x: x.split('(')[0])
    temp_concat['violation_code'][temp_concat['violation_code'].apply(lambda x: x.find('-')<=0)] = np.nan
    #print(temp_concat.loc[:,'violation_code'])

    #print(temp_concat)
    # Make all codes with < 10 occurrences null
    #counts = temp_concat['violation_code'].value_counts()
    #temp_concat['violation_code'][temp_concat['violation_code'].isin(counts[counts < 10].index)] = np.nan
    print('check1')
    for column_name in cols:
        #print'Converting to categorical...', column_name, '# variables:', len(temp_concat[column_name].unique())
        dummies = pd.get_dummies(temp_concat[column_name])
        temp_concat[dummies.columns] = dummies
        temp_concat.drop(column_name, axis=1, inplace=True)
        train.drop(column_name, axis=1, inplace=True)
        test.drop(column_name, axis=1, inplace=True)
    print('check2')

    train[temp_concat.columns] = temp_concat.loc[train.index]
    test[temp_concat.columns] = temp_concat.loc[test.index]
    print('here')
    print(train.index)
    features = list( test.columns )
    response = ['compliance']
    #print(features)
    x_train = train[features]
    y_train = np.array(train[response]).ravel()

    #y_train = train.iloc[:,-1]
    #x_train = train.iloc[:,:-1]
    #print(x_train.axes)
    #print(y_train.axes)
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    #x_train[pd.isnull(x_train)] = 0

    print(y_train)
    x_test = test[features]
    #print(x_train)
    #sg = SGDClassifier().fit(x_train,y_train)
    #acc_train = sg.score(x_train, y_train)

    nb = GaussianNB().fit(x_train, y_train)
    acc_train = nb.score(x_train, y_train)
    print(acc_train)
    res = nb.predict_proba(x_test)
    print(len(test_id),len(res[:,1]))
    #df = {"ticket_id":test_id, "compliance":res[:,1]}
    #df = pd.DataFrame(df, columns=["ticket_id", "compliance"])
    #mat = np.vstack((test_id,res[:,1]))
    ser = pd.Series(res[:,1],index=test_id)
    return ser#test_id, res[:,1]
res = blight_model()
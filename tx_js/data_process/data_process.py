import pandas as pd
data_click_train = pd.read_csv('./train_data/click_log.csv')
data_ad_train = pd.read_csv('./train_data/ad.csv')
data_all_train = pd.merge(data_click_train,data_ad_train,on=['creative_id'])
del data_click_train
del data_ad_train
data_click_test = pd.read_csv('./train_data/test_data/click_log.csv')
data_ad_test = pd.read_csv('./train_data/test_data/ad.csv')
data_all_test = pd.merge(data_click_test,data_ad_test,on=['creative_id'])
del data_click_test
del data_ad_test
######得出user时间窗口内的点击次数
for i in range(0,90,7):
    print(i)
    f_sum_click = data_all_train.loc[data_all_train.time>i].groupby('user_id').sum()
    f_sum_click.reset_index(inplace=True)
    temp_data = f_sum_click[['user_id','click_times']]
    temp_data['click_times'+str(90-i)] = temp_data['click_times']
    temp_data.drop(['click_times'],axis=1,inplace=True)
    if i == 0:
        train_data = temp_data
    else:
        train_data = pd.merge(train_data,temp_data,how='left',on='user_id')
for i in range(0,90,7):
    print(i)
    f_sum_click = data_all_test.loc[data_all_test.time>i].groupby('user_id').sum()
    f_sum_click.reset_index(inplace=True)
    temp_data = f_sum_click[['user_id','click_times']]
    temp_data['click_times'+str(90-i)] = temp_data['click_times']
    temp_data.drop(['click_times'],axis=1,inplace=True)
    if i == 0:
        test_data = temp_data
    else:
        test_data = pd.merge(test_data,temp_data,how='left',on='user_id')
######2.点击ad_id	product_id	product_category	advertiser_id	industry，总类的数量,各个类别的数量的方差,天点击次数的方差

for item in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
    print(item)
    a = data_all_train.loc[data_all_train.time > 0, ['user_id', item, 'click_times']].groupby(['user_id'])[item].agg(
        pd.Series.nunique)
    print(a)
    train_data[str(item) + 'click_types' + str(90 - 0)] = a.tolist()
for item in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'time']:
    print(item)
    for i in range(0, 90, 7):
        print(i)
        temp_ = data_all_train.loc[data_all_train.time > i, ['user_id', item, 'click_times']].groupby(
            ['user_id', item]).sum()
        temp_.reset_index(inplace=True)
        temp_ = temp_.groupby('user_id')
        a = temp_['click_times'].std()
        dict_temp = {'user_id': a.index, str(item) + 'click_times_std' + str(90 - i): a.values}
        df_temp = pd.DataFrame(dict_temp)
        train_data = pd.merge(train_data, df_temp, how='left', on='user_id')
for item in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
    print(item)
    a = data_all_test.loc[data_all_test.time > 0, ['user_id', item, 'click_times']].groupby(['user_id'])[item].agg(
        pd.Series.nunique)
    print(a)
    test_data[str(item) + 'click_types' + str(90 - 0)] = a.tolist()

for item in ['ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry', 'time']:
    print(item)
    for i in range(0, 90, 7):
        print(i)
        temp_ = data_all_test.loc[data_all_test.time > i, ['user_id', item, 'click_times']].groupby(
            ['user_id', item]).sum()
        temp_.reset_index(inplace=True)
        temp_ = temp_.groupby('user_id')
        a = temp_['click_times'].std()
        dict_temp = {'user_id': a.index, str(item) + 'click_times_std' + str(90 - i): a.values}
        df_temp = pd.DataFrame(dict_temp)
        test_data = pd.merge(test_data, df_temp, how='left', on='user_id')
# 用户点击product_catgory,industry 的次数
data_all_train = pd.get_dummies(data_all_train, columns=['product_category'])
features = []
for key in ['product_category']:
    for item in data_all_train.columns.tolist():
        if item.startswith(key):
            features.append(item)
print(features)
i = 0
for item in features:
    print(item)
    f_sum_click = data_all_train.loc[data_all_train.time > 0, ['user_id', item, 'click_times']].groupby('user_id').sum()
    f_sum_click.reset_index(inplace=True)
    temp_data = f_sum_click[['user_id', item]]
    temp_data[item + 'click_times' + str(90 - i)] = temp_data[item]
    temp_data.drop([item], axis=1, inplace=True)
    train_data = pd.merge(train_data, temp_data, how='left', on='user_id')

####3.用户点击product_catgory,industry 的次数
data_all_test = pd.get_dummies(data_all_test, columns=['product_category'])
features = []
for key in ['product_category']:
    for item in data_all_test.columns.tolist():
        if item.startswith(key):
            features.append(item)
print(features)
i = 0
for item in features:
    print(item)
    f_sum_click = data_all_test.loc[data_all_test.time > 0, ['user_id', item, 'click_times']].groupby('user_id').sum()
    f_sum_click.reset_index(inplace=True)
    temp_data = f_sum_click[['user_id', item]]
    temp_data[item + 'click_times' + str(90 - i)] = temp_data[item]
    temp_data.drop([item], axis=1, inplace=True)
    test_data = pd.merge(test_data, temp_data, how='left', on='user_id')
#####4.构造点击序列特征
data_all_train = data_all_train.sort_values(by=['user_id','time'])
data_all_test = data_all_test.sort_values(by=['user_id','time'])
train_data = data_all_train['user_id'].drop_duplicates()
for item in ['creative_id','ad_id','product_id','product_category','advertiser_id','industry','time']:
    print(item)
    a = data_all_train.loc[data_all_train.time>0,['user_id',item]].groupby('user_id')[item].apply(list)
    dict_temp = {'user_id':a.index,str(item)+'click_list'+str(90-0):a.values}
    df_temp = pd.DataFrame(dict_temp)
    train_data = pd.merge(train_data,df_temp,how='left',on='user_id')
del data_all_train
test_data = data_all_test['user_id'].drop_duplicates()
for item in ['creative_id','ad_id','product_id','product_category','advertiser_id','industry','time']:
    print(item)
    a = data_all_test.loc[data_all_test.time>0,['user_id',item]].groupby('user_id')[item].apply(list)
    dict_temp = {'user_id':a.index,str(item)+'click_list'+str(90-0):a.values}
    df_temp = pd.DataFrame(dict_temp)
    test_data = pd.merge(test_data,df_temp,how='left',on='user_id')
del data_all_test
data_user = pd.read_csv('./train_data/user.csv')
train_data = pd.merge(train_data,data_user,on=['user_id'])
train_data.to_csv('./true_list_train.csv',index=False)
test_data.to_csv('./true_list_test.csv',index=False)

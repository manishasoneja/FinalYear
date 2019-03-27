
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def divide_to_xy_train(train,test):
    x_train = train.iloc[:,:-1]
    #print(x_train)
    x_test = test.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    y_test = test.iloc[:,-1]
    #print(y_train)
    return x_train,y_train,x_test,y_test


# In[6]:


def get_value_code(x_train):
    cat=['IRC', 'X11', 'Z39_50', 'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois']
    col=x_train[2]
    cat_list=set(col)
    cat = set(cat)
    final=list(cat.union(cat_list))
    print(len(final))
    return final


# In[4]:


#read train data from file into pandas:
def read_data():
    data = pd.read_csv('KDDTrain2.csv',header=None)
    test = pd.read_csv('KDDTest2.csv',header=None) 
    return data,test


# In[15]:


def one_hot_encode_train(x_train,cat):
    x_train=pd.concat([x_train,pd.get_dummies(x_train[1])],axis=1)
    x_train.drop([1],axis=1,inplace=True)
    print(x_train.columns.values)
    xt2 = list(x_train[2])
    t2 = pd.Series(xt2)
    t2 = t2.astype('category',categories=cat)
    dt2 = pd.get_dummies(t2)
    x_train = pd.concat([x_train,dt2],axis=1)
    x_train.drop([2],axis=1,inplace=True)
    x_train = pd.concat([x_train,pd.get_dummies(x_train[3])],axis=1)
    x_train.drop([3],axis=1,inplace=True)
    print(x_train)
    #print(x_train.shape)
    return x_train


# In[21]:


def one_hot_encode_test(x_test,cat):
    t1 = pd.get_dummies(x_test[1])
    x_test = pd.concat([x_test,t1],axis=1)
    x_test.drop([1],axis=1,inplace=True)
    #print(x_test)
    print("length",len(cat))
    xt2 = list(x_test[2])
   # print(xt2)
    t2 = pd.Series(xt2)
    t2 = t2.astype('category',categories=cat)
    dt2 = pd.get_dummies(t2)
    x_test = pd.concat([x_test,dt2],axis=1)
    x_test.drop([2],axis=1,inplace=True)
    #print(x_test.shape)
    #xt3 = x_test.iloc[:,3]
    #print(xt3)
    t3= pd.get_dummies(x_test[3])
    x_test = pd.concat([x_test,t3],axis=1)
    x_test.drop([3],axis=1,inplace=True)
    #print(x_test.shape)
    return x_test
    
#t2 = t2.T.reindex(cat).T.fillna(int(0))
#print(t2.http)


# In[9]:


def select_features(x_train,y_train):
    selected = SelectPercentile(percentile=50)
    selected.fit(x_train,y_train)
    header=x_train.columns.values.tolist()
    x_train_selected=selected.transform(x_train)
    #print('X train shape:',x_train.shape)
    #print('x train selected shape',x_train_selected.shape)
    mask = selected.get_support()
    #print(mask)
    #print(mask.shape)
    plt.matshow(mask.reshape(1,-1),cmap="gray_r")
    dictionary = dict(zip(header,mask))
    #print(dictionary)
    for cat in dictionary:
        if dictionary[cat]==False:
            x_train=x_train.drop([cat],axis=1)
    return dictionary,x_train,y_train


# In[10]:


def select_features_by_model(x_train,y_train):
    select = SelectFromModel(DecisionTreeClassifier(),threshold="median")
    select.fit(x_train,y_train)
    select_m=select.transform(x_train)
    header=x_train.columns.values.tolist()
    print(header)
    print('Selected from model',select_m.shape)
    mask2 = select.get_support()
    #print(len(mask2))
    plt.matshow(mask2.reshape(1,-1),cmap="gray_r")
    dictionary = dict(zip(header,mask2))
    #print(dictionary)
    for cat in dictionary:
        if dictionary[cat]==False:
            x_train=x_train.drop([cat],axis=1)
    print(x_train.shape)
    return dictionary,x_train,y_train


# In[11]:


def prep_test_data(x_test,dictionary):
    for cat in dictionary:
        if dictionary[cat]==False:
            x_test=x_test.drop([cat],axis=1)
    return x_test


# In[12]:


def build_model(x_train,y_train):
    model = tree.DecisionTreeClassifier()
    model = model.fit(x_train,y_train)
    return model


# In[13]:


def test_model(model,x_test):
    y_predict=model.predict(x_test)
    #print(y_predict)
    return y_predict


# In[28]:


def acc(y_test,y_predict):
    target_names=['normal','r2l','probe','dos','u2r']
    
    print(accuracy_score(y_test,y_predict))
    print(classification_report(y_test,y_predict,target_names=target_names))
    print(confusion_matrix(y_test,y_predict))


# In[41]:


def main():
    data,test=read_data()
    data=data.drop([42],axis=1)
    test=test.drop([42],axis=1)
    x_train,y_train,x_test,y_test=divide_to_xy_train(data,test)
    cat=get_value_code(x_train)
    x_train=one_hot_encode_train(x_train,cat)
    dictionary,x_train,y_train=select_features(x_train,y_train)
    #print(x_train.shape)
    x_test = one_hot_encode_test(x_test,cat)
    x_test_h = x_test.columns.values.tolist()
    x_test = prep_test_data(x_test,dictionary)
    """dictionary2,x_train,y_train=select_features_by_model(x_train,y_train)
    x_test = prep_test_data(x_test,dictionary2)"""
    #print("shape ",x_test.shape,x_train.shape)
    model = build_model(x_train,y_train)
    y_predict=test_model(model,x_test)
    #print(y_test)
    acc(y_test,y_predict)
main()


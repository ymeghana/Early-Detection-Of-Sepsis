#!/usr/bin/env python
# coding: utf-8

# In[138]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# In[139]:


dataset = pd.read_csv("C:/Users/hcyen/SCA/sepsis.csv")


# In[140]:


from sklearn.utils import resample
df_majority = dataset[dataset.SepsisLabel==0]
df_minority = dataset[dataset.SepsisLabel==1]


# In[141]:


df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=37945,    # to match majority class
                                 random_state=123) # reproducible results


# In[142]:


df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.SepsisLabel.value_counts()


# In[143]:


plt.pie(df_upsampled['SepsisLabel'].value_counts(), labels=['1','0'], autopct='%1.1f%%', shadow=True)
plt.show()


# In[ ]:





# In[144]:


X =df_upsampled[df_upsampled.columns[0:40]].values


# In[145]:


Y = df_upsampled[df_upsampled.columns[40:]].values


# In[146]:


print("sca dimensions : {}".format(df_upsampled.shape))


# In[147]:


print("sca dimensions : {}".format(X.shape))


# In[148]:


print("sca dimensions : {}".format(Y.shape))


# In[149]:


a = dataset.isnull().sum()


# In[150]:


b = dataset.isna().sum()


# In[151]:


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[152]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print("Training data dimensions :{}".format(X_train.shape))
print("Testing data dimensions :{}".format(X_test.shape))


# In[153]:


# from mlxtend.classifier import MultiLayerPerceptron as MLP
clf = MLPClassifier(
#     hidden_layers=[40], 
#           l2=0.00, 
#           l1=0.0, 
#           epochs=100, 
#           eta=0.05, 
#           momentum=0.1,
#           decrease_const=0.0,
#           minibatches=100, 
#           random_seed=1,
#           print_progress=3
    activation='tanh',
    solver='lbfgs',
    early_stopping=False,
   hidden_layer_sizes=(40,10,10,10,10, 2),
    random_state=1,
    batch_size='auto',
    max_iter=5000,
    learning_rate_init=1e-5,
    tol=1e-4,
)


# In[154]:


clf.fit(X_train, Y_train)


# In[157]:


A=df_upsampled[df_upsampled.columns[0:40]]


# In[159]:


predicted = clf.predict(X_test)
idx = 0
true = 0
false = 0
"""print("\npredicted result - original result")
for i in X_test:
    # Printing the predicted values for each test case
    if(predicted[idx]==0):
        a="NoSepsis"
    else:
        a="Sepsis  "
    if(Y_test[idx]==0):
        b="NoSepsis"
    else:
        b="Sepsis"
    
    print(X[idx],a ,"         ", b)
    # Calculating the number of correct predictions and wrong predictions
    if predicted[idx] == Y_test[idx]:
        true += 1
    else:
        false += 1
    idx += 1"""
# Printing the number of correct predictions
#print(true)
# Printing the number of wrong predictions
#print(false)
# Accuracy is calculated
#accuracy = (true / (true + false)) * 100
#print(accuracy)


# In[ ]:
import pickle
pickle.dump(clf,open('model.pkl','wb'))




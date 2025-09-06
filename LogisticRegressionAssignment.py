#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


df = pd.read_csv('Titanic_train.csv')
df.info()
df.describe() 


# In[8]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe(include = object)


# In[11]:


df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df


# In[12]:


df_encoded = pd.get_dummies(df)
df_encoded


# In[13]:


correlation_matrix = df_encoded.corr()
correlation_matrix


# In[14]:
import matplotlib
matplotlib.use("Agg")   # must come before pyplot/seaborn imports


import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('correlationmap')
st.pyplot(fig)


# In[15]:


df_encoded.isnull().sum()


# In[16]:


mean_age = df['Age'].mean()
mean_age


# In[17]:


df_encoded['Age'] = df_encoded['Age'].fillna(df_encoded['Age'].mean())
df_encoded['Age']


# In[18]:


df_encoded['Age'].isnull().sum()


# In[19]:


print('Unique values in categorical columns')
for col in ['Sex','Embarked']:
    print(f"{col}:{df[col].unique()}")


# In[20]:


numerical_features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
df[numerical_features].describe()


# In[21]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Survived',y='Pclass',data=df,hue='Survived',palette='pastel')
plt.title('Pclass distribution by Survived')
plt.xlabel('Survived(0 = Not Survived,1 = Survived)')
plt.ylabel('Pclass')
plt.grid()
plt.show()


# In[22]:


plt.figure(figsize=(10,5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='pastel')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Passengers')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()


# In[23]:


#box plot for age vs churn
plt.figure(figsize=(10,5))
sns.boxplot(x='Survived',y='Age',data=df,hue='Survived',palette='pastel')
plt.title('Age distribution by Survival status')
plt.xlabel('Survival(0 = Not Survived,1 = Survived)')
plt.ylabel('Age')
plt.grid()
plt.show()


# In[24]:


lower_outliers = df['Age'].quantile(0.05)
higher_outliers = df['Age'].quantile(0.95)
df['Age'] = np.where(df['Age'] < lower_outliers, lower_outliers,
             np.where(df['Age'] > higher_outliers, higher_outliers, df['Age']))


# In[25]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Survived',y='Age',data=df,hue='Survived',palette='pastel')
plt.title('Age distribution by Survival status')
plt.xlabel('Survival(0 = Not Survived,1 = Survived)')
plt.ylabel('Age')
plt.grid()
plt.show()


# In[26]:


mean_age = df['Age'].mean()
std_age = df['Age'].std()
print(mean_age)
print(std_age)


# In[27]:


z_threshold = 3
upper_limit = mean_age + z_threshold * std_age
lower_limit = mean_age - z_threshold * std_age
for i in range(len(df)):
    if df.loc[i, 'Age'] < lower_limit:
        df.loc[i, 'Age'] = lower_limit
    elif df.loc[i, 'Age'] > upper_limit:
        df.loc[i, 'Age'] = upper_limit
plt.figure(figsize=(10,5))
sns.boxplot(x='Survived',y='Age',data=df,hue='Survived',palette='pastel')
plt.title('Age distribution by Survival status')
plt.xlabel('Survival(0 = Not Survived,1 = Survived)')
plt.ylabel('Age')
plt.grid()
plt.show()


# In[28]:


sns.heatmap(pd.crosstab(df['Survived'], df['Pclass'], normalize='columns') * 100)


# In[29]:


df_encoded['Family size'] = df['SibSp'] + df['Parch'] + 1
df_encoded['Family size'].value_counts()
df_encoded.describe()


# In[30]:


def transform_family_size(num):
    if num == 1:
        return 'alone'
    elif num >1 and num <5:
        return 'small'
    else:
        return 'large'


# In[31]:


import pandas as pd
df_encoded['Family_type'] = df_encoded['Family size'].apply(transform_family_size)
df_encoded


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

counts = df_encoded[['Embarked_C', 'Embarked_Q', 'Embarked_S']].sum()

counts.plot(kind='bar')
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.title("Passenger count by Embarked port")
plt.show()


# In[33]:


counts.plot(kind='pie')
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.title("Passenger count by Embarked port")
plt.show()


# In[149]:


df_encoded['Fare'].plot(kind='box')


# In[150]:


df_encoded['individual_fare'] = df_encoded['Fare']/(df_encoded['SibSp'] + df_encoded['Parch'] + 1)
df_encoded


# In[151]:


df_encoded['individual_fare'].plot(kind='box')


# In[153]:


from scipy import stats
import numpy as np

z_scores = stats.zscore(df_encoded['individual_fare'])
mean_fare = df_encoded['individual_fare'].mean()

outliers = (np.abs(z_scores) > 3)
df_encoded.loc[outliers, 'individual_fare'] = mean_fare
df_encoded


# In[167]:


x = df_encoded.drop(['SibSp','Parch','Survived','Family_type'],axis=1)
y=df_encoded['Survived']


# In[168]:


X = pd.get_dummies(x,drop_first=True)
print(X.isnull().sum())


# In[169]:


from sklearn.preprocessing import StandardScaler
X=x
num_cols = ['Pclass', 'Age', 'individual_fare', 'Family size']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])


# In[170]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[171]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(max_iter=1000)


# In[172]:


logr.fit(x_train,y_train)


# In[80]:


df1 = pd.read_csv('Titanic_test.csv')
df1.head()


# In[81]:


df1.isnull().sum()


# In[83]:


df1.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df1


# In[86]:


df1_encoded = pd.get_dummies(df1)
df1_encoded


# In[88]:


df1_encoded['Age'] = df1_encoded['Age'].fillna(df1_encoded['Age'].mean())
df1_encoded['Age']


# In[89]:


df1_encoded['Age'].isnull().sum()


# In[90]:


correlation_m1 = df1_encoded.corr()
correlation_m1


# In[91]:


df1_encoded['Family size'] = df1_encoded['SibSp'] + df1_encoded['Parch'] + 1
df1_encoded['Family size'].value_counts()
df1_encoded.describe()


# In[92]:


import pandas as pd
df1_encoded['Family_type'] = df1_encoded['Family size'].apply(transform_family_size)
df1_encoded


# In[95]:


x1 = df1_encoded.drop(['SibSp','Parch','Family_type'],axis=1)
print(x1)


# In[125]:


mean_age = x1['Age'].mean()
std_age = x1['Age'].std()
print(mean_age)
print(std_age)


# In[128]:


plt.figure(figsize=(10, 5))
sns.histplot(df1['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[134]:


mean_age = df1_encoded['Age'].mean()
std_age = df1_encoded['Age'].std()
z_threshold = 3

upper_limit = mean_age + z_threshold * std_age
lower_limit = mean_age - z_threshold * std_age

df1_encoded['Age'] = df1_encoded['Age'].clip(lower=lower_limit, upper=upper_limit)
x1=df1_encoded
print(x1)


# In[158]:


df1_encoded['Fare'].plot(kind='box')


# In[ ]:





# In[135]:


num_cols1 = ['Pclass', 'Age', 'Fare', 'Family size']

scaler = StandardScaler()
X[num_cols1] = scaler.fit_transform(X[num_cols1])   
x1[num_cols1] = scaler.transform(x1[num_cols1])     


# In[136]:


print(x1)


# In[137]:


fare_mean = x1['Fare'].mean()
fare_mean


# In[138]:


x1['Fare'] = x1['Fare'].fillna(fare_mean)


# In[140]:


x1 = df1_encoded.drop(['SibSp','Parch','Family_type'],axis=1)
print(x1)


# In[180]:


x1['individual_fare'] = df1_encoded['Fare']/(df1_encoded['SibSp'] + df1_encoded['Parch'] + 1)
x1


# In[181]:


x1['individual_fare'].plot(kind='box')


# In[182]:


from scipy import stats
import numpy as np

z_testscores = stats.zscore(x1['individual_fare'])
mean_individualfare = x1['individual_fare'].mean()

outliers = (np.abs(z_testscores) > 3)
x1.loc[outliers, 'individual_fare'] = mean_individualfare
x1


# In[183]:


y_pred_log = logr.predict(x1)


# In[186]:


print(y_pred_log[:10])


# In[187]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
logr.fit(x_train, y_train)
y_pred_log = logr.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, zero_division=0))


# In[ ]:





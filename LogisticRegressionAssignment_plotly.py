import pandas as pd

df = pd.read_csv('Titanic_train.csv')
df.info()
df.describe() 

df.head()

df.isnull().sum()

df.describe(include = object)

df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df

df_encoded = pd.get_dummies(df)
df_encoded

correlation_matrix = df_encoded.corr()
correlation_matrix

import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

fig = px.imshow(correlation_matrix,text_auto=True,color_continuous_scale="RdBu_r",title="Correlation Map")
fig.update_layout(width=900,height=700)
fig.show()

df_encoded.isnull().sum()

mean_age = df['Age'].mean()
mean_age

df_encoded['Age'] = df_encoded['Age'].fillna(df_encoded['Age'].mean())
df_encoded['Age']

df_encoded['Age'].isnull().sum()

print('Unique values in categorical columns')
for col in ['Sex','Embarked']:
    print(f"{col}:{df[col].unique()}")

numerical_features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
df[numerical_features].describe()

fig = px.box(df,x="Survived",y="Pclass",color="Survived", color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(title="Pclass distribution by Survived",xaxis_title="Survived (0 = Not Survived, 1 = Survived)",yaxis_title="Pclass",width=1000,height=500)
fig.show()

fig = px.histogram(df,x="Sex",color="Survived",  barmode="group",  color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(title="Survival Rate by Sex",xaxis_title="Sex",yaxis_title="Number of Passengers",legend_title_text="Survived",width=1000,height=500)
fig.for_each_trace(lambda t: t.update(name="Not Survived" if t.name == "0" else "Survived"))
fig.show()

#box plot for age vs churn
fig = px.box(df,x="Survived",y="Age",color="Survived")
fig.update_layout(title="Age distribution by Survival status",xaxis_title="Survival (0 = Not Survived, 1 = Survived)",yaxis_title="Age",width=1000,height=500)

lower_outliers = df['Age'].quantile(0.05)
higher_outliers = df['Age'].quantile(0.95)
df['Age'] = np.where(df['Age'] < lower_outliers, lower_outliers,
             np.where(df['Age'] > higher_outliers, higher_outliers, df['Age']))

fig = px.box(df,x="Survived",y="Age",color="Survived", color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(title="Age distribution by Survival status",xaxis_title="Survival (0 = Not Survived, 1 = Survived)",yaxis_title="Age",width=1000,height=500)

mean_age = df['Age'].mean()
std_age = df['Age'].std()
print(mean_age)
print(std_age)

z_threshold = 3
upper_limit = mean_age + z_threshold * std_age
lower_limit = mean_age - z_threshold * std_age
for i in range(len(df)):
    if df.loc[i, 'Age'] < lower_limit:
        df.loc[i, 'Age'] = lower_limit
    elif df.loc[i, 'Age'] > upper_limit:
        df.loc[i, 'Age'] = upper_limit
fig = px.box(df,x="Survived",y="Age",color="Survived",color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(title="Age distribution by Survival status",xaxis_title="Survival (0 = Not Survived, 1 = Survived)",yaxis_title="Age",width=1000,height=500)

crosstab = pd.crosstab(df['Survived'], df['Pclass'], normalize='columns') * 100

# Plot as heatmap
fig = px.imshow(crosstab,text_auto=".1f", color_continuous_scale="Blues", aspect="auto")
fig.update_layout(title="Survival % by Passenger Class",xaxis_title="Pclass",yaxis_title="Survived",width=800,height=500)
fig.show()

df_encoded['Family size'] = df['SibSp'] + df['Parch'] + 1
df_encoded['Family size'].value_counts()
df_encoded.describe()

def transform_family_size(num):
    if num == 1:
        return 'alone'
    elif num >1 and num <5:
        return 'small'
    else:
        return 'large'

import pandas as pd
df_encoded['Family_type'] = df_encoded['Family size'].apply(transform_family_size)
df_encoded

import pandas as pd
counts = df_encoded[['Embarked_C', 'Embarked_Q', 'Embarked_S']].sum()

fig = px.bar(counts,x=counts.index,y=counts.values,text=counts.values,color_discrete_sequence=["skyblue"])
fig.update_layout(title="Passenger count by Embarked port",xaxis_title="Embarked",yaxis_title="Count",width=800,height=500)
fig.show()

fig = px.pie(names=counts.index,values=counts.values,color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(title="Passenger count by Embarked port",width=800,height=500)
fig.show()

df_encoded['Fare'].plot(kind='box')

df_encoded['individual_fare'] = df_encoded['Fare']/(df_encoded['SibSp'] + df_encoded['Parch'] + 1)
df_encoded

df_encoded['individual_fare'].plot(kind='box')

from scipy import stats
import numpy as np

z_scores = stats.zscore(df_encoded['individual_fare'])
mean_fare = df_encoded['individual_fare'].mean()

outliers = (np.abs(z_scores) > 3)
df_encoded.loc[outliers, 'individual_fare'] = mean_fare
df_encoded

x = df_encoded.drop(['SibSp','Parch','Survived','Family_type'],axis=1)
y=df_encoded['Survived']

X = pd.get_dummies(x,drop_first=True)
print(X.isnull().sum())

from sklearn.preprocessing import StandardScaler
X=x
num_cols = ['Pclass', 'Age', 'individual_fare', 'Family size']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(max_iter=1000)

logr.fit(x_train,y_train)

df1 = pd.read_csv('Titanic_test.csv')
df1.head()

df1.isnull().sum()

df1.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
df1

df1_encoded = pd.get_dummies(df1)
df1_encoded

df1_encoded['Age'] = df1_encoded['Age'].fillna(df1_encoded['Age'].mean())
df1_encoded['Age']

df1_encoded['Age'].isnull().sum()

correlation_m1 = df1_encoded.corr()
correlation_m1

df1_encoded['Family size'] = df1_encoded['SibSp'] + df1_encoded['Parch'] + 1
df1_encoded['Family size'].value_counts()
df1_encoded.describe()

import pandas as pd
df1_encoded['Family_type'] = df1_encoded['Family size'].apply(transform_family_size)
df1_encoded

x1 = df1_encoded.drop(['SibSp','Parch','Family_type'],axis=1)
print(x1)

mean_age = x1['Age'].mean()
std_age = x1['Age'].std()
print(mean_age)
print(std_age)

fig = px.histogram(df1,x="Age",nbins=30,color_discrete_sequence=["skyblue"])
fig.update_traces(opacity=0.7)
fig.update_layout(title="Age Distribution of Customers",xaxis_title="Age",yaxis_title="Frequency",width=1000,height=500)
fig.show()

mean_age = df1_encoded['Age'].mean()
std_age = df1_encoded['Age'].std()
z_threshold = 3

upper_limit = mean_age + z_threshold * std_age
lower_limit = mean_age - z_threshold * std_age

df1_encoded['Age'] = df1_encoded['Age'].clip(lower=lower_limit, upper=upper_limit)
x1=df1_encoded
print(x1)

df1_encoded['Fare'].plot(kind='box')


num_cols1 = ['Pclass', 'Age', 'Fare', 'Family size']

scaler = StandardScaler()
X[num_cols1] = scaler.fit_transform(X[num_cols1])   
x1[num_cols1] = scaler.transform(x1[num_cols1])     

print(x1)

fare_mean = x1['Fare'].mean()
fare_mean

x1['Fare'] = x1['Fare'].fillna(fare_mean)

x1 = df1_encoded.drop(['SibSp','Parch','Family_type'],axis=1)
print(x1)

x1['individual_fare'] = df1_encoded['Fare']/(df1_encoded['SibSp'] + df1_encoded['Parch'] + 1)
x1

x1['individual_fare'].plot(kind='box')

from scipy import stats
import numpy as np

z_testscores = stats.zscore(x1['individual_fare'])
mean_individualfare = x1['individual_fare'].mean()

outliers = (np.abs(z_testscores) > 3)
x1.loc[outliers, 'individual_fare'] = mean_individualfare
x1

y_pred_log = logr.predict(x1)

print(y_pred_log[:10])

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
logr.fit(x_train, y_train)
y_pred_log = logr.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log, zero_division=0))



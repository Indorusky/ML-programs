import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("missing_data.csv")
df = df.drop_duplicates(subset='Country')
df = df.dropna(thresh=3)
df[['Age','Salary']] = SimpleImputer(strategy='mean').fit_transform(df[['Age','Salary']])
df['Country'] = LabelEncoder().fit_transform(df['Country'])
df['Purchased'] = LabelEncoder().fit_transform(df['Purchased'])
df[['Age','Salary']] = StandardScaler().fit_transform(df[['Age','Salary']])
df.to_csv("updated_data.csv", index=False)

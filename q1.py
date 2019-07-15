'''
========================================================================================================================
Soal 1 - Diagnosis Kesuburan
========================================================================================================================
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
========================================================================================================================
Read the CSV file
========================================================================================================================
'''
df = pd.read_csv('fertility.csv')

'''
========================================================================================================================
Check the unique value for each column
========================================================================================================================
'''

# print(df.head())
# print(type(df))
# print(df['Smoking habit'].unique())                           # ['occasional' 'daily' 'never']
# print(df['Frequency of alcohol consumption'].unique())        # ['once a week' 'hardly ever or never' 'several times a week'
                                                                # 'several times a day' 'every day']
# print(df['High fevers in the last year'].unique())            # ['more than 3 months ago' 'less than 3 months ago' 'no']
# print(df['Surgical intervention'].unique())                   # ['yes' 'no']
# print(df['Accident or serious trauma'].unique())              # ['yes' 'no']
# print(df['Childish diseases'].unique())                       # ['no' 'yes']

'''
========================================================================================================================
Labeling
========================================================================================================================
'''
# from sklearn.preprocessing import LabelEncoder

# label = LabelEncoder()

# df['Childish diseases'] = label.fit_transform(df['Childish diseases'])                                   # ['no' 'yes']
# # print(label.classes_)

# df['Accident or serious trauma'] = label.fit_transform(df['Accident or serious trauma'])                 # ['no' 'yes']
# # print(label.classes_) 

# df['Surgical intervention'] = label.fit_transform(df['Surgical intervention'])                           # ['no' 'yes']
# # print(label.classes_) 

# df['High fevers in the last year'] = label.fit_transform(df['High fevers in the last year'])             # ['less than 3 months ago' 'more than 3 months ago' 'no']
# # print(label.classes_) 

# df['Frequency of alcohol consumption'] = label.fit_transform(df['Frequency of alcohol consumption'])     # ['every day' 'hardly ever or never' 'once a week' 'several times a day'
#                                                                                                         # 'several times a week']
# # print(label.classes_) 

# df['Smoking habit'] = label.fit_transform(df['Smoking habit'])                                           # ['daily' 'never' 'occasional']
# # print(label.classes_) 

df['Childish diseases'] = df['Childish diseases'].map({
    'no': 0 ,
    'yes': 1
})         # ['no' 'yes']

df['Accident or serious trauma'] = df['Accident or serious trauma'].map({
    'no': 0 ,
    'yes': 1
})          # ['no' 'yes']


df['Surgical intervention'] = df['Surgical intervention'].map({
    'no': 0 ,
    'yes': 1
})          # ['no' 'yes']


df['High fevers in the last year'] = df['High fevers in the last year'].map({
    'less than 3 months ago' : 0,
    'more than 3 months ago' : 1,
    'no' : 2
})          # ['less than 3 months ago' 'more than 3 months ago' 'no']

df['Frequency of alcohol consumption'] = df['Frequency of alcohol consumption'].map({
    'every day' : 0,
    'hardly ever or never' : 1,
    'once a week' : 2,
    'several times a day': 3,
    'several times a week' : 4
})          # ['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']

df['Smoking habit'] = df['Smoking habit'].map({
    'daily': 0,
    'never': 1,
    'occasional': 2
})          # ['daily' 'never' 'occasional']

# print(df.head())
'''
========================================================================================================================
Clean the data by deleting unused columns
========================================================================================================================
'''
df = df.drop(
    ['Season'],
    axis = 1
)
# print(df.head())

'''
==============================================================================================================
Check the null data
==============================================================================================================
'''
# print(df.isnull().sum())

'''
==============================================================================================================
Split: Feature X & Target Y
==============================================================================================================
'''
x = df.drop(['Diagnosis'], axis=1)
# print(x.head())
# print(x.iloc[0])

y = df['Diagnosis']
# print(y)

'''
==============================================================================================================
One Hot Encoder
==============================================================================================================
'''
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# coltrans = ColumnTransformer(
#     [
#         ('one_hot_encoder', OneHotEncoder(categories='auto'), [1:7])
#     ], 
#     remainder='passthrough'
# )

# x = np.array(coltrans.fit_transform(x), dtype=np.float64)
# print(x[0])

'''
==============================================================================================================
Train
==============================================================================================================
'''
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size = .1
)

'''
==============================================================================================================
Predict
==============================================================================================================
'''
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

modelLog = LogisticRegression()
modelDecTree = DecisionTreeClassifier()
modelExtra = ExtraTreesClassifier(n_estimators = 50)

modelLog.fit(xtrain,ytrain)
modelDecTree.fit(xtrain,ytrain)
modelExtra.fit(xtrain,ytrain)


'''
Input Description:
['Age',Childish diseases','Accident','Surgical','Fevers','Alcohol','Smoking','Sitting']
['no' 'yes']
['no' 'yes']
['no' 'yes']
['less than 3 months ago' 'more than 3 months ago' 'no']
['every day' 'hardly ever or never' 'once a week' 'several times a day' 'several times a week']
['daily' 'never' 'occasional']
['Sitting']
'''

nama = ['Arin','Bebi','Caca','Dini','Enno']
arin = [29, 0, 0, 0, 2, 0, 0, 5]
bebi = [31, 0, 1, 1, 2, 4, 1, 8]        # Lamanya duduk tidak disebut, jadi saya asumsikan dia bekerja 8-5 setiap hari (8 jam including 1 jam break)
caca = [25, 1, 0, 0, 0, 0, 0, 7]
dini = [28, 0, 1, 1, 0, 1, 0, 16]       # Asumsi lamanya duduk karena dikursi roda, duduk dari bangun tidur sampai tidur lagi, jadi 16 jam
enno = [42, 1, 0, 1, 0, 0, 0, 8]


print('{}, prediksi kesuburan: {} - (Logistric Regression)'.format(nama[0], modelLog.predict([arin])[0]))
print('{}, prediksi kesuburan: {} - (Decision Tree)'.format(nama[0], modelDecTree.predict([arin])[0]))
print('{}, prediksi kesuburan: {} - (Extra Tree Classifier)'.format(nama[0], modelExtra.predict([arin])[0]))


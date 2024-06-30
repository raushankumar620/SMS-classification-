#!/usr/bin/env python
# coding: utf-8

# ## SMS classification Prediction Using Multiple Machine Learning Models

# ### Steps and tasks :
# 
# ### Step 1: Reading and Understanding the Data
# 
# ### Step 2 : Data Cleaning and Preparation
# 
# ### Step 3: Visualizing the data
# 
# ### Step 4: Visualising Categorical Data
#    
# ### Step 5: Residual Analysis of Model
# 
# ### Step 6: Train-Test Split and feature scaling
# 
# ### Step 7: Model Building
# 
# ### Step 8: Prediction and Evaluation

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pd
# List of possible encodings to try
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

file_path = 'spam.csv' # Change this to the path of your CSV file

# Attempt to read the CSV file with different encodings
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"File successfully read with encoding: {encoding}")
        break # Stop the loop if successful
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue # Try the next encoding
        
# If the Loop completes without success, df will not be defined
if 'df' in locals():
    print("CSV file has been successfully loaded.")
else:
    print("All encoding attempts failed. Unable to read the CSV file.")


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


# 1. Data cleaning
# 2. EDA
# 3. Text preprocessing
# 4. Model Building


# ### Data Cleaning

# In[6]:


df.info()


# In[7]:


df.drop(columns=['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4'],inplace=True)


# In[8]:


df.sample(5)


# In[9]:


#renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[11]:


df['target']=encoder.fit_transform(df['target'])
df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


#remove duplicated value
df=df.drop_duplicates(keep='first')
df.duplicated().sum()


# In[15]:


df.shape


# ### EDA

# In[16]:


df.head()


# In[17]:


df['target'].value_counts()


# In[18]:


plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[19]:


#big chunk of ham and very less spam so out data is not balanced


# In[20]:


import nltk


# In[21]:


#!pip install nltk


# In[22]:


nltk.download('punkt')


# In[23]:


df['num_characters'] = df['text'].apply(len) #number of char


# In[24]:


df.head()


# In[25]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x))) #words


# In[26]:


df.head()


# In[27]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x))) #sentence


# In[28]:


df.head()


# In[29]:


df[['num_characters','num_words','num_sentences']].describe()


# In[30]:


#targeting ham
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[31]:


#targeting spam
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[32]:


import seaborn as sns


# In[33]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[34]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[35]:


sns.pairplot(df,hue='target')


# In[36]:


sns.heatmap(df.corr(),annot=True)


# ### Data Preprocessing

# #### -lower case
# #### -Tokenization
# #### -Removing special characters
# #### -Removing stop words and punctuation
# #### -stemming

# In[37]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')   # you may need to download the stopwords dataset

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

transformed_text = transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight k? I've cri enough today")
print(transformed_text)


# In[38]:


df['text'][10]


# In[39]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('walking')


# In[40]:


df['transformed_text']=df['text'].apply(transform_text)


# In[41]:


df.head()


# In[42]:


from wordcloud import WordCloud  # Note the corrected import statement
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[43]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[44]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[45]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[46]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[47]:


df.head()


# In[48]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[49]:


len(spam_corpus)


# In[50]:


from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming spam_corpus is a list of words
spam_counter = Counter(spam_corpus)
most_common_words = pd.DataFrame(spam_counter.most_common(30), columns=['0', '1'])

sns.barplot(data=most_common_words, x='0', y='1')
plt.xticks(rotation='vertical')
plt.show()


# In[51]:


ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[52]:


len(ham_corpus)


# In[53]:


from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming ham_corpus is a list of words
ham_counter = Counter(ham_corpus)
most_common_words = pd.DataFrame(ham_counter.most_common(30), columns=['0','1'])

sns.barplot(data=most_common_words, x='0', y='1')
plt.xticks(rotation='vertical')
plt.show()


# In[54]:


#text vectorization
#using bag of words
df.head()


# ### Building the model

# In[55]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[56]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[57]:


X.shape


# In[58]:


y=df['target'].values


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[61]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[62]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[63]:


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[64]:


mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[65]:


bnb=BernoulliNB()
bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[66]:


#tfidf -->MNB


# In[67]:


#!pip install xgboost


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier


# In[69]:


svc=SVC(kernel='sigmoid',gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear',penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)

abc=AdaBoostClassifier(n_estimators=50,random_state=2)

bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)

gbdt=GradientBoostingClassifier(n_estimators=50,random_state=2)

Xgb=XGBClassifier(n_estimators=50,random_state=2)


# In[70]:


clfs={
    'SVC':svc,
    'KN':knc,
    'NB':mnb,
    'DT':dtc,
    'LR':lrc,
    'RF':rfc,
    'Adaboost':abc,
    'Bgc':bc,
    'ETC':etc,
    'GBDT':gbdt,
    'xgb':Xgb
    
}


# In[71]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[72]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[73]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)

    print("For",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[74]:


performance_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy': accuracy_scores, 'Precision': precision_scores}).sort_values(by='Precision', ascending=False)
performance_df


# In[75]:


performance_df1=pd.melt(performance_df,id_vars="Algorithm")


# In[76]:


performance_df1


# In[77]:


sns.catplot(x='Algorithm',y='value',
           hue = 'variable',data=performance_df1,kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[78]:


#model improve
# 1. change the max_features parameter of Tfidf


# In[79]:


temp_df=pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'precision_max_ft_3000':precision_scores})


# In[80]:


new_df=performance_df.merge(temp_df,on='Algorithm')


# In[81]:


new_df_scaled=new_df.merge(temp_df,on='Algorithm')


# In[82]:


temp_df = pd.DataFrame({'Algorithm': clfs.keys(), 'Accuracy_num_chars': accuracy_scores, 'Precision_num_chars': precision_scores}).sort_values(by='Precision_num_chars', ascending=False)


# In[83]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[84]:


# voting Classifier
svc=SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb=MultinomialNB()
etc=ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[85]:


voting=VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')


# In[86]:


voting.fit(X_train,y_train)


# In[87]:


y_pred=voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))


# In[88]:


# Applying stacking
estimators=[('svm',svc),('nb',mnb),('et',etc)]
final_estimator=RandomForestClassifier


# In[89]:


from sklearn.ensemble import StackingClassifier


# In[90]:


clf=StackingClassifier(estimators=estimators,final_estimator=final_estimator)


# In[92]:


#clf.fit(X_train,y_train)
#y_pred = clf.predict(X_test)
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Precision:", precision_score(y_test, y_pred))


# In[93]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[94]:


import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
X_train=['Sample text 1','Sample text 2','Sample text 3']
y_train=[0,1,0]
tfidf=TfidfVectorizer(lowercase=True,stop_words='english')
X_train_tfidf=tfidf.fit_transform(X_train)

with open('vectorizer.pkl','wb') as vectorizer_file:
    pickle.dump(tfidf,vectorizer_file)

with open('model.pkl','wb') as model_file:
    pickle.dump(mnb,model_file)


# In[ ]:





# In[ ]:





# In[ ]:





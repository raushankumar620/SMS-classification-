# 💬 SMS Spam Classification using Machine Learning

This project focuses on classifying SMS messages as **Spam** or **Ham (Not Spam)** using various machine learning models and text processing techniques in Python. The pipeline includes preprocessing, exploratory data analysis, vectorization, model training, evaluation, and model saving.

---
  <h3> Project Steps Summary <h3/>

1. **Reading and Understanding the Data**  
   - Load dataset using `pandas`
   - Rename columns, clean data, handle duplicates and missing values

2. **Data Visualization**  
   - Use `matplotlib`, `seaborn`, and `wordcloud` to explore and understand message patterns

3. **Text Preprocessing**  
   - Tokenization, stopword removal, stemming, punctuation removal using `nltk` and `string`

4. **Feature Extraction**  
   - Convert text to numerical features using `CountVectorizer` and `TfidfVectorizer`

5. **Model Building & Evaluation**  
   - Train and compare various machine learning models from `scikit-learn`
   - Evaluate using `accuracy_score`, `confusion_matrix`, `precision_score`

6. **Model Saving**  
   - Save the best model and vectorizer using `pickle` for future predictions

---

## 🧰 Libraries & Techniques Used

### 1. 📊 Data Handling and Manipulation

- **pandas**
  - `read_csv`, `drop`, `rename`, `apply`, `value_counts`, `describe`, `isnull`, `duplicated`, `drop_duplicates`

- **numpy**
  - For numerical operations (imported but not heavily used)

---

### 2. 📈 Data Visualization

- **matplotlib**
  - `pyplot`, `figure`, `show`

- **seaborn**
  - `histplot`, `pairplot`, `heatmap`, `catplot`, `barplot`

- **wordcloud**
  - `WordCloud` class to visualize most common words

---

### 3. 📝 Text Preprocessing

- **nltk (Natural Language Toolkit)**
  - `word_tokenize`, `sent_tokenize`, `stopwords`, `PorterStemmer`, `download`

- **string**
  - Used to remove punctuation

---

### 4. 🤖 Machine Learning Models (scikit-learn)

- **Preprocessing**
  - `LabelEncoder` for encoding target labels

- **Data Splitting**
  - `train_test_split` to split data into train and test

- **Evaluation Metrics**
  - `accuracy_score`, `confusion_matrix`, `precision_score`

- **Models Used**
  - 🧠 **Naive Bayes:** `GaussianNB`, `MultinomialNB`, `BernoulliNB`
  - 📈 **Logistic Regression:** `LogisticRegression`
  - 🔍 **Support Vector Machine:** `SVC`
  - 🌳 **Tree-Based Models:** `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`
  - 📊 **Ensemble Methods:** `AdaBoostClassifier`, `BaggingClassifier`, `GradientBoostingClassifier`
  - 👥 **Others:** `KNeighborsClassifier`, `VotingClassifier`, `StackingClassifier`

- **Feature Extraction**
  - `CountVectorizer`, `TfidfVectorizer`

---

### 5. 💾 Model Saving & Loading

- **pickle**
  - `dump()` to save model and vectorizer  
  - `load()` to load them later for predictions

---

### 6. 🔄 Miscellaneous

- **collections**
  - `Counter` class used for word frequency count

- **warnings**
  - Used for ignoring unnecessary warnings during model training

---

## ✅ Output & Performance

- Multiple models were trained and evaluated
- Accuracy and precision were used as evaluation metrics
- The best performing model was saved using `pickle`

---

## 📌 Requirements

Create a `requirements.txt` with:

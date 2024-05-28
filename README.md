# DATA622
ML &amp; Big Data: SVM, LDA/QDA, NLP, RF

This repository covered topics from course DATA622 that covered various machine learning techniques. The most useful were mentioned above.

RPUBS for Homeworks 

https://rpubs.com/devcraig/DATA622as3_re

https://rpubs.com/devcraig/DATA622as2_re

The most note worthy project in this course was the NLP project used as my final which is detailed below.

## Final - NLP Methods for Topic Prediction


This dataset was originally meant for the Detect AI Competition on Kaggle as a baseline dataset to be used with a "Human" or "AI - Generated" label for supervised learning. Handily, it also contained topics for each text blurb. I decided to re-purpose the dataset to practice applying several models using the "Bag of Words" approach in Natural Language Processing. 

The [dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/)

The [competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)


### Feature Values

This process starts with removing stop words and creating feature values for each word in the training text. To create the feature values, we use the TF-IDF Vectorizer function

The TF-IDF Vectorizer creates a feature for each word by calculating two things.

1. Term Frequency - $$\frac{\text{number of times a word appears in a document}}{\text{total number of terms in document}}$$
    - this represents a term's importance to a particular document
2. Inverse Document Frequency - $$log(\frac{\text{total number of documents}}{\text{number of documents containg the word}})$$
    - this represents a term's importance across all documents
    - the IDF will weaken words that appear in many documents and strengthen words that appear in only a few documents

Using this method, we can vectorize our words into numeric values that can be computed and interpreted by the models we will choose later. The "documents" in the context of this problem, would be the prompt categories or "prompt_name".

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 42)

# Vectorizing the words using TF-IDF
# Remove the stop words
vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)

Vec_X_train = vectorizer.fit_transform(X_train)
Vec_X_test = vectorizer.transform(X_test)
```

### Feature Shapes & Visualizations

From here, it'd be nice to get an idea of the most common words and how much they comprise each prompt category, to compare with the resulting powerful words to ensure the TF-IDF Vectorizer is appropriate. To visualize this, a quick plot of the most common words across all prompt names will be collected.

```
feature_names = vectorizer.get_feature_names_out() # grab the features
arrays_of_words = vectorizer.inverse_transform(Vec_X_train) # grab the words without stop words


# Pass a dictionary of the column name I want + the list of non-stop word arrays
words_df = pd.DataFrame({'Words': arrays_of_words})

# Join the words DataFrame with y_train along the columns (axis=1)
nonstop_data = pd.concat([words_df, data['prompt_name']], axis=1)


feature_names = vectorizer.get_feature_names_out() # grab the features
arrays_of_words = vectorizer.inverse_transform(Vec_X_train) # grab the words without stop words


# Pass a dictionary of the column name I want + the list of non-stop word arrays
words_df = pd.DataFrame({'Words': arrays_of_words})

# Join the words DataFrame with y_train along the columns (axis=1)
nonstop_data = pd.concat([words_df, data['prompt_name']], axis=1)

fig, ax = plt.subplots(figsize=(10, 6))

top_words.unstack().plot(kind='bar', ax=ax)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 6 Words per prompt_name')
plt.legend(title='Words')
plt.tight_layout()
plt.show()
```

![Top 5 Words](Final_Project/plots/top_5_words.png)



### Predicting Prompt Category

For this prediction, the "bag of words" approach will be used. It is quite simple, but is great for simple tasks. The concept behind the "bag of words" approach is to create a "bag of words" with their own relative value to each category, and use their frequency to determine what prompt category they are for.

Simplicty is the advantage, while lack of semantic meaning, context, and connotation is the disadvantage.

The words have already been valued with the use of the TFIDF Vectorizer. The next step is to establish the models for classification, and train them on the word values.

The first model will be the Ridge Classifier that uses Ridge regression.

##### Ridge Regression

Ridge regression is a variant of ordinary least squares (OLS) that aims to solve issues related to multi-collinearity and overfitting due to large coefficients. Ridge accomplishes this by introducing a regularization/punish term to the cost function of OLS.

$$\text{Cost}(w)= \frac{1}{2n} \sum_{n=1}^{n} (y_i - \hat{y_i})^2 + \sigma\sum_{n=1}^{n} w^2_i$$
​

The second term in the Cost equation above is the regularization term added to the OLS cost function to create Ridge Regression. This resulting cost value is then used to adjust and train the model.

```

from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Establish our RidgeClassifier
clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")

# Fit Ridge to data
clf.fit(Vec_X_train, y_train)

# Make predictions
pred = clf.predict(Vec_X_test)

# Start plot
fig, ax = plt.subplots(figsize=(10, 5))

# Confusion Matrix Disply has a method 'from_predictions' that allow you to reference predictions and actual values directly
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(target_names) # Establishing tick labels from the groups
ax.yaxis.set_ticklabels(target_names) 
_ = ax.set_title(
    f"Confusion Matrix for {clf.__class__.__name__}" # Referencing our classifier object's name directly
)
plt.xticks(rotation='vertical')
```
![RR Conf Mat](Final_Project/plots/conf_mat.png)


#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# In[4]:


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()


# In[5]:


print(df.count())          # non-null counts
print(df.isnull().sum())   # missing counts
print(df.shape)            # total rows, columns


# In[6]:


# Count missing values per column
print(df.isnull().sum())

# Check for non-standard placeholders
print(df.eq("").sum())        # empty strings
print(df.eq("NA").sum())      # literal "NA"
print(df.eq("null").sum())    # literal "null"
print(df.eq("?").sum())       # question marks


# In[7]:


df = df.dropna()
df.info()


# In[8]:


print(df.columns.tolist())


# In[9]:


df_clean = df.dropna()


# In[10]:


print("Original:", df.shape)
print("After dropna:", df_clean.shape)


# In[11]:


df.columns


# In[12]:


# Remove duplicate column names
df = df.loc[:, ~df.columns.duplicated()]


# In[13]:


df = df.rename(
    columns={
        'RainToday': 'RainYesterday',
        'RainTomorrow': 'RainToday'
    }
)


# In[14]:


# Count observations per location
location_counts = df['Location'].value_counts()

print(location_counts)


# In[15]:


melbourne_df = df[df['Location'].isin(['MelbourneAirport', 'Watsonia', 'Melbourne'])]
print(melbourne_df.shape)


# In[16]:


print(df.columns)


# In[17]:


print(df.shape)  # current size
print(df['RainYesterday'].isnull().sum())  # how many missing target labels
print(df['RainToday'].isnull().sum())     # check another key column


# In[18]:


df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# In[19]:


print(df.shape)  # current size
print(df.columns)  # confirm column names
print(df.isnull().sum())  # see missing values per column


# In[20]:


df.isnull().sum()


# In[21]:


# Show all rows that have at least one missing value
df[df.isnull().any(axis=1)]


# In[22]:


missing_rows = df.isnull().any(axis=1).sum()
print("Rows with at least one missing value:", missing_rows)


# In[23]:


def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


# In[24]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the original 'Date' column
df = df.drop(columns=['Date'])

# Display the updated DataFrame
df.head()


# In[25]:


# Define feature and target dataframes
X = df.drop(columns='RainToday', axis=1)
y = df['RainToday']


# In[26]:


print(X.shape)
print(y.shape)


# In[27]:


print(y.value_counts())


# In[28]:


print(y.value_counts(normalize=True))


# In[29]:


from sklearn.model_selection import train_test_split

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)


# In[30]:


# Automatically detect numerical and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# In[31]:


print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# In[32]:


# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[33]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[34]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[35]:


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}


# In[36]:


cv = StratifiedKFold(n_splits=5, shuffle=True)


# In[37]:


# Instantiate GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,        # your pipeline with preprocessor + classifier
    param_grid=param_grid,     # dictionary of hyperparameters to search
    cv=5,                      # number of cross-validation folds
    scoring='accuracy',        # evaluation metric
    verbose=2                  # show progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)


# In[38]:


print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))


# In[39]:


print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)


# In[40]:


y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))


# In[41]:


test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))


# In[42]:


y_pred = grid_search.predict(X_test)


# In[43]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[44]:


conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# In[45]:


# Assuming binary classification with labels [No Rain, Rain]
tn, fp, fn, tp = conf_matrix.ravel()

tpr = tp / (tp + fn)
print("True Positive Rate (Recall): {:.2f}".format(tpr))


# In[46]:


# Extract feature importances from the best estimator
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


# In[47]:


# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


# In[48]:


# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

# Update the grid search parameter grid
grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
model = grid_search.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# In[49]:


print(classification_report(y_test, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# In[50]:


# Random Forest pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Logistic Regression pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])


# In[51]:


# Random Forest parameter grid
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Logistic Regression parameter grid
lr_param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}


# In[52]:


# Random Forest GridSearch
rf_grid = GridSearchCV(
    estimator=rf_pipeline,       # pipeline with RandomForestClassifier
    param_grid=rf_param_grid,    # RF hyperparameters
    cv=5,
    scoring='accuracy',
    verbose=2
)
rf_model = rf_grid.fit(X_train, y_train)   # <-- now rf_model exists

# Logistic Regression GridSearch
lr_grid = GridSearchCV(
    estimator=lr_pipeline,       # pipeline with LogisticRegression
    param_grid=lr_param_grid,    # LR hyperparameters
    cv=5,
    scoring='accuracy',
    verbose=2
)
lr_model = lr_grid.fit(X_train, y_train)   # <-- now lr_model exists


# In[53]:


# Random Forest results
rf_pred = rf_model.predict(X_test)   # assuming rf_model is your fitted grid_search with RandomForest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Logistic Regression results
lr_pred = lr_model.predict(X_test)   # assuming lr_model is your fitted grid_search with LogisticRegression
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

# Side-by-side confusion matrices
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))


# In[54]:


# Random Forest
rf_pred = rf_model.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Logistic Regression
lr_pred = lr_model.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))


# In[55]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Random Forest accuracy
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred) * 100
print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy))

# Logistic Regression accuracy
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred) * 100
print("Logistic Regression Accuracy: {:.2f}%".format(lr_accuracy))


# In[56]:


# Random Forest correct predictions
rf_correct = (rf_pred == y_test).sum()
print("Random Forest correct predictions:", rf_correct)

# Logistic Regression correct predictions
lr_correct = (lr_pred == y_test).sum()
print("Logistic Regression correct predictions:", lr_correct)


# In[57]:


from sklearn.metrics import confusion_matrix

# Confusion matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, lr_pred)

# Assuming binary classification with labels [No Rain, Rain]
tn, fp, fn, tp = conf_matrix.ravel()

tpr_lr = tp / (tp + fn)
print("True Positive Rate (Logistic Regression): {:.2f}".format(tpr_lr))


# In[58]:


from sklearn.metrics import accuracy_score, confusion_matrix

# --- Random Forest ---
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred) * 100
rf_correct = (rf_pred == y_test).sum()

rf_conf_matrix = confusion_matrix(y_test, rf_pred)
tn_rf, fp_rf, fn_rf, tp_rf = rf_conf_matrix.ravel()
rf_tpr = tp_rf / (tp_rf + fn_rf)

# --- Logistic Regression ---
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred) * 100
lr_correct = (lr_pred == y_test).sum()

lr_conf_matrix = confusion_matrix(y_test, lr_pred)
tn_lr, fp_lr, fn_lr, tp_lr = lr_conf_matrix.ravel()
lr_tpr = tp_lr / (tp_lr + fn_lr)

# --- Print comparison ---
print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy))
print("Random Forest Correct Predictions:", rf_correct)
print("Random Forest TPR: {:.2f}".format(rf_tpr))

print("\nLogistic Regression Accuracy: {:.2f}%".format(lr_accuracy))
print("Logistic Regression Correct Predictions:", lr_correct)
print("Logistic Regression TPR: {:.2f}".format(lr_tpr))


# In[59]:


import matplotlib.pyplot as plt
import numpy as np

# Metrics you already computed
rf_metrics = [rf_accuracy, rf_correct, rf_tpr]
lr_metrics = [lr_accuracy, lr_correct, lr_tpr]

# Labels for the metrics
labels = ['Accuracy (%)', 'Correct Predictions', 'True Positive Rate (TPR)']

x = np.arange(len(labels))  # positions for groups
width = 0.35                # width of the bars

fig, ax = plt.subplots(figsize=(8,6))

# Bars for Random Forest and Logistic Regression
rects1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest', color='skyblue')
rects2 = ax.bar(x + width/2, lr_metrics, width, label='Logistic Regression', color='lightgreen')

# Add labels, title, legend
ax.set_ylabel('Scores')
ax.set_title('Comparison of Random Forest vs Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Annotate bars with values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()


# In[60]:


import matplotlib.pyplot as plt
import numpy as np

# Total test samples
total_samples = len(y_test)

# Normalize correct predictions into percentages
rf_correct_pct = (rf_correct / total_samples) * 100
lr_correct_pct = (lr_correct / total_samples) * 100

# Metrics (all in percentages now)
rf_metrics = [rf_accuracy, rf_correct_pct, rf_tpr * 100]
lr_metrics = [lr_accuracy, lr_correct_pct, lr_tpr * 100]

# Labels for the metrics
labels = ['Accuracy (%)', 'Correct Predictions (%)', 'True Positive Rate (%)']

x = np.arange(len(labels))  # positions for groups
width = 0.35                # width of the bars

fig, ax = plt.subplots(figsize=(8,6))

# Bars for Random Forest and Logistic Regression
rects1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest', color='skyblue')
rects2 = ax.bar(x + width/2, lr_metrics, width, label='Logistic Regression', color='lightgreen')

# Add labels, title, legend
ax.set_ylabel('Percentage (%)')
ax.set_title('Comparison of Random Forest vs Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Annotate bars with values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()


# In[61]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score

# --- Compute Precision for both models ---
rf_precision = precision_score(y_test, rf_pred, pos_label='Yes') * 100
lr_precision = precision_score(y_test, lr_pred, pos_label='Yes') * 100

# Normalize correct predictions into percentages
total_samples = len(y_test)
rf_correct_pct = (rf_correct / total_samples) * 100
lr_correct_pct = (lr_correct / total_samples) * 100

# Metrics (all in percentages now)
rf_metrics = [rf_accuracy, rf_correct_pct, rf_tpr * 100, rf_precision]
lr_metrics = [lr_accuracy, lr_correct_pct, lr_tpr * 100, lr_precision]

# Labels for the metrics
labels = ['Accuracy (%)', 'Correct Predictions (%)', 'True Positive Rate (%)', 'Precision (%)']

x = np.arange(len(labels))  # positions for groups
width = 0.35                # width of the bars

fig, ax = plt.subplots(figsize=(10,6))

# Bars for Random Forest and Logistic Regression
rects1 = ax.bar(x - width/2, rf_metrics, width, label='Random Forest', color='skyblue')
rects2 = ax.bar(x + width/2, lr_metrics, width, label='Logistic Regression', color='lightgreen')

# Add labels, title, legend
ax.set_ylabel('Percentage (%)')
ax.set_title('Comparison of Random Forest vs Logistic Regression')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Annotate bars with values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# HIC_FINTECH1_WIN_BACK

Here’s a structured list of 15 tasks, each with its purpose, description, and the code needed to complete the task. These tasks will guide a beginner-level data scientist through the process of performing exploratory data analysis (EDA), building models, and calculating the win-back ratio to assess the model's impact.

---

### **Task 1: Load and Explore the Data**
**Description:**  
The first step in any analysis is to load the dataset and get a basic understanding of its structure.

```python
import pandas as pd

# Load the data
df = pd.read_csv('win_back_data.csv')

# Check the first few rows of the data
df.head()

# Get a summary of the dataset
df.info()
```

---

### **Task 2: Summary Statistics**
**Description:**  
Generate summary statistics to understand the distribution of numerical features in the dataset.

```python
# Generate summary statistics
df.describe()
```

---

### **Task 3: Missing Values Check**
**Description:**  
Identify and quantify missing values in the dataset to understand the data's completeness.

```python
# Check for missing values
df.isnull().sum()
```

---

### **Task 4: Distribution of the Target Variable**
**Description:**  
Analyze the distribution of the target variable (`winback_success`) to understand the class balance.

```python
# Distribution of the target variable
df['winback_success'].value_counts(normalize=True)
```

---

### **Task 5: Visualize Numerical Features**
**Description:**  
Visualize the distribution of important numerical features to better understand their spread and skewness.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize numerical features
num_cols = ['user_tenure', 'num_logins_last_30_days', 'num_transactions_last_30_days', 'app_usage_time', 'discount_offered']

for col in num_cols:
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```

---

### **Task 6: Categorical Features Analysis**
**Description:**  
Examine the distribution of categorical features to get insights into the variety of values present in each.

```python
# Analyze categorical features
cat_cols = ['customer_segment', 'campaign_type', 'prev_winback_success']

for col in cat_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()
```

---

### **Task 7: Correlation Analysis**
**Description:**  
Understand relationships between numerical features by computing a correlation matrix.

```python
# Correlation matrix
corr_matrix = df[num_cols].corr()

# Heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

---

### **Task 8: Handle Missing Data**
**Description:**  
Handle missing data by imputing or dropping missing values to ensure a clean dataset for model building.

```python
# Fill missing values in discount_offered with the median
df['discount_offered'].fillna(df['discount_offered'].median(), inplace=True)

# Drop rows with missing 're_engagement_date'
df.dropna(subset=['re_engagement_date'], inplace=True)
```

---

### **Task 9: Feature Engineering**
**Description:**  
Create new features such as the time between unsubscription and re-engagement to add more predictive power.

```python
# Create a new feature: days between unsub and re-engagement
df['days_until_re_engagement'] = (pd.to_datetime(df['re_engagement_date']) - pd.to_datetime(df['unsub_date'])).dt.days
```

---

### **Task 10: Train-Test Split**
**Description:**  
Split the dataset into training and testing sets for model building and evaluation.

```python
from sklearn.model_selection import train_test_split

# Define features and target
X = df[['user_tenure', 'num_logins_last_30_days', 'num_transactions_last_30_days', 'app_usage_time', 'discount_offered', 'prev_winback_success', 'days_until_re_engagement']]
y = df['winback_success']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **Task 11: Build a Classification Model**
**Description:**  
Build a logistic regression model to predict win-back success.

```python
from sklearn.linear_model import LogisticRegression

# Instantiate the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
```

---

### **Task 12: Model Evaluation**
**Description:**  
Evaluate the performance of the model using accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
```

---

### **Task 13: Feature Importance**
**Description:**  
Examine the importance of each feature in predicting the target variable.

```python
# Get feature importance
importance = model.coef_[0]
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Score: {v}')
```

---

### **Task 14: Predict on New Data**
**Description:**  
Use the trained model to predict win-back success on new data.

```python
# Predict win-back propensity on new data
new_data = X_test.iloc[0:5]  # Taking a sample of 5 new records
predictions = model.predict(new_data)
predictions
```

---

### **Task 15: Calculate the Win-Back Ratio**
**Description:**  
Calculate the win-back ratio and assess the model’s impact on improving the success rate.

```python
# Calculate win-back success rate before the model (random targeting)
baseline_success_rate = df['winback_success'].mean()

# Calculate predicted win-back success rate after the model
predicted_success_rate = y_pred.mean()

# Improvement in win-back success rate
improvement = (predicted_success_rate - baseline_success_rate) / baseline_success_rate * 100

print(f'Baseline Success Rate: {baseline_success_rate:.2%}')
print(f'Predicted Success Rate: {predicted_success_rate:.2%}')
print(f'Improvement in Success Rate: {improvement:.2f}%')
```

---

These 15 tasks will guide a beginner-level data scientist through the end-to-end process of data exploration, model building, evaluation, and calculating the final win-back improvement ratio.

Here’s a breakdown of the 15 tasks with clear, step-by-step bullet points, aimed at guiding a beginner-level data scientist to write their own code. The focus is to help them understand what needs to be done at each step without ambiguity.

---

### **Task 1: Load and Explore the Data**
- Load the dataset into a pandas DataFrame.
- Display the first few rows of the dataset to get an initial look.
- Use the `.info()` method to check data types and the presence of any missing values.

---

### **Task 2: Summary Statistics**
- Use the `.describe()` method to generate summary statistics for all numerical columns.
- This should give you a good idea of the range, mean, and spread of the data.

---

### **Task 3: Missing Values Check**
- Check if there are any missing values in the dataset.
- Use the `.isnull().sum()` method to get the count of missing values in each column.
- Identify which columns need attention for handling missing data.

---

### **Task 4: Distribution of the Target Variable**
- Find the distribution of the target column `winback_success`.
- Use `.value_counts()` to count the unique values (0 and 1).
- Check if the dataset is imbalanced (i.e., one class significantly outnumbers the other).

---

### **Task 5: Visualize Numerical Features**
- Import `matplotlib` and `seaborn` libraries for visualization.
- Plot histograms for the following columns: `user_tenure`, `num_logins_last_30_days`, `num_transactions_last_30_days`, `app_usage_time`, and `discount_offered`.
- Use `sns.histplot()` to plot the distribution of these columns and examine their shapes (normal, skewed, etc.).

---

### **Task 6: Categorical Features Analysis**
- Plot bar charts for categorical columns: `customer_segment`, `campaign_type`, `prev_winback_success`.
- Use `sns.countplot()` to show the frequency of each category.
- This will help you understand how different categories are distributed in your data.

---

### **Task 7: Correlation Analysis**
- Compute the correlation matrix using `.corr()` to understand relationships between numerical features.
- Use a heatmap from `seaborn` (`sns.heatmap()`) to visualize these correlations.
- Focus on any strong correlations (close to +1 or -1).

---

### **Task 8: Handle Missing Data**
- Identify columns with missing data from Task 3.
- For `discount_offered`, fill missing values with the median using `.fillna()`.
- For `re_engagement_date`, drop rows with missing values using `.dropna()`.

---

### **Task 9: Feature Engineering**
- Create a new feature, `days_until_re_engagement`, which calculates the difference between `unsub_date` and `re_engagement_date`.
- Convert these columns to date format using `pd.to_datetime()`.
- Subtract the two date columns and convert the result to integer days.

---

### **Task 10: Train-Test Split**
- Define the input features (`X`) and the target variable (`y`).
- Select columns like `user_tenure`, `num_logins_last_30_days`, `num_transactions_last_30_days`, etc., as your features.
- Use `train_test_split()` from `sklearn.model_selection` to split your data into training and testing sets (80-20 split).

---

### **Task 11: Build a Classification Model**
- Import `LogisticRegression` from `sklearn.linear_model`.
- Create an instance of the model and fit it on the training data (`X_train`, `y_train`).
- Ensure the model learns the relationships between the features and the target variable.

---

### **Task 12: Model Evaluation**
- Predict on the test set (`X_test`) using the `.predict()` method.
- Import accuracy, precision, recall, and F1-score metrics from `sklearn.metrics`.
- Calculate each metric for the predictions and display the results.

---

### **Task 13: Feature Importance**
- After training the model, retrieve the model coefficients using `.coef_` to understand feature importance.
- Print the features and their corresponding coefficients to see which features are most influential.
- Interpret which features have positive or negative impacts on the prediction.

---

### **Task 14: Predict on New Data**
- Take a few samples from your test set (`X_test`) and predict the win-back success.
- Use the model’s `.predict()` method on the new data.
- Display the predicted values to understand how the model generalizes to unseen data.

---

### **Task 15: Calculate the Win-Back Ratio**
- Calculate the baseline win-back success rate (before the model) by taking the mean of the `winback_success` column.
- Calculate the predicted win-back success rate after applying the model (using `y_pred` from Task 12).
- Compute the percentage improvement in win-back success by comparing the predicted success rate to the baseline.

---

This step-by-step approach will guide the beginner through writing their own code for each task while understanding why each task is important. The instructions are concise, clear, and structured to eliminate ambiguity.

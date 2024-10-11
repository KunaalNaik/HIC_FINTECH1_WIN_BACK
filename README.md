# HIC_FINTECH1_WIN_BACK

Here's a detailed set of 16 tasks, each with a title, description, code, and a `TODO` comment for further learning or improvement.

---

### **Task 1: Load and Explore the Data**
**Description:**  
Load the dataset into a pandas DataFrame and explore its basic structure to understand the data you will work with.

```python
import pandas as pd

# Load the data
df = pd.read_csv('win_back_data.csv')

# Display the first few rows
df.head()

# Check the data types and missing values
df.info()

# TODO: Explore the dataset further by checking unique values for each column using df.nunique().
```

---

### **Task 2: Summary Statistics**
**Description:**  
Generate summary statistics to understand the spread and central tendencies of the numerical data.

```python
# Generate summary statistics for numerical columns
df.describe()

# TODO: Check summary statistics for categorical columns using df.describe(include=['object']).
```

---

### **Task 3: Missing Values Check**
**Description:**  
Identify missing values in the dataset and quantify them to determine if further cleaning is required.

```python
# Check for missing values in each column
df.isnull().sum()

# TODO: Create a heatmap to visualize missing data using seaborn's heatmap function.
```

---

### **Task 4: Distribution of the Target Variable**
**Description:**  
Examine the distribution of the `winback_success` target variable to check for class imbalance.

```python
# Check the distribution of the target variable
df['winback_success'].value_counts(normalize=True)

# TODO: Plot the target variable distribution using a bar chart.
```

---

### **Task 5: Visualize Numerical Features**
**Description:**  
Visualize the distribution of key numerical features to identify trends, skewness, and potential outliers.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot histograms for numerical columns
num_cols = ['user_tenure', 'num_logins_last_30_days', 'num_transactions_last_30_days', 'app_usage_time', 'discount_offered']

for col in num_cols:
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# TODO: Experiment with different bins in histograms to better visualize the data.
```

---

### **Task 6: Outlier Detection (Numerical Features)**
**Description:**  
Use boxplots to detect outliers in the numerical features, which can distort model performance.

```python
# Plot boxplots to detect outliers in numerical features
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# TODO: Learn about other outlier detection methods like Z-score and Mahalanobis distance.
```

---

### **Task 7: Categorical Features Analysis**
**Description:**  
Visualize categorical features to understand their distribution and identify any skewed categories.

```python
# Plot bar charts for categorical columns
cat_cols = ['customer_segment', 'campaign_type', 'prev_winback_success']

for col in cat_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# TODO: Explore using pie charts or stacked bar charts for categorical analysis.
```

---

### **Task 8: Handle Missing Data**
**Description:**  
Handle missing data in critical columns by imputing values or removing rows to clean the dataset.

```python
# Fill missing values in 'discount_offered' with the median
df['discount_offered'].fillna(df['discount_offered'].median(), inplace=True)

# Drop rows with missing 're_engagement_date'
df.dropna(subset=['re_engagement_date'], inplace=True)

# TODO: Try other imputation methods like mean or K-Nearest Neighbors (KNN) imputation for continuous variables.
```

---

### **Task 9: Outlier Removal**
**Description:**  
Remove outliers using the Interquartile Range (IQR) method to ensure that extreme values don't skew model results.

```python
# Remove outliers using the IQR method
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Filter out rows where any numerical feature falls outside the IQR bounds
df_filtered = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# TODO: Explore how to handle outliers using techniques like Winsorization or capping.
```

---

### **Task 10: Feature Engineering**
**Description:**  
Create new features that add predictive power, such as calculating the difference between unsubscription and re-engagement dates.

```python
# Convert date columns to datetime
df['unsub_date'] = pd.to_datetime(df['unsub_date'])
df['re_engagement_date'] = pd.to_datetime(df['re_engagement_date'])

# Create a new feature: days_until_re_engagement
df['days_until_re_engagement'] = (df['re_engagement_date'] - df['unsub_date']).dt.days

# TODO: Create other features such as interaction terms or scaling numerical columns.
```

---

### **Task 11: Train-Test Split with Stratification**
**Description:**  
Split the dataset into training and testing sets, ensuring the same distribution of the target variable in both using stratification.

```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df_filtered[['user_tenure', 'num_logins_last_30_days', 'num_transactions_last_30_days', 'app_usage_time', 'discount_offered', 'prev_winback_success', 'days_until_re_engagement']]
y = df_filtered['winback_success']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TODO: Try different test sizes (e.g., 30%) and see how it affects model performance.
```

---

### **Task 12: Build a Classification Model**
**Description:**  
Train a logistic regression model on the training data to predict win-back success.

```python
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# TODO: Explore other classification algorithms like Random Forest, Decision Trees, or XGBoost.
```

---

### **Task 13: Model Evaluation**
**Description:**  
Evaluate the model on the test set by calculating accuracy, precision, recall, and F1-score.

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

# TODO: Learn about other metrics like ROC-AUC and confusion matrix for a deeper understanding of model performance.
```

---

### **Task 14: Feature Importance**
**Description:**  
Check which features contributed most to the predictions by inspecting the logistic regression model’s coefficients.

```python
# Get feature importance from model coefficients
importance = model.coef_[0]
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Score: {v}')

# TODO: Visualize feature importance using bar plots or other techniques for better interpretation.
```

---

### **Task 15: Predict on New Data**
**Description:**  
Use the trained model to predict win-back success on new, unseen data points.

```python
# Predict on a sample of new data
new_data = X_test.iloc[0:5]  # Select a few rows from the test set
predictions = model.predict(new_data)

# Display predictions
predictions

# TODO: Try predicting on a fully new dataset to check model robustness.
```

---

### **Task 16: Calculate the Win-Back Ratio**
**Description:**  
Calculate the win-back success rate from both the original data and model predictions to assess the impact.

```python
# Baseline win-back success rate (before the model)
baseline_success_rate = df_filtered['winback_success'].mean()

# Predicted win-back success rate
predicted_success_rate = y_pred.mean()

# Improvement in win-back success rate
improvement = (predicted_success_rate - baseline_success_rate) / baseline_success_rate * 100

print(f'Baseline Success Rate: {baseline_success_rate:.2%}')
print(f'Predicted Success Rate: {predicted_success_rate:.2%}')
print(f'Improvement in Success Rate: {improvement:.2f}%')

# TODO: Experiment with different model thresholds (e.g., 0.5, 0.6) to optimize success rate.
```

---

This structured set of tasks helps the beginner-level data scientist go through the process, while encouraging further learning with `TODO` comments for deeper exploration. Each task is simple, well-explained, and provides room for growth.

---

These 15 tasks will guide a beginner-level data scientist through the end-to-end process of data exploration, model building, evaluation, and calculating the final win-back improvement ratio.

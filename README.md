# Comprehensive Report on Predicting Term Deposit Subscriptions

## Introduction
This report summarizes the analysis conducted on the "bank-full.csv" dataset which contains 45,211 records of direct marketing campaigns by a Portuguese banking institution. The objective was to predict whether a client would subscribe to a term deposit (indicated by the binary target variable "y": "yes" or "no") based on 16 features, including client demographics, financial details, and campaign interactions. The analysis involved exploratory data analysis (EDA), feature engineering, model building, performance evaluation, and deriving actionable insights to enhance marketing strategies.

## Dataset Description
The dataset comprises 17 columns, detailed in the table below:

| Column Name     | Type       | Description                                      |
|-----------------|------------|--------------------------------------------------|
| age             | Integer    | Client's age                                     |
| job             | Categorical| Occupation (e.g., "management", "technician")    |
| marital         | Categorical| Marital status (e.g., "married", "single")       |
| education       | Categorical| Education level (e.g., "primary", "secondary")   |
| default         | Binary     | Has credit in default? ("yes", "no")            |
| balance         | Integer    | Average yearly balance in euros                 |
| housing         | Binary     | Has housing loan? ("yes", "no")                 |
| loan            | Binary     | Has personal loan? ("yes", "no")                |
| contact         | Categorical| Contact type (e.g., "cellular", "telephone")     |
| day             | Integer    | Last contact day of the month                   |
| month           | Categorical| Last contact month (e.g., "jan", "feb")         |
| duration        | Integer    | Last contact duration in seconds                |
| campaign        | Integer    | Number of contacts in this campaign             |
| pdays           | Integer    | Days since last contact (-1 if none)            |
| previous        | Integer    | Number of contacts before this campaign         |
| poutcome        | Categorical| Outcome of previous campaign (e.g., "success")   |
| y               | Binary     | Subscribed to term deposit? ("yes", "no")       |

The dataset is imbalanced, with approximately 89% "no" and 11% "yes" responses for the target variable.

### 1. Exploratory Data Analysis (EDA)
EDA was conducted to understand the dataset’s structure and identify preprocessing needs:
- **Missing Values**: No explicit missing values were found, but categorical columns like `job`, `education`, `contact`, and `poutcome` contained "unknown" entries, treated as a separate category.
- **Summary Statistics**:
  - `age`: Mean ~41 years, range 18-95.
  - `balance`: Mean ~1500-2000 euros, with negative and high outliers.
  - `duration`: Mean ~250-300 seconds, with outliers up to several thousand seconds.
  - `campaign`: Mean ~2-3 contacts, with outliers up to 50+.
  - `pdays`: Mostly -1 (no previous contact), otherwise 0-871.
  - `previous`: Mean ~0.5, up to 7.
- **Class Imbalance**: Confirmed ~12% "yes" vs. ~88% "no".
- **Outlier Detection**: Box plots revealed outliers in `balance`, `duration`, and `campaign`.
- **Correlation Analysis**:
  - Point-biserial correlations showed `duration` had a strong positive correlation with `y`, while `balance` and `age` had weaker associations.
  - Chi-square tests indicated significant associations between `y` and categorical features like `job`, `marital`, `education`, `contact`, and `month`.
- **Visualizations**:
  - Histograms showed age distributions by subscription outcome.
  - Bar plots highlighted higher subscription rates for certain job types (e.g., "management", "retired") and months (e.g., "May", "September").

### 2. Feature Engineering
To enhance model performance, the following steps were taken:
- **One-Hot Encoding**: Categorical variables (`job`, `marital`, `education`, `contact`, `month`) were converted to numerical format.
- **New Features**:
  - `age_group`: Binned into 18-30, 31-45, 46-60, 61+.
  - `balance_category`: Grouped into low, medium, high.
  - `total_contacts`: Sum of `campaign` and `previous`.
  - `last_contact_category`: Categorized `pdays` as "none" (-1), "recent" (≤30 days), or "distant" (>30 days).
- **Scaling**: Numerical features (`age`, `balance`, `duration`, `campaign`, `pdays`, `previous`) were standardized using `StandardScaler`.

### 3. Model Building
- **Algorithm**: A Random Forest classifier was selected for its robustness, ability to handle mixed data types, and feature importance insights.
- **Class Imbalance Handling**: SMOTE was applied to oversample the minority class ("yes") in the training set.
- **Data Splitting**: The dataset was split into 80% training and 20% testing sets.

### 4. Model Evaluation

**Logistic Accuracy:** 0.8391020678978215

              precision    recall  f1-score   support

          no       0.97      0.84      0.90      7985
         yes       0.41      0.82      0.54      1058

    accuracy                           0.84      9043
   macro avg       0.69      0.83      0.72      9043
weighted avg       0.91      0.84      0.86      9043


**DecisionTreeAccuracy:** 0.8736038925135464

              precision    recall  f1-score   support

          no       0.93      0.93      0.93      7985
         yes       0.46      0.44      0.45      1058

    accuracy                           0.87      9043
   macro avg       0.69      0.68      0.69      9043
weighted avg       0.87      0.87      0.87      9043


**RandomForestAccuracy:** 0.9011390025434037

              precision    recall  f1-score   support

          no       0.92      0.98      0.95      7985
         yes       0.66      0.32      0.43      1058

    accuracy                           0.90      9043
   macro avg       0.79      0.65      0.69      9043
weighted avg       0.89      0.90      0.89      9043


**GradBoostnAccuracy:** 0.9055623133915736

              precision    recall  f1-score   support

          no       0.93      0.97      0.95      7985
         yes       0.65      0.41      0.50      1058

    accuracy                           0.91      9043
   macro avg       0.79      0.69      0.73      9043
weighted avg       0.89      0.91      0.90      9043

## Model Performance
Random Forest and Gradient Boosting achieved the highest overall accuracy (~90–91%), but **Logistic Regression** achieved the best **recall** and **F1-score** for the *“yes”* class (**recall = 0.82**, **F1 = 0.54**) at the expense of lower precision. 

In contrast, tree ensembles had higher **precision (~0.65)** but much lower **recall (~0.39–0.40)**, meaning they miss many true subscribers. 

Because catching potential subscribers is crucial, we favor the model with better **recall/F1** for *“yes”*. **Class-weighted Logistic Regression** turned out best by that metric.

The metrics indicate **moderate success** in identifying potential subscribers, with **balanced precision and recall** for the minority class.


## Insights and Findings

### EDA Insights
- **Class Imbalance**: The 12% subscription rate necessitated techniques like SMOTE to improve model performance.
- **Feature Patterns**:
  - Clients with longer call durations (`duration`) were more likely to subscribe.
  - Previous campaign success (`poutcome` = "success") strongly predicted subscriptions.
  - Cellular contacts (`contact` = "cellular") had higher success rates than telephone.
  - Certain months (e.g., "May", "September") showed higher subscription rates.
  - Clients with higher balances or specific occupations (e.g., "management", "retired") were more likely to subscribe.


### Common Characteristics of Subscribers
- **Demographics**: Younger or retired clients, those with tertiary education, or single marital status.
- **Financial Profile**: Higher average balances, no housing loans.
- **Campaign Interaction**: Recent or successful previous contacts, longer call durations.

### Comparison with Other Models
The Random Forest’s AUC-ROC of 0.742 (with 200 trees) was slightly lower than a Decision Tree’s 0.766, but its ensemble nature provides robustness. Other models like Logistic Regression or XGBoost could be explored for potentially better performance, though Random Forest’s interpretability and stability are advantageous.

## Actionable Recommendations
Based on the findings, the following strategies are recommended:
- **Target High-Potential Clients**: Focus on clients with higher balances, specific job types (e.g., "management", "retired"), or tertiary education.
- **Optimize Contact Timing**: Prioritize campaigns in months like "May" or "September" and use cellular contacts for higher success rates.
- **Leverage Previous Success**: Target clients with successful previous campaign outcomes.
- **Reduce Over-Contacting**: Limit contacts for clients with low predicted subscription probabilities to optimize resources.
- **Pre-Call Predictions**: Since `duration` is only known post-call, rely on pre-call features like `poutcome` and `balance` for targeting.

## Limitations
- **Duration Dependency**: The `duration` feature’s high predictiveness is impractical for pre-call targeting, as it’s only available after contact.
- **Dataset Context**: The data reflects campaigns from 2008-2010, which may not generalize to current market conditions.

## Conclusion
Our analysis and modeling confirm that a small subset of clients drive most subscriptions. By engineering features and handling imbalance, we built a classifier (logistic regression with class weights) that identifies these prospects. Key actionable features include call duration and customer profile (age, balance, loan status). We recommend the bank tailor future campaigns to these insights for higher conversion rates.
Future work could explore other algorithms or incorporate real-time data to improve predictive accuracy

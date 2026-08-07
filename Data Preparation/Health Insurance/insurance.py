# ============================================================
# Part 1 - EDA + Cleaning + Preprocessing + Feature Engineering
# ============================================================

import numpy as np                  # Numerical operations
import pandas as pd                 # Data handling
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Better statistical plots
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency


from sklearn.model_selection import train_test_split # Train-test split
from sklearn.linear_model import LinearRegression # Regression model


from sklearn.metrics import r2_score # Model accuracy metric

"""
=========================================================
Load Dataset

read_csv()
Reads CSV file and stores it as a Pandas DataFrame.

Target Variable:
charges -> Value we want to predict.
=========================================================
"""
data = pd.read_csv("./health_insurance_data.csv", encoding="utf-8")


# ==========================================================
# BASIC EDA
# ==========================================================

# Dataset size -> (Rows, Columns)
data.shape

# First 5 rows
data.head()

# Last 5 rows
data.tail()

"""
=========================================================
info()

Shows:
• Datatypes
• Non-null values
• Memory usage
=========================================================
"""
# data.info()


"""
=========================================================
describe()

Returns summary statistics of numerical columns.

Useful for:
• Mean
• Std
• Min / Max
• Quartiles

Quick mathematical overview of the dataset.
=========================================================
"""
data.describe()


"""
=========================================================
isnull().sum()

isnull() -> Detect missing values.
sum()    -> Count total missing values.

Why?
ML algorithms generally cannot learn directly
from missing values.
=========================================================
"""
data.isnull().sum()

'''
Dataset Quality Evaluated From Above Operations:-
  • Total Samples : 1338
  • Features      : 7
  • Missing Values: 0
  • Duplicate Rows: 1

  Insight:
  This is a very clean dataset. That's why this project
  focuses more on EDA than heavy cleaning.
'''


# ==========================================================
# VISUALIZATION
# ==========================================================

numeric_columns = ["age", "bmi", "children", "charges"]

"""
=========================================================
Histogram

histplot() -> Distribution
kde=True   -> Smooth density curve
bins=20    -> Histogram divisions

Goal:
Understand distribution, skewness and possible outliers.
=========================================================
"""
for col in numeric_columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()

'''
charges:-
   Observation:
   Histogram is positively (right) skewed.

   Insight:
   Most people have relatively low/medium insurance
   charges, while a small number have very high charges.
   Those high values stretch the distribution.

Age:-
   Observation:
   Age distribution is fairly balanced.

   Insight:
   Good! The model gets examples from different age groups
   instead of being biased toward only young or old people.

BMI:-
   Observation:
   BMI looks close to a bell-shaped distribution with a few
   high-value observations.

   Insight:
   Creating BMI categories (Normal, Overweight, Obese)
   is a meaningful feature engineering step because doctors
   naturally think in these categories.

Children:-
   Observation:
   Most families have fewer children.

   Insight:
   High child-count families are comparatively rare.
'''


"""
=========================================================
Count Plot

Shows frequency of categorical values.

Useful for checking class balance.
=========================================================
"""
sns.countplot(x=data["children"])
plt.show()

data.groupby("sex")['charges'].mean()
sns.countplot(x=data["sex"])
plt.show()

'''
Sex:-
   Average Charges:
   • Male   : 13956.75
   • Female : 12569.58

   Insight:
   Average charges are relatively close compared to smoker.
   This suggests smoking may be a much stronger predictor.
'''

data.groupby("smoker")["charges"].mean()
sns.countplot(x=data["smoker"])
plt.show()

'''
Smoker:-
   Average Charges:
   • Non-Smoker : 8434.27
   • Smoker     : 32050.23

   Insight:
   Smoking has a very strong effect on insurance cost.
   Even before training a model, this feature looks highly
   informative.
'''

"""
=========================================================
Box Plot

Detect possible outliers.

Never remove outliers just by looking at a boxplot.
Always verify whether they are genuine values.
=========================================================
"""
for col in numeric_columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=data[col])
    plt.tight_layout()
    plt.show()


"""
=========================================================
Heatmap

corr()              -> Correlation matrix
numeric_only=True   -> Only numerical columns
annot=True          -> Show correlation values

Goal:
Find relationships between numerical features.
=========================================================
"""
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title("Correlation between Features")
plt.tight_layout()
plt.show()

'''
Correlation Heatmap:-

   Strongest numerical correlations with 'charges':
   charges     1.000000
   age         0.299008
   bmi         0.198341
   children    0.067998

   Insight:
   Correlation only measures linear relationships.
   A feature with low correlation can still become important
   when combined with other features.
'''


# ==========================================================
# DATA CLEANING
# ==========================================================

# Dataset shape before removing duplicates
# data.shape

"""
=========================================================
drop_duplicates()

Removes exactly identical rows.

Why?
Duplicate records may bias the model by repeating
the same information multiple times.
=========================================================
"""
data.drop_duplicates(inplace=True)

# Dataset shape after cleaning
# data.shape


# ==========================================================
# LABEL ENCODING
# ==========================================================

"""
=========================================================
Always inspect categories before encoding.

Reason:
If dataset contains
Male / male / MALE

encoding directly will create incorrect mapping.
=========================================================
"""
data["sex"].value_counts()

"""
=========================================================
map()

Replaces old values with new ones.

male   -> 0
female -> 1

Numbers are only representations.
They do NOT imply ranking.
=========================================================
"""
data["sex"] = data["sex"].map({"male":0,"female":1})

data["smoker"].value_counts()

data["smoker"] = data["smoker"].map({"yes":1,"no":0})


# Better column names
data.rename(
    columns={
        "sex":"is_female",
        "smoker":"is_smoker"
    },
    inplace=True
)

data.head()


# ==========================================================
# ONE HOT ENCODING
# ==========================================================

data["region"].value_counts()

"""
=========================================================
get_dummies()

Creates separate binary columns.

Why?
Regions have no natural order.

drop_first=True
Avoids Dummy Variable Trap.
(We'll study this later.)
=========================================================
"""
data = pd.get_dummies(
    data,
    columns=["region"],
    drop_first=True
)

data = data.astype(int)

data.head()

"""
Why is 'region_northeast' missing?

Because we used:

drop_first=True

Pandas removes the first category and treats it as the
baseline (reference category).

If all remaining region columns are 0,

it automatically means the sample belongs to
'Northeast'.

This avoids the Dummy Variable Trap.

We'll study this concept in depth later.
"""


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

sns.histplot(data["bmi"], kde=True)
plt.show()

"""
=========================================================
pd.cut()

Converts continuous values into categories.

BMI becomes

• Underweight
• Normal
• Overweight
• Obese

This is an example of Feature Engineering using
domain knowledge.
=========================================================
"""
data["bmi_category"] = pd.cut(
    data["bmi"],
    bins=[0,18.5,24.9,29.9,float("inf")],
    labels=[
        "Underweight",
        "Normal",
        "Overweight",
        "Obese"
    ]
)

data = pd.get_dummies(
    data,
    columns=["bmi_category"],
    drop_first=True
)

data = data.astype(int)

data.head()

# ==========================================================
# FEATURE SCALING
# ==========================================================

"""
=========================================================
Feature Scaling

Why?

Different numerical features usually have different ranges.

Example:

Age      -> 18 to 64
BMI      -> 15 to 53
Charges  -> 1,000 to 63,000

Some Machine Learning algorithms (KNN, SVM, K-Means,
Neural Networks, etc.) are distance-based.

A feature having larger values can dominate the learning
process.

So, we bring all numerical features to a similar scale.

In this project, we'll use Standardization.

(We'll study Standardization vs Normalization in depth later.)
=========================================================
"""

# Numerical columns to scale
scale_columns = ["age", "bmi", "children"]

"""
=========================================================
StandardScaler()

Transforms data such that

• Mean ≈ 0
• Standard Deviation ≈ 1

fit_transform()

fit()
→ Learns Mean & Standard Deviation from data.

transform()
→ Uses those learned values to scale the data.

fit_transform()
→ Performs both together.
=========================================================
"""

scaler = StandardScaler()

data[scale_columns] = scaler.fit_transform(data[scale_columns])

# Let's verify the result
data.head()


"""
=========================================================
Engineer Thinking

Notice that only

• age
• bmi
• children

are scaled.

Why not 'charges'?

Because 'charges' is our Target Variable.
Currently we're only preparing the input features.

Also notice that binary columns like

is_female
is_smoker

are NOT scaled.

Reason:
They already have values between 0 and 1, so scaling
them usually doesn't provide any benefit.
=========================================================
"""

# print(data.columns)

# ============================================================
# Project 1 - Part 3
# Feature Selection (Annotated)
# ============================================================

"""
=========================================================
CONCEPT 1 : Pearson Correlation

Question:
Har numerical feature target ('charges') ke saath
kitna related hai?

Pearson Correlation isi question ka answer deta hai.

Range:
+1  -> Strong Positive Relation
 0  -> No Linear Relation
-1  -> Strong Negative Relation

Hum mathematics nahi padh rahe.
Bas relation measure kar rahe hain.

Goal:
Useful numerical features identify karna.
=========================================================
"""

# -----------------------------
# Pearson Correlation Code
# -----------------------------

selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight',
    'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(data[feature], data['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)

correlation_df = correlation_df.sort_values(
    by='Pearson Correlation',
    ascending=False
)

"""
-----------------------------
CODE EXPLANATION

selected_features
→ Jin features ko test karna hai unki list.

Dictionary Comprehension
→ Har feature uthao.
→ Pearson Correlation nikalo.
→ Dictionary me store karo.

pearsonr(x, y)
→ x aur y ka correlation nikalta hai.

[0]
→ Correlation Value
[1]
→ P-value
Hume sirf correlation chahiye, isliye [0].

items()
→ Dictionary ko (Key, Value) pairs me convert karta hai.

list()
→ Un pairs ko list bana deta hai.

pd.DataFrame()
→ List ko table bana deta hai.

columns=[]
→ Output table ke column names.

sort_values()
→ Correlation ke basis par sorting.

ascending=False
→ Highest correlation sabse upar.

Engineer Thinking:
Correlation sirf ranking de raha hai.
Decision abhi final nahi hai.
"""

"""
=========================================================
CONCEPT 2 : Chi-Square Test

Pearson sirf numerical relationship check karta hai.

Ab categorical features ko check karna hai.

Question:

"Kya ye category target se related hai?"

Isi liye Chi-Square Test use karte hain.

Goal:
Useful categorical features identify karna.
=========================================================
"""

# -----------------------------
# Chi Square Code
# -----------------------------

cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast',
    'region_southwest',
    'bmi_category_Normal',
    'bmi_category_Overweight',
    'bmi_category_Obese'
]

alpha = 0.05

data['charges_bin'] = pd.qcut(
    data['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:

    contingency = pd.crosstab(
        data[col],
        data['charges_bin']
    )

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = (
        'Reject Null (Keep Feature)'
        if p_val < alpha
        else 'Accept Null (Drop Feature)'
    )

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T

chi2_df = chi2_df.sort_values(by='p_value')

"""
-----------------------------
CODE EXPLANATION

cat_features
→ Test karne wale categorical features.

alpha = 0.05
→ Decision threshold.

qcut()
→ Numerical charges ko 4 equal groups me divide karta hai.

q=4
→ Total 4 bins.

labels=False
→ Bin names ki jagah 0,1,2,3.

crosstab()
→ Frequency table banata hai.
Chi-Square ko ye format chahiye.

chi2_contingency()
Returns:
1. Chi2 Statistic
2. P-value
3. Degrees of Freedom
4. Expected Frequency

_, _
→ In values ko ignore kar rahe hain.

p_val < alpha
→ Feature useful.

Else
→ Drop consider kar sakte hain.

.T
→ Rows aur Columns interchange.

Engineer Thinking:
Yahan statistics ka goal nahi hai.
Goal hai:
"Kaunsa categorical feature model ko dena chahiye?"
"""

"""
=========================================================
CONCEPT 3 : Final Feature Selection

Ab hamare paas teen sources hain:

1. EDA
2. Pearson Correlation
3. Chi-Square

In sabko combine karke final features choose karte hain.

Yehi Feature Selection ka final output hota hai.
=========================================================
"""

# -----------------------------
# Final Dataset
# -----------------------------

final_df = data[
    [
        'age',
        'is_female',
        'bmi',
        'children',
        'is_smoker',
        'charges',
        'region_southeast',
        'bmi_category_Obese'
    ]
]

# print(final_df)

"""
=========================================
PROJECT 1 HALF COMPLETED:
✔ EDA
✔ Data Cleaning
✔ Data Preprocessing
✔ Feature Engineering
✔ Feature Selection

The dataset is now ready for model training.

Note:
Feature Selection is not fixed. During model building,
we may add or remove features based on model performance
to improve accuracy and generalization.

Next Step → Model Training
=========================================
"""


# ============================================================
# Part 2 - Train-Test Split & Model Training
# ============================================================

# 'charges' ko predict karna hai, isliye ise target (y) banaya.
# Baaki saare columns input features (X) hain.
X = final_df.drop('charges', axis=1)
y = final_df['charges']

# Dataset ko 80% training aur 20% testing me divide kiya.
# Training data se model seekhega aur testing data se uski performance check hogi.
# random_state=42 dene se har baar same data split milega.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split hone ke baad har dataset ka size dekhne ke liye.
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Linear Regression algorithm ka object banaya.
# Abhi model sirf create hua hai, isne kuch seekha nahi hai.
model = LinearRegression()

# Training data (X_train, y_train) dekar model ko train kiya.
# Ab model input features aur charges ke beech relationship seekh chuka hai.
model.fit(X_train, y_train)

# ============================================================
# Part 3 - Measuring Performance Metrics of the Model
# ============================================================

# Test data (X_test) par prediction karna
# Model har test sample ke liye predicted value return karega
y_pred = model.predict(X_test)

# R² Score calculate karna
# Ye batata hai ki model actual data ko kitna achhe se explain kar raha hai.
# R² = 1  -> Perfect model
# R² = 0  -> Average performance
# R² < 0  -> Poor model
r2 = r2_score(y_test, y_pred)

# Test dataset me total kitne samples (rows) hain
n = X_test.shape[0]

# Total independent variables (features/columns) kitni hain
p = X_test.shape[1]

# Adjusted R² Score calculate karna
# Ye R² ka improved version hai.
# Agar model me unnecessary features add kiye gaye hain,
# to Adjusted R² unke liye penalty lagata hai.
#
# Formula:
# Adjusted R² = 1 - ((1 - R²) * (n - 1) / (n - p - 1))
#
# jahan:
# n = total samples
# p = total features
# R² = normal R² score
adjusted_r2_score = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Adjusted R² Score print karne ke liye
# print(adjusted_r2_score)

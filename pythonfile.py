#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Set plot style
sns.set(style="whitegrid")

#Load the Dataset
df = pd.read_csv(r"C:\Users\YADVENDRA\Desktop\Trending_Movies_Metadata.csv")

#Basic Info
print("Dataset Shape:", df.shape)
print("Column Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

#EDA Distributions
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['budget'].dropna(), bins=30, kde=True)
plt.title('Budget Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['revenue'].dropna(), bins=30, kde=True)
plt.title('Revenue Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['vote_average'].dropna(), bins=30, kde=True)
plt.title('Vote Average Distribution')
plt.tight_layout()
plt.show()

#EDA - Vote Count vs Rating
plt.figure(figsize=(8, 5))
sns.scatterplot(x='vote_count', y='vote_average', data=df)
plt.title("Vote Count vs Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Average Rating")
plt.show()

#Movies Released per Year
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
plt.figure(figsize=(12, 5))
df['release_year'].value_counts().sort_index().plot(kind='bar')
plt.title("Movies Released Each Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Feature Selection for Regression
selected_columns = [
    'budget', 'revenue', 'vote_average',
    'vote_count', 'runtime', 'release_year', 'language'
]

df_model = df[selected_columns + ['popularity']].copy()

# Handle missing values
df_model = df_model.fillna(0)

# Encode original_language
le = LabelEncoder()
df_model['language'] = le.fit_transform(df_model['language'])

#Defining Features and Target
X = df_model.drop('popularity', axis=1)
y = df_model['popularity']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

#Train Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict & Evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nw Model Evaluation:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

#Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Popularity")
plt.tight_layout()
plt.show()

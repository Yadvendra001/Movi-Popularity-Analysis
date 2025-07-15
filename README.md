# Trending Movies Data Analysis and Popularity Prediction

This project analyzes trending movie data and builds a regression model to predict a movie's popularity using Python and Power BI.

#Dataset
The dataset Trending_Movies_Metadata.csv contains metadata about trending movies, including:

* Budget
* Revenue
* Vote Average & Vote Count
* Genres
* Release Date
* Language
* Popularity Score (target)

#Objectives:
* Analyze movie trends using visualizations (EDA)
* Select key features that impact popularity
* Build a linear regression model to predict movie popularity
* Visualize insights using Power BI

# Tools & Libraries Used
* Python 3.11+
* pandas, numpy, matplotlib, seaborn
* scikit-learn (for ML model)
* Power BI (for dashboard visualization)

#Exploratory Data Analysis (EDA)
Performed visual analysis to understand the distribution and relationships between features:
* Budget Distribution
* Revenue Distribution
* Vote Average Histogram
* Vote Count vs Rating (Scatter Plot)
* Number of Movies Released Each Year

# Feature Selection for Regression
Selected key numerical features for training the model.
* Language column encoded using LabelEncoder
* Target variable: popularity

# Regression Model (Linear Regression):
Steps:
1. Handle missing values (fillna)
2. Encode categorical data (language)
3. Split dataset into training and test sets (60:40)
4. Train LinearRegression model
5. Evaluate using:
   * R² Score
   * RMSE (Root Mean Squared Error)

#Power BI Dashboard 
* Used cleaned dataset to build an interactive dashboard
* Features included:
  * Filters (Year, Language, Genre, Rating)
  * KPI Cards (Budget, Revenue, Ratings)
  * Graphs (Scatter, Bar, Line, Column)

# Key Learnings
* How to clean and analyze real-world movie data
* Importance of EDA before modeling
* Built a working regression model to predict popularity
* Built a business-ready Power BI dashboard to present insights

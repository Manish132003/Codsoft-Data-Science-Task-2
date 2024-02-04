import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your movie dataset
# Replace 'your_dataset.csv' with the actual file path or URL of your dataset
# The dataset should contain features like genre, director, actors, and the movie rating
movie_data = pd.read_csv('your_dataset.csv')

# Assuming 'genre', 'director', and 'actors' are categorical features
categorical_features = ['genre', 'director', 'actors']
numeric_features = ['other_numeric_features']  # Add other numeric features if available

# Define a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Combine preprocessing with a linear regression model in a pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into features (X) and target variable (y)
X = movie_data.drop('rating', axis=1)
y = movie_data['rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

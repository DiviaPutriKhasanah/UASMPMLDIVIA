import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import math
import streamlit as st

# Load dataset
df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\onlinefoods.csv")

# Display basic information and statistics
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualizations
sns.countplot(x='Output', data=df)
plt.title('Distribution of Output')
plt.show()

sns.boxplot(x='Output', y='Age', data=df)
plt.title('Age Distribution by Output')
plt.show()

# Select only numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numeric_columns].corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Encode categorical variables
categorical_features = ['Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Feedback']
numerical_features = ['Age', 'Family size', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the dataset into training and testing sets
X = df.drop('Output', axis=1)
y = df['Output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)

# Train the model
logreg_model.fit(X_train, y_train_encoded)

# Predict on test set
y_pred_encoded = logreg_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_encoded, y_pred_encoded)
mse = mean_squared_error(y_test_encoded, y_pred_encoded)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_encoded, y_pred_encoded)

# Print evaluation metrics
print('Logistic Regression Metrics =')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}\n')

# Define models
models = {
    'Logistic Regression': logreg_model,
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate models using cross-validation
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring='accuracy')
    results[name] = cv_scores
    print(f'{name} Cross-Validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})')

# Train and evaluate models on the test set
for name, model in models.items():
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f'{name} Accuracy: {accuracy:.2f}')
    print(classification_report(y_test_encoded, y_pred))

# Plot model performance
plt.figure(figsize=(10, 5))
plt.boxplot(results.values(), labels=results.keys())
plt.title('Model Comparison')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

# Train the best model
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train_encoded)
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Best Model = Random Forest \nRandom Forest Accuracy: {accuracy:.2f}')
print(classification_report(y_test_encoded, y_pred))

# Save model and preprocessor
joblib.dump(best_model, 'random_forest_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Streamlit app
st.title('Prediksi Output untuk Online Foods')

# Input features
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
occupation = st.selectbox('Occupation', ['Employee', 'House wife', 'Self Employeed', 'Student'])
monthly_income = st.selectbox('Monthly Income', ['Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000', 'No Income'])
educational_qualifications = st.selectbox('Educational Qualifications', ['School', 'Graduate', 'Post Graduate', 'Ph.D', 'Uneducated'])
feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
age = st.number_input('Age', min_value=0)
family_size = st.number_input('Family Size', min_value=0)
latitude = st.number_input('Latitude')
longitude = st.number_input('Longitude')

# Create DataFrame from inputs
user_input = pd.DataFrame({
    'Gender': [gender],
    'Marital Status': [marital_status],
    'Occupation': [occupation],
    'Monthly Income': [monthly_income],
    'Educational Qualifications': [educational_qualifications],
    'Feedback': [feedback],
    'Age': [age],
    'Family size': [family_size],
    'latitude': [latitude],
    'longitude': [longitude]
})

# Button to make prediction
if st.button('Predict'):
    try:
        # Apply preprocessing
        user_input_processed = preprocessor.transform(user_input)
        
        # Make prediction
        prediction = best_model.predict(user_input_processed)
        prediction_proba = best_model.predict_proba(user_input_processed)
        
        # Display prediction
        st.write('### Hasil Prediksi')
        st.write(f'Output Prediksi: {prediction[0]}')
        st.write(f'Probabilitas Prediksi: {prediction_proba[0]}')
    except ValueError as e:
        st.error(f"Error during preprocessing: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
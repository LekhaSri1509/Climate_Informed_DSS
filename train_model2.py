import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

# Function for Exploratory Data Analysis (EDA)
def perform_eda(data, target):
    print("First few rows of the data:")
    print(data.head())
    
    print("\nData types and summary statistics:")
    print(data.info())
    print(data.describe(include='all'))
    
    # Missing data visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

    # Distribution of target variable
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=data)
    plt.title(f'Distribution of {target}')
    plt.show()

    # Filter only numeric columns for the correlation heatmap
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    # Histograms for numerical features
    data.hist(figsize=(12, 10), bins=20, color='teal')
    plt.tight_layout()
    plt.show()

# Function for training the disaster management model
def train_disaster_management_model(data, features, target, model_filename, scaler_filename):
    # Perform EDA
    perform_eda(data, target)
    
    # Handle missing values
    data = data.fillna(data.median(numeric_only=True))
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    
    categorical_columns = ['Disaster Type', 'Supply Chain Issues', 'Resource Allocation']
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    
    # Encode the target variable if necessary
    if data[target].dtype == 'object':
        data[target] = label_encoder.fit_transform(data[target])
    
    # Split data into features and target
    X = data[features]
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    except ValueError as e:
        print(f"SMOTE error: {e}")
        print("Using original training data without SMOTE.")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train a classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the classifier
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy for {model_filename}: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save the model and scaler to disk
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model {model_filename} trained and saved successfully.")

# Load the disaster_management_data.csv file
file_path = 'disaster_management_data.csv'
data = pd.read_csv(file_path)

# Updated feature list based on your dataset
features = [
    'Disaster Type', 'Supply Chain Issues', 'Resource Allocation',
    'Duration of Disaster (hours)', 'Total Population Affected',
    'Number of Populations Survived', 'Amount of Food Supplies (kg)',
    'Food Distribution Quantity (kg)', 'Delivery Times (hours)', 
    'Response Time (hours)'
]
target = 'Severity Level'

# Train the model and save it
train_disaster_management_model(
    data,
    features,
    target,
    model_filename='disaster_management_model.pkl',
    scaler_filename='disaster_management_scaler.pkl'
)

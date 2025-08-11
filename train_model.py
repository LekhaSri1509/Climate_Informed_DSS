import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import pickle

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

    # Boxplots to detect outliers
    # for col in numeric_data.columns:
    #     plt.figure(figsize=(6, 4))
    #     sns.boxplot(x=data[col])
    #     plt.title(f'Boxplot of {col}')
    #     plt.show()


def train_disaster_model(data, features, target, model_filename, scaler_filename):
    # Perform EDA
    perform_eda(data, target)
    
    # Handle missing values
    data = data.fillna(data.median(numeric_only=True))
    
    # Encode the target variable if necessary
    if data[target].dtype == 'object':
        le = LabelEncoder()
        data[target] = le.fit_transform(data[target])
    
    # Split data into features and target
    X = data[features]
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check if target is categorical or continuous
    if y_train.dtype == 'int' or y_train.dtype == 'object':
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
    
    else:
        # Train a regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the regressor
        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import mean_squared_error, r2_score
        print(f"R^2 score for {model_filename}: {r2_score(y_test, y_pred)}")
        print(f"Mean Squared Error for {model_filename}: {mean_squared_error(y_test, y_pred)}")
    
    # Save the model and scaler to disk
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Model {model_filename} trained and saved successfully.")

# Load and process each dataset, then train the model
datasets = {
    'earthquake': {
        'file': 'earthquake.csv',
        'features': ['magnitude', 'depth', 'latitude', 'longitude', 'dmin', 'gap', 'nst'],
        'target': 'alert',
        'model_filename': 'earthquake_model.pkl',
        'scaler_filename': 'earthquake_scaler.pkl'
    },
    'flood': {
        'file': 'flood.csv',
        'features': ['MonsoonIntensity', 'Urbanization', 'ClimateChange'],
        'target': 'FloodProbability',  # This is numerical
        'model_filename': 'flood_model.pkl',
        'scaler_filename': 'flood_scaler.pkl'
    },
    'tornado': {
        'file': 'tornadoes.csv',
        'features': ['magnitude', 'state', 'injuries', 'fatalities'],
        'target': 'state',
        'model_filename': 'tornado_model.pkl',
        'scaler_filename': 'tornado_scaler.pkl'
    },
    'tsunami': {
        'file': 'tsunami.csv',
        'features': ['EQ_MAGNITUDE','EQ_DEPTH','TS_INTENSITY'],
        'target': 'EVENT_VALIDITY',
        'model_filename': 'tsunami_model.pkl',
        'scaler_filename': 'tsunami_scaler.pkl'
    }
}

# Iterate through each dataset and train models
for disaster, config in datasets.items():
    print(f"Training model for {disaster}...")
    data = pd.read_csv(config['file'])
    train_disaster_model(data, config['features'], config['target'], config['model_filename'], config['scaler_filename'])

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import check_X_y

# Set page config
st.set_page_config(page_title="Bank Marketing Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-full.csv")
    return df

df = load_data()

# Data preprocessing
def preprocess_data(df):
    # Convert target to binary
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Select features
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'contact', 'month']
    
    # Preprocessing
    X = df[numerical_features + categorical_features]
    y = df['y']
    
    # One-hot encoding
    encoder = OneHotEncoder(drop='first')
    X_cat = encoder.fit_transform(X[categorical_features])
    X_cat = pd.DataFrame(X_cat.toarray(), columns=encoder.get_feature_names_out(categorical_features))
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X[numerical_features]), columns=numerical_features)
    
    # Combine features
    X_processed = pd.concat([X_num, X_cat], axis=1)
    
    return X_processed, y, encoder, scaler, numerical_features, categorical_features

# Train model
def train_model(X, y, model_name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, class_weight='balanced')
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
    elif model_name == 'Gradient Boosting':
        # Gradient Boosting doesn't support class_weight, so we use sample_weight
        model = GradientBoostingClassifier(random_state=42)
        sample_weight = np.where(y_train == 1, 
                               len(y_train[y_train == 0]) / len(y_train[y_train == 1]), 
                               1)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, X_test, y_test
    
    # Train model
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# User input form
def user_input_features(numerical_features, categorical_features):
    st.sidebar.header("Client Information")
    
    # Numerical inputs
    inputs = {}
    for feature in numerical_features:
        if feature == 'age':
            inputs[feature] = st.sidebar.slider('Age', 18, 95, 30)
        elif feature == 'balance':
            inputs[feature] = st.sidebar.number_input('Account Balance', min_value=-8000, max_value=100000, value=1000)
        elif feature == 'duration':
            inputs[feature] = st.sidebar.number_input('Last Contact Duration (seconds)', min_value=0, max_value=5000, value=180)
        elif feature == 'campaign':
            inputs[feature] = st.sidebar.slider('Number of Contacts', 1, 50, 2)
        elif feature == 'pdays':
            inputs[feature] = st.sidebar.number_input('Days Since Last Contact', min_value=-1, max_value=900, value=-1)
        elif feature == 'previous':
            inputs[feature] = st.sidebar.slider('Previous Contacts', 0, 50, 0)
    
    # Categorical inputs
    categorical_values = {
        'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                'management', 'retired', 'self-employed', 'services', 
                'student', 'technician', 'unemployed', 'unknown'],
        'marital': ['divorced', 'married', 'single'],
        'education': ['primary', 'secondary', 'tertiary', 'unknown'],
        'contact': ['cellular', 'telephone', 'unknown'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    }
    
    for feature in categorical_features:
        inputs[feature] = st.sidebar.selectbox(feature.capitalize(), categorical_values[feature])
    
    return pd.DataFrame(inputs, index=[0])

# Main function
def main():
    st.title("Bank Marketing Campaign Prediction")
    
    # Preprocess data
    X, y, encoder, scaler, numerical_features, categorical_features = preprocess_data(df)
    
    # Model selection
    model_name = st.sidebar.selectbox('Select Model', 
                                    ['Logistic Regression', 
                                     'Decision Tree', 
                                     'Random Forest', 
                                     'Gradient Boosting'])
    
    # Get user input
    user_data = user_input_features(numerical_features, categorical_features)
    
    # Train model
    model, X_test, y_test = train_model(X, y, model_name)
    
    # Make prediction
    if st.sidebar.button('Predict Subscription'):
        # Preprocess user input
        user_cat = encoder.transform(user_data[categorical_features])
        user_cat = pd.DataFrame(user_cat.toarray(), columns=encoder.get_feature_names_out(categorical_features))
        user_num = pd.DataFrame(scaler.transform(user_data[numerical_features]), columns=numerical_features)
        user_processed = pd.concat([user_num, user_cat], axis=1)
        
        # Prediction
        prediction = model.predict(user_processed)
        prediction_proba = model.predict_proba(user_processed)
        
        # Display results
        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.success("✅ This client is LIKELY to subscribe to a term deposit")
        else:
            st.error("❌ This client is UNLIKELY to subscribe to a term deposit")
        
        st.write("")  # Spacer
        st.subheader("Prediction Confidence")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of subscribing", f"{prediction_proba[0][1]:.2%}")
        with col2:
            st.metric("Probability of not subscribing", f"{prediction_proba[0][0]:.2%}")
    
    # Model evaluation metrics
    st.subheader('Model Performance Metrics')
    y_pred = model.predict(X_test)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    with col4:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

if __name__ == '__main__':
    main()

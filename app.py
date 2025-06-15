import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
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
    
    # Handle class imbalance
    try:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    except Exception as e:
        st.warning(f"SMOTE not available: {e}. Using class weights instead.")
        class_weight = 'balanced'
    else:
        class_weight = None
    
    # Initialize model
    if model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, class_weight=class_weight)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42, class_weight=class_weight)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, class_weight=class_weight)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
    
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
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.success("This client is likely to subscribe to a term deposit")
        else:
            st.error("This client is unlikely to subscribe to a term deposit")
        
        st.subheader("Prediction Probability")
        st.write(f"Probability of subscribing: {prediction_proba[0][1]:.2%}")
        st.write(f"Probability of not subscribing: {prediction_proba[0][0]:.2%}")
    
    # Model evaluation
    st.subheader('Model Performance')
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
    
    # Confusion matrix
    st.subheader('Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Feature importance/coefficients
    if model_name == 'Logistic Regression':
        st.subheader('Logistic Regression Coefficients')
        coefficients = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coefficients, ax=ax)
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)
        
    elif model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
        st.subheader('Feature Importance')
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)

if __name__ == '__main__':
    main()

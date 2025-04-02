import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import ollama

# Load dataset from CSV
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_financial_data.csv")
    return df

# Data Preprocessing
def preprocess_data(df):
    # Drop non-numeric and unnecessary columns
    df.drop(columns=["CustomerID", "TransactionTimestamp", "TransactionType"], errors="ignore", inplace=True)

    # Fill numeric columns with mean values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df

# Model training and evaluation function
def train_and_evaluate(df):
    X = df.drop(columns=["Class"])  # Drop target variable
    y = df["Class"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models for evaluation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=800),
        "Support Vector Machine (SVM)": SVC(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
        "Support Vector Machine (SVM)": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        "K-Nearest Neighbors (KNN)": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        "Decision Tree": {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]},
        "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    }

    results = {}
    best_models = {}

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grid=param_grids[name], cv=3, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        y_pred = best_model.predict(X_test_scaled)
        results[name] = {
            "Best Parameters": grid_search.best_params_,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_pred)
        }

    return results, best_models, scaler, X

# Predict fraud using selected model
def predict_fraud(model, scaler, user_input, X):
    input_df = pd.DataFrame([user_input])

    # Drop any extra columns from user_input that are not used in training
    input_df = input_df[X.columns]  # Ensure input columns match training columns

    # Scale user input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    return "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"

# Perform EDA and generate insights
def eda_analysis(df):
    st.write("### 📊 EDA Insights for Fraudulent Patterns")

    # Correlation heatmap
    st.write("#### 🔥 Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

    # Distribution of target variable
    st.write("#### 🕵️ Class Distribution")
    class_dist = df["Class"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    class_dist.plot(kind='bar', color=["green", "red"], ax=ax)
    plt.title("Class Distribution (% of Fraud and Non-Fraud)")
    st.pyplot(fig)

    # Distribution of transaction amount
    st.write("#### 💸 Transaction Amount Distribution (Fraud vs Non-Fraud)")
    fig, ax = plt.subplots()
    sns.histplot(df[df["Class"] == 1]["TransactionAmount"], bins=50, color="red", label="Fraudulent", kde=True)
    sns.histplot(df[df["Class"] == 0]["TransactionAmount"], bins=50, color="green", label="Legitimate", kde=True)
    plt.legend()
    st.pyplot(fig)

    # Box plot of Credit Score
    st.write("#### 🧠 Box Plot: Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x="Class", y="CreditScore", data=df, palette=["green", "red"])
    st.pyplot(fig)

    # Key Insights
    st.write("### 🔍 Key EDA Insights:")
    st.markdown(""" 
    - 💡 **Higher Transaction Amounts** are more likely to be associated with fraud.
    - 🛑 **Credit Score Range:** Low credit scores increase the likelihood of fraudulent transactions.
    - 🚨 Fraudulent transactions tend to show **anomalies in account balances** and **portfolio risk scores.**
    - 📉 Legitimate transactions typically follow a more consistent distribution in terms of amount and account balance.
    """)

# Ask an Expert with Ollama (Gemma 2B)
def ask_ollama(conversation_history):
    try:
        response = ollama.chat(model="gemma:2b", messages=conversation_history)
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Streamlit Interface
def run_app():
    st.title("💸 Financial Risk Prediction and Fraud Detection")
    
    # Inject custom CSS
    st.markdown("""
        <style>
        body {
            background-color: #f7f8fa;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #00bfae;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #008e7a;
        }
        .stTitle {
            color: #008e7a;
        }
        .stMarkdown {
            color: #555555;
        }
        .stRadio>div {
            margin: 20px;
        }
        .stTextArea textarea {
            font-family: 'Arial', sans-serif;
            font-size: 14px;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    results, best_models, scaler, X = train_and_evaluate(df)

    # Initialize session state for conversation history if not already done
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Main options for user
    analysis_option = st.radio(
        "🔎 Select Analysis to View",
        [
            "✅ Assess credit risk before lending",
            "🚨 Detect early warning signals of financial instability",
            "🔒 Prevent fraud and high-risk transactions",
            "📈 Optimize portfolio and investment strategies",
            "💬 Ask an Expert (Ollama - Gemma 2B)",
            "📊 EDA Analysis for Fraud Patterns"
        ]
    )

    if analysis_option == "💬 Ask an Expert (Ollama - Gemma 2B)":
        st.write("🔍 **Ask an expert (Gemma 2B) to provide financial insights.**")

        # Display conversation history
        if st.session_state.conversation_history:
            for message in st.session_state.conversation_history:
                st.write(f"**User:** {message['content']}" if message["role"] == "user" else f"**Gemma 2B:** {message['content']}")

        # Text input for user to ask a question
        user_question = st.text_area("Type your question for Gemma 2B", key="question_input", height=200)

        # Button to ask question
        if user_question:
            if st.button("Ask Ollama"):
                # Add the user's question to the history
                st.session_state.conversation_history.append({"role": "user", "content": user_question})

                # Get Gemma 2B's response
                response = ask_ollama(st.session_state.conversation_history)

                # Add Gemma's response to the conversation history
                st.session_state.conversation_history.append({"role": "assistant", "content": response})

                # Reinitialize the text_area widget with a new key to reset it
                st.text_area("Type your question for Gemma 2B", key=f"question_input_{len(st.session_state.conversation_history)}", height=200)

                # Display the response
                st.write(f"**Gemma 2B:** {response}")

    elif analysis_option == "📊 EDA Analysis for Fraud Patterns":
        eda_analysis(df)

    elif analysis_option == "📈 Optimize portfolio and investment strategies":
        st.write("🔍 **Optimize Portfolio and Investment Strategies**")

        # Input fields to get more context about user's portfolio
        user_investment_question = st.text_area("Ask Gemma about Portfolio & Investment Strategies:", 
                                                key="investment_query", 
                                                placeholder="Ask about optimizing portfolio, risk management, etc.", 
                                                height=200)

        if user_investment_question:
            if st.button("Get Portfolio Advice"):
                # Add the user's investment question to the conversation history for context
                st.session_state.conversation_history.append({"role": "user", "content": user_investment_question})

                # Ask Gemma for advice based on the question
                response = ask_ollama(st.session_state.conversation_history)

                # Add Gemma's response to the conversation history
                st.session_state.conversation_history.append({"role": "assistant", "content": response})

                # Reinitialize the text_area widget with a new key to reset it
                st.text_area("Ask Gemma about Portfolio & Investment Strategies:", 
                             key=f"investment_query_{len(st.session_state.conversation_history)}", 
                             placeholder="Ask about optimizing portfolio, risk management, etc.", 
                             height=200)

                # Display the response from Gemma
                st.write(f"**Gemma 2B:** {response}")

    elif analysis_option == "✅ Assess credit risk before lending":
        st.write(f"### 🧠 Best Model Identified for {analysis_option}")

        # Automatically select the best-performing model
        best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
        st.write(f"✅ Using Best Model: **{best_model_name}**")
        model = best_models[best_model_name]

        # Get user input for model prediction
        st.write("⚡️ **Provide Transaction Details for Analysis:**")

        # Input fields for the model
        user_input = {
            "CreditScore": st.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=700),
            "TransactionAmount": st.number_input("Transaction Amount", min_value=10.0, max_value=10000.0, value=1000.0),
            "AccountBalance": st.number_input("Account Balance", min_value=100.0, max_value=50000.0, value=5000.0),
            "PortfolioRiskScore": st.number_input("Portfolio Risk Score (0-100)", min_value=0.0, max_value=100.0, value=50.0),
            "NumPreviousDefaults": st.number_input("Number of Previous Defaults", min_value=0, max_value=5, value=0),
            "CustomerIncome": st.number_input("Customer Income", min_value=20000.0, max_value=200000.0, value=80000.0),
            "InvestmentRatio": st.number_input("Investment Ratio (0.05 - 0.5)", min_value=0.05, max_value=0.5, value=0.12),
        }

        # Button to detect result
        if st.button("🎯 Detect Result"):
            prediction = predict_fraud(model, scaler, user_input, X)
            if prediction == "Legitimate Transaction":
                st.success("### ✅ **This is a Legitimate Transaction.** The credit risk is low, and the lending decision can proceed.")
            else:
                st.warning("### 🚨 **This is a Fraudulent Transaction.** The credit risk is high, and the lending decision should be reconsidered.")
    
    else:
        st.write("### ⚡ View Model Performance")
        results_df = pd.DataFrame(results).T
        st.write(results_df)

if __name__ == '__main__':
    run_app()

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import load_data, clean_data
from utils.visualization import (
    plot_age_distribution,
    plot_diagnosis_pie,
    plot_correlation,
    plot_mmse_distribution
)
from utils.classification import (
    preprocess_data_for_classification,
    train_model,
    evaluate_model
)
from utils.filters import add_filters
from utils.patient_profile import display_patient_profile

# Define the categorical columns in the dataset
categorical_cols = [
    'Gender',  # Patient's gender
    'Ethnicity',  # Ethnicity
    'EducationLevel',  # Education level
    'Smoking',  # Smoking status
    'FamilyHistoryAlzheimers',  # Family history of Alzheimer's
    'CardiovascularDisease',  # Cardiovascular disease
    'Diabetes',  # Diabetes
    'Depression',  # Depression
    'HeadInjury',  # Head injury
    'Hypertension',  # Hypertension
    'MemoryComplaints',  # Memory complaints
    'BehavioralProblems',  # Behavioral problems
    'Confusion',  # Confusion
    'Disorientation',  # Disorientation
    'PersonalityChanges',  # Personality changes
    'DifficultyCompletingTasks',  # Difficulty completing tasks
    'Forgetfulness'  # Forgetfulness
]

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Data Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main title
st.title("Alzheimer's Data Dashboard")

# Add dataset source link in the main page
st.markdown("### Dataset Source")
st.markdown("[Alzheimer's Disease Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)")

# Load and clean data
data = load_data('data/alzheimers_disease_data.csv')
data = clean_data(data)

# Sidebar: Filters (apply filters before mapping)
filtered_data = add_filters(data)

# Now, replace numeric codes with meaningful labels in filtered_data
# Mapping for Gender
gender_mapping = {0: 'Male', 1: 'Female'}
filtered_data['Gender'] = filtered_data['Gender'].map(gender_mapping)

# Mapping for Ethnicity
ethnicity_mapping = {
    0: 'Caucasian',
    1: 'African American',
    2: 'Asian',
    3: 'Other'
}
filtered_data['Ethnicity'] = filtered_data['Ethnicity'].map(ethnicity_mapping)

# Mapping for Education Level
education_mapping = {
    0: 'None',
    1: 'High School',
    2: 'Bachelor',
    3: 'Graduate'
}
filtered_data['EducationLevel'] = filtered_data['EducationLevel'].map(education_mapping)

# Update categorical_cols if necessary
categorical_cols = [col for col in categorical_cols if col in filtered_data.columns]

# Update numeric_cols
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = numeric_cols.drop('PatientID', errors='ignore')

# Create tabs
tabs = st.tabs([
    "Descriptive Statistics",
    "Correlations",
    "Individual Patient Profile",
    "Classification",
    "Prediction"
])

# Tab 1: Descriptive Statistics
with tabs[0]:
    st.header("📊 Descriptive Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        age_dist_fig = plot_age_distribution(filtered_data)
        st.plotly_chart(age_dist_fig, use_container_width=True)

    with col2:
        st.subheader("Diagnosis Distribution")
        diagnosis_pie_fig = plot_diagnosis_pie(filtered_data)
        st.plotly_chart(diagnosis_pie_fig, use_container_width=True)

    st.subheader("MMSE Distribution")
    mmse_fig = plot_mmse_distribution(filtered_data)
    st.plotly_chart(mmse_fig, use_container_width=True)

# Tab 2: Correlations
with tabs[1]:
    st.header("📈 Correlations")

    numeric_cols = filtered_data.select_dtypes(
        include=['float64', 'int64']
    ).columns

    x_var = st.selectbox(
        "X-axis Variable",
        options=numeric_cols,
        index=numeric_cols.get_loc('Age') if 'Age' in numeric_cols else 0
    )
    y_var = st.selectbox(
        "Y-axis Variable",
        options=numeric_cols,
        index=numeric_cols.get_loc('MMSE') if 'MMSE' in numeric_cols else 0
    )

    correlation_fig = plot_correlation(filtered_data, x_var, y_var)
    st.plotly_chart(correlation_fig, use_container_width=True)

# Tab 3: Individual Patient Profile
with tabs[2]:
    st.header("👤 Individual Patient Profile")

    patient_ids = filtered_data['PatientID'].unique()
    selected_patient_id = st.selectbox(
        "Select Patient ID",
        options=patient_ids
    )

    display_patient_profile(filtered_data, selected_patient_id)  # Use filtered_data here

# Tab 4: Classification
with tabs[3]:
    st.header("🔍 Binary Classification of Alzheimer's Diagnosis")

    # Preprocessing data (use original data without mapping)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data_for_classification(data)

    # Save training columns and numeric columns
    st.session_state['X_train_columns'] = X_train.columns.tolist()
    st.session_state['train_columns'] = X_train.columns.tolist()
    numeric_cols_train = X_train.select_dtypes(
        include=['float64', 'int64']
    ).columns
    st.session_state['numeric_cols'] = numeric_cols_train.tolist()

    # Model selection
    st.subheader("Model Selection")
    model_name = st.selectbox(
        "Choose a classification model",
        options=[
            "Logistic Regression",
            "Random Forest"
        ]
    )

    # Train the model
    if st.button("Train Model"):
        with st.spinner('Training model...'):
            model = train_model(X_train, y_train, model_name)
            acc, cm, report, roc_fig = evaluate_model(model, X_test, y_test)

            # Display metrics
            st.success(f"Model trained with accuracy of {acc*100:.2f}%")

            st.subheader("Model Metrics")
            st.text(f"Accuracy : {acc:.2f}")
            st.text("Classification Report :")
            st.text(classification_report(y_test, model.predict(X_test)))

            st.subheader("Confusion Matrix")
            cm_fig = px.imshow(cm,
                               x=['Negative', 'Positive'],
                               y=['Negative', 'Positive'],
                               labels=dict(
                                   x="Predicted",
                                   y="Actual"
                               ),
                               text_auto=True)
            st.plotly_chart(cm_fig)

            st.subheader("ROC Curve")
            st.plotly_chart(roc_fig)

            # Save model in session
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
    else:
        st.info("Click 'Train Model' to start.")

# Tab 5: Prediction
with tabs[4]:
    st.header("🩺 Predict Diagnosis for a New Patient")

    # Check if the model has been trained
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning(
            "Please train the model in the ‘Classification’ tab before "
            "making predictions."
        )
    else:
        # Input patient features
        patient_features = {}
        for col in st.session_state['X_train_columns']:
            # Check column type to provide appropriate widget
            if col in categorical_cols:
                options = data[col].dropna().unique().tolist()
                patient_features[col] = st.selectbox(f"{col}", options=options)
            elif col in [
                'AlcoholConsumption',
                'PhysicalActivity',
                'DietQuality',
                'SleepQuality'
            ]:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                mean_val = float(data[col].mean())
                patient_features[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            else:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                mean_val = float(data[col].mean())
                patient_features[col] = st.slider(
                    f"{col}", min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )

        if st.button("Predict"):
            # Create a dataframe for the patient
            patient_df = pd.DataFrame([patient_features])

            # Preprocess patient data
            # Encode categorical variables
            for col in categorical_cols:
                if col in patient_df.columns:
                    le = LabelEncoder()
                    le.fit(data[col].astype(str))
                    patient_df[col] = le.transform(patient_df[col].astype(str))

            # Add missing columns with default values
            train_columns = st.session_state['train_columns']
            for col in train_columns:
                if col not in patient_df.columns:
                    patient_df[col] = 0  # or another appropriate default value

            # Reorder columns to match training order
            patient_df = patient_df[train_columns]

            # Get numeric columns
            numeric_cols = st.session_state['numeric_cols']

            # Normalize numeric data
            patient_df[numeric_cols] = st.session_state['scaler'].transform(
                patient_df[numeric_cols]
            )

            # Make prediction
            prediction = st.session_state['model'].predict(patient_df)
            proba = st.session_state['model'].predict_proba(patient_df)[:, 1]

            # Display result
            if prediction[0] == 1:
                st.error(
                    "The model predicts that the patient is **Positive** "
                    f"with a probability of {proba[0]*100:.2f}%."
                )
            else:
                st.success(
                    "The model predicts that the patient is **Negative** "
                    f"with a probability of {(1 - proba[0])*100:.2f}%."
                )

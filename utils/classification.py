from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go


def preprocess_data_for_classification(data):
    """
    Preprocess the data for binary classification.
    - Encode categorical variables
    - Split data into training and test sets
    - Normalize numeric data
    """
    df = data.copy()  # Copy the dataframe to avoid modifying the original

    # Define categorical variables
    categorical_cols = [
        'Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
        'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion',
        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
        'Forgetfulness'
    ]

    # Ensure the variables are of type string or category
    df[categorical_cols] = df[categorical_cols].astype(str)

    # Encode categorical variables using LabelEncoder
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Separate features (X) and target variable (y)
    X = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
    y = df['Diagnosis'].map({'Negative': 0, 'Positive': 1})

    # Handle missing values by filling them with 0
    X = X.fillna(0)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize numeric data
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, model_name):
    """
    Train a classification model based on the selected model name.
    - Supported models: Logistic Regression, Random Forest
    """
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(
            "Unsupported model: choose 'Logistic Regression' or 'Random Forest'"
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance on the test set.
    - Calculate accuracy, confusion matrix,
    classification report, and ROC curve
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create ROC curve plot
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        )
    )
    roc_fig.update_layout(
        title=f'ROC Curve (AUC = {roc_auc:.2f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )

    return acc, cm, report, roc_fig


def predict_new_patient(model, scaler, patient_data):
    """
    Predict the diagnosis for a new patient.
    - Encode and normalize patient data
    - Use the trained model to make predictions
    """
    # Preprocess patient data (assume it's already preprocessed for simplicity)
    y_pred = model.predict(patient_data)
    y_prob = model.predict_proba(patient_data)[:, 1]

    return y_pred, y_prob

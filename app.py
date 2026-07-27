"""
Airline Passenger Satisfaction Prediction App
==============================================
Interactive Streamlit app to demonstrate ML classification models

Features:
1. Upload CSV file
2. Select ML model
3. Display evaluation metrics
4. Show confusion matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1, h2, h3 {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD MODELS AND DATA
# ==============================================================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'K-Nearest Neighbor': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Random Forest': joblib.load('model/random_forest.pkl')
    }
    return models

@st.cache_resource
def load_scaler():
    """Load the scaler"""
    return joblib.load('model/scaler.pkl')

@st.cache_data
def load_metrics():
    """Load metrics from CSV"""
    return pd.read_csv('model/metrics.csv', index_col=0)

@st.cache_data
def load_test_data():
    """Load test data"""
    return pd.read_csv('test_data.csv')

# Load all resources
models = load_models()
scaler = load_scaler()
metrics_df = load_metrics()
test_data = load_test_data()

# ==============================================================================
# TITLE AND HEADER
# ==============================================================================

st.title("Airline Passenger Satisfaction Prediction")
st.markdown("""
This app demonstrates **5 ML classification models** trained to predict
whether airline passengers are satisfied or not.
""")

st.divider()

# ==============================================================================
# SIDEBAR: FILE UPLOAD AND MODEL SELECTION
# ==============================================================================

with st.sidebar:
    st.markdown("## Configuration Panel")
    st.markdown("---")

    # Data Source Section
    st.markdown("### Data Source")
    uploaded_file = st.file_uploader(
        "Upload a CSV file for predictions",
        type=['csv'],
        help="CSV file should have 'satisfaction' column for comparison"
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"File loaded: **{uploaded_file.name}**")
        st.caption(f"Rows: {len(data)} | Columns: {len(data.columns)}")
    else:
        data = test_data.copy()
        st.info("Using default test data")
        st.caption(f"Rows: {len(data)} | Columns: {len(data.columns)}")

    st.markdown("---")

    # Model Selection Section
    st.markdown("### Model Selection")

    model_descriptions = {
        'Logistic Regression': '📈 Linear classification model',
        'Decision Tree': '🌳 Tree-based non-linear model',
        'K-Nearest Neighbor': '🔍 Instance-based learning',
        'Naive Bayes': '🎯 Probabilistic classifier',
        'Random Forest': '🎋 Ensemble of decision trees'
    }

    selected_model = st.selectbox(
        "Choose a model:",
        list(models.keys()),
        help="Select which ML model to use"
    )

    # Model info card
    st.markdown(f"""
    <div style="background-color: #1f77b4; padding: 15px; border-radius: 8px; color: white;">
        <p style="margin: 0; font-weight: bold; font-size: 16px;">{selected_model}</p>
        <p style="margin: 5px 0 0 0; font-size: 13px;">{model_descriptions.get(selected_model, '')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Stats
    st.markdown("### Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns))

# ==============================================================================
# MAIN CONTENT: TABS
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Metrics",
    "Predictions",
    "Confusion Matrix",
    "About"
])

# ==============================================================================
# TAB 1: DISPLAY EVALUATION METRICS (Feature 3)
# ==============================================================================

with tab1:
    st.header("Model Performance Metrics")

    # Get metrics for selected model
    model_metrics = metrics_df.loc[selected_model]

    # Display metrics in a grid layout
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.metric(
            label="Accuracy",
            value=f"{model_metrics['Accuracy']:.2%}",
            help="Overall correctness of predictions"
        )
        st.metric(
            label="AUC Score",
            value=f"{model_metrics['AUC']:.4f}",
            help="Class separation ability (0-1)"
        )

    with col2:
        st.metric(
            label="Precision",
            value=f"{model_metrics['Precision']:.4f}",
            help="True positives / All positives"
        )
        st.metric(
            label="Recall",
            value=f"{model_metrics['Recall']:.4f}",
            help="True positives / Actual positives"
        )

    with col3:
        st.metric(
            label="F1 Score",
            value=f"{model_metrics['F1']:.4f}",
            help="Harmonic mean of Precision & Recall"
        )
        st.metric(
            label="MCC Score",
            value=f"{model_metrics['MCC']:.4f}",
            help="Correlation coefficient (-1 to +1)"
        )

    st.markdown("---")

    # Explanation of metrics
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Accuracy**: Percentage of correct predictions overall
        - **AUC Score**: Area Under ROC Curve (0-1, higher is better)
        - **Precision**: Of positive predictions, how many were correct?
        - **Recall**: Of actual positives, how many did we find?
        - **F1 Score**: Harmonic mean of Precision and Recall
        - **MCC**: Matthews Correlation Coefficient (-1 to +1)
        """)

    # Display all metrics comparison
    st.subheader("Comparison with Other Models")
    st.dataframe(metrics_df, width='stretch')

# ==============================================================================
# TAB 2: MAKE PREDICTIONS
# ==============================================================================

with tab2:
    st.header("Make Predictions")

    st.markdown(f"**Model:** {selected_model}")

    # Separate features and target
    if 'satisfaction' in data.columns:
        X_data = data.drop('satisfaction', axis=1)
        y_data = data['satisfaction']
        has_labels = True
    else:
        X_data = data
        has_labels = False

    # Make a copy to avoid modifying original
    X_data = X_data.copy()

    # Encode categorical variables (same as training)
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    for col in categorical_columns:
        if col in X_data.columns:
            if col == 'Gender':
                X_data[col] = X_data[col].map({'Male': 0, 'Female': 1})
            elif col == 'Customer Type':
                X_data[col] = X_data[col].map({'Loyal Customer': 0, 'disloyal Customer': 1})
            elif col == 'Type of Travel':
                X_data[col] = X_data[col].map({'Personal Travel': 0, 'Business travel': 1})
            elif col == 'Class':
                X_data[col] = X_data[col].map({'Eco': 0, 'Business': 1, 'Eco Plus': 2})

    X_data = X_data.fillna(0)
    X_scaled = scaler.transform(X_data)

    model = models[selected_model]
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Display predictions summary with better formatting
    st.markdown("### Prediction Summary")

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.metric("Total Samples", f"{len(predictions):,}")
    with col2:
        satisfied_count = int((predictions == 1).sum())
        satisfied_pct = (satisfied_count / len(predictions)) * 100
        st.metric("Satisfied", satisfied_count, f"{satisfied_pct:.1f}%")
    with col3:
        not_satisfied_count = int((predictions == 0).sum())
        not_satisfied_pct = (not_satisfied_count / len(predictions)) * 100
        st.metric("Not Satisfied", not_satisfied_count, f"{not_satisfied_pct:.1f}%")

    st.markdown("---")

    # Display predictions table
    st.subheader("Detailed Predictions")
    results_df = pd.DataFrame({
        'Satisfied_Probability': probabilities[:, 1],
        'Not_Satisfied_Probability': probabilities[:, 0],
        'Prediction': ['Satisfied' if p == 1 else 'Not Satisfied' for p in predictions]
    })

    if has_labels:
        # Encode y_data for comparison
        y_data_for_comparison = y_data.copy()

        try:
            # Check if already numeric or string
            if y_data_for_comparison.dtype == 'object' or y_data_for_comparison.dtype == 'string':
                # String type - convert to lowercase and check if contains 'satisfied'
                y_data_for_comparison = y_data_for_comparison.astype(str).str.lower().str.contains('satisfied').astype(int)
            else:
                # Already numeric - fill NaN and convert to int
                y_data_for_comparison = y_data_for_comparison.fillna(0).astype(int)

            results_df['Actual'] = ['Satisfied' if p == 1 else 'Not Satisfied' for p in y_data_for_comparison]
            results_df['Correct'] = (predictions == y_data_for_comparison)
        except Exception as e:
            st.warning(f"Could not compare with actual values: {str(e)}")

    st.dataframe(results_df, width='stretch')

# ==============================================================================
# TAB 3: CONFUSION MATRIX (Feature 4)
# ==============================================================================

with tab3:
    st.header("Confusion Matrix & Classification Report")

    st.info("Confusion Matrix shows how the model made predictions. Accuracy and F1-Score are from Model Performance Metrics tab (same evaluation dataset).")

    # Check if we have labels
    if 'satisfaction' in data.columns:
        X_data = data.drop('satisfaction', axis=1)
        y_data = data['satisfaction']

        # Make a copy to avoid modifying original
        X_data = X_data.copy()

        # Encode categorical variables (same as training)
        categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col in categorical_columns:
            if col in X_data.columns:
                # Map values using the same mapping as training
                if col == 'Gender':
                    X_data[col] = X_data[col].map({'Male': 0, 'Female': 1})
                elif col == 'Customer Type':
                    X_data[col] = X_data[col].map({'Loyal Customer': 0, 'disloyal Customer': 1})
                elif col == 'Type of Travel':
                    X_data[col] = X_data[col].map({'Personal Travel': 0, 'Business travel': 1})
                elif col == 'Class':
                    X_data[col] = X_data[col].map({'Eco': 0, 'Business': 1, 'Eco Plus': 2})

        # Fill any NaN values from encoding
        X_data = X_data.fillna(0)

        # Encode y_data (actual labels) to match predictions
        y_data_encoded = y_data.copy()

        try:
            # Check if already numeric or string
            if y_data_encoded.dtype == 'object' or y_data_encoded.dtype == 'string':
                # String type - convert to lowercase and check if contains 'satisfied'
                y_data_encoded = y_data_encoded.astype(str).str.lower().str.contains('satisfied').astype(int)
            else:
                # Already numeric - fill NaN and convert to int
                y_data_encoded = y_data_encoded.fillna(0).astype(int)
        except Exception as e:
            st.error(f"Error encoding satisfaction column: {str(e)}")
            st.info("Please ensure the 'satisfaction' column contains either: '0'/'1' or 'Satisfied'/'Neutral or Dissatisfied'")
            st.stop()

        # Scale and predict
        X_scaled = scaler.transform(X_data)
        model = models[selected_model]
        predictions = model.predict(X_scaled)

        # Calculate confusion matrix
        cm = confusion_matrix(y_data_encoded, predictions)

        # Display sample count information
        st.markdown("### Sample Count Summary")
        total_samples = len(y_data_encoded)
        true_negatives = cm[0, 0]
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        true_positives = cm[1, 1]

        col_counts = st.columns(5)
        with col_counts[0]:
            st.metric("Total Samples", f"{total_samples:,}")
        with col_counts[1]:
            st.metric("True Negatives", f"{true_negatives:,}")
        with col_counts[2]:
            st.metric("False Positives", f"{false_positives:,}")
        with col_counts[3]:
            st.metric("False Negatives", f"{false_negatives:,}")
        with col_counts[4]:
            st.metric("True Positives", f"{true_positives:,}")

        st.markdown("---")

        # Plot confusion matrix
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create heatmap with white background for visibility
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Not Satisfied', 'Satisfied'],
                yticklabels=['Not Satisfied', 'Satisfied'],
                ax=ax,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 18, 'weight': 'bold'},
                linewidths=2,
                linecolor='white'
            )

            ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=15, fontweight='bold', pad=15)
            ax.tick_params(axis='both', labelsize=12)

            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")

            report = classification_report(
                y_data_encoded,
                predictions,
                target_names=['Not Satisfied', 'Satisfied'],
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()

            # Display accuracy metric from Model Performance Metrics (same dataset)
            stored_accuracy = model_metrics['Accuracy']
            stored_f1 = model_metrics['F1']

            col2a, col2b = st.columns([1, 1])
            with col2a:
                st.metric("Accuracy", f"{stored_accuracy:.2%}")
                st.caption("From Model Performance Metrics")
            with col2b:
                st.metric("F1-Score", f"{stored_f1:.4f}")
                st.caption("From Model Performance Metrics")

            st.dataframe(report_df, width='stretch')
    else:
        st.warning("Please upload a CSV file with 'satisfaction' column to see confusion matrix")

# ==============================================================================
# TAB 4: ABOUT
# ==============================================================================

with tab4:
    st.header("About This Project")

    st.markdown("""
    ### Project Overview
    This is an ML assignment that demonstrates end-to-end machine learning:
    - **Dataset**: Airline Passenger Satisfaction
    - **Task**: Binary Classification
    - **Models**: 5 different algorithms
    - **Metrics**: 6 evaluation metrics per model

    ### Models Implemented
    1. **Logistic Regression**: Linear classification model
    2. **Decision Tree**: Tree-based non-linear model
    3. **K-Nearest Neighbor**: Instance-based learning
    4. **Naive Bayes**: Probabilistic classifier
    5. **Random Forest**: Ensemble of decision trees

    ### Dataset Information
    - **Samples**: 129,880 airline passengers
    - **Features**: 22 (mix of numerical and categorical)
    - **Target**: Satisfaction (0 = Not Satisfied, 1 = Satisfied)
    - **Features Include**: Age, Flight Distance, Type of Travel, Cabin Class, etc.

    ### Evaluation Metrics
    - **Accuracy**: Overall correctness
    - **AUC Score**: Discriminative ability
    - **Precision**: False positive rate
    - **Recall**: False negative rate
    - **F1 Score**: Balance between precision and recall
    - **MCC**: Correlation coefficient

    ### Technologies Used
    - **Python**: Programming language
    - **Scikit-learn**: ML library
    - **Streamlit**: Web framework
    - **Pandas**: Data manipulation
    - **Matplotlib & Seaborn**: Visualization
    """)



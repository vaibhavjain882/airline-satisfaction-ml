"""
Airline Passenger Satisfaction Prediction App
==============================================
Interactive Streamlit app to demonstrate ML classification models
"""

import io
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Airline Satisfaction Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    h1, h2, h3 { font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    return {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'K-Nearest Neighbor': joblib.load('model/knn.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes.pkl'),
        'Random Forest': joblib.load('model/random_forest.pkl'),
    }


@st.cache_resource
def load_scaler():
    return joblib.load('model/scaler.pkl')


@st.cache_resource
def load_label_encoders():
    return joblib.load('model/label_encoders.pkl')


@st.cache_resource
def load_target_encoder():
    return joblib.load('model/label_encoder_target.pkl')


@st.cache_resource
def load_feature_columns():
    return joblib.load('model/feature_columns.pkl')


@st.cache_data
def load_metrics():
    return pd.read_csv('model/metrics.csv', index_col=0)


@st.cache_data
def load_test_data():
    return pd.read_csv('test_data.csv')


def preprocess_features(X_data, label_encoders, feature_columns):
    """Fast vectorized preprocessing matching model training."""
    X = X_data.copy()

    for col in ['Unnamed: 0', 'id']:
        if col in X.columns:
            X = X.drop(columns=col)

    missing_columns = [col for col in feature_columns if col not in X.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = X[feature_columns].copy()

    for col, encoder in label_encoders.items():
        if col not in X.columns:
            continue
        if not pd.api.types.is_numeric_dtype(X[col]):
            mapping = {
                str(cls): float(encoder.transform([cls])[0])
                for cls in encoder.classes_
            }
            X[col] = X[col].astype(str).str.strip().map(mapping).fillna(-1.0)

    return X.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)


def encode_target(y_data, target_encoder):
    """Encode target labels using the same encoder from training."""
    if not pd.api.types.is_numeric_dtype(y_data):
        mapping = {
            str(cls): int(target_encoder.transform([cls])[0])
            for cls in target_encoder.classes_
        }
        return (
            y_data.astype(str)
            .str.strip()
            .map(mapping)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

    return y_data.fillna(0).astype(int).to_numpy()


def scale_features(scaler, feature_matrix):
    X = np.asarray(feature_matrix, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


@st.cache_data(show_spinner=False)
def run_inference(csv_bytes, model_name, data_source_label):
    """Preprocess once and run the selected model (cached per file + model)."""
    if csv_bytes == b"__default__":
        df = load_test_data()
    else:
        df = pd.read_csv(io.BytesIO(csv_bytes))

    label_encoders = load_label_encoders()
    target_encoder = load_target_encoder()
    feature_columns = load_feature_columns()
    scaler = load_scaler()
    model = load_models()[model_name]

    has_labels = 'satisfaction' in df.columns
    X_raw = df.drop('satisfaction', axis=1) if has_labels else df

    X_features = preprocess_features(X_raw, label_encoders, feature_columns)
    X_scaled = scale_features(scaler, X_features)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    y_encoded = None
    accuracy = None
    f1 = None
    if has_labels:
        y_encoded = encode_target(df['satisfaction'], target_encoder)
        accuracy = float(accuracy_score(y_encoded, predictions))
        f1 = float(f1_score(y_encoded, predictions, average='weighted'))

    return {
        'row_count': len(df),
        'data_source': data_source_label,
        'has_labels': has_labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'y_encoded': y_encoded,
        'accuracy': accuracy,
        'f1': f1,
    }


# Load static resources
models = load_models()
metrics_df = load_metrics()
test_data = load_test_data()

st.title("Airline Passenger Satisfaction Prediction")
st.markdown(
    "This app demonstrates **5 ML classification models** trained to predict "
    "whether airline passengers are satisfied or not."
)
st.divider()

with st.sidebar:
    st.markdown("## Configuration Panel")
    st.markdown("---")
    st.markdown("### Data Source")

    uploaded_file = st.file_uploader(
        "Upload a CSV file for predictions",
        type=['csv'],
        help="Default uses test_data.csv (1000 rows). Upload data/test.csv for ~26k rows."
    )

    if uploaded_file is not None:
        csv_bytes = uploaded_file.getvalue()
        data_source = uploaded_file.name
    else:
        csv_bytes = b"__default__"
        data_source = "test_data.csv (default sample)"

    st.markdown("---")
    st.markdown("### Model Selection")

    model_descriptions = {
        'Logistic Regression': 'Linear classification model',
        'Decision Tree': 'Tree-based non-linear model',
        'K-Nearest Neighbor': 'Instance-based learning',
        'Naive Bayes': 'Probabilistic classifier',
        'Random Forest': 'Ensemble of decision trees',
    }

    selected_model = st.selectbox("Choose a model:", list(models.keys()))

    st.markdown(
        f"**{selected_model}**  \n"
        f"{model_descriptions.get(selected_model, '')}"
    )

with st.spinner("Processing data and generating predictions..."):
    try:
        results = run_inference(csv_bytes, selected_model, data_source)
    except FileNotFoundError as exc:
        st.error(f"Missing model artifact: {exc}. Push all model files to GitHub and redeploy.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to process CSV: {exc}")
        st.stop()

predictions = results['predictions']
probabilities = results['probabilities']
y_encoded = results['y_encoded']
has_labels = results['has_labels']

with st.sidebar:
    st.markdown("---")
    st.markdown("### Dataset Info")
    if uploaded_file is None:
        st.info("Using default test data (1000 rows)")
    else:
        st.success(f"Loaded: {data_source}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples", f"{results['row_count']:,}")
    with col2:
        st.metric("Model", selected_model.split()[0])

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Metrics",
    "Predictions",
    "Confusion Matrix",
    "About",
])

with tab1:
    st.header("Model Performance Metrics")
    st.caption(
        "Pre-computed metrics from the 80-20 training holdout (~25,900 rows). "
        "These do not change when you upload a different CSV."
    )

    model_metrics = metrics_df.loc[selected_model]
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.metric("Accuracy", f"{model_metrics['Accuracy']:.2%}")
        st.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
    with col2:
        st.metric("Precision", f"{model_metrics['Precision']:.4f}")
        st.metric("Recall", f"{model_metrics['Recall']:.4f}")
    with col3:
        st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
        st.metric("MCC Score", f"{model_metrics['MCC']:.4f}")

    if has_labels and results['accuracy'] is not None:
        st.markdown("---")
        st.subheader("Live Accuracy on Current Dataset")
        live_col1, live_col2 = st.columns(2)
        with live_col1:
            st.metric(
                "Current Dataset Accuracy",
                f"{results['accuracy']:.2%}",
                help=f"Computed on {results['row_count']:,} rows from {results['data_source']}"
            )
        with live_col2:
            stored = model_metrics['Accuracy']
            delta = results['accuracy'] - stored
            st.metric(
                "Holdout Accuracy (training)",
                f"{stored:.2%}",
                delta=f"{delta:+.2%} vs current",
                delta_color="off",
            )

    st.subheader("Comparison with Other Models")
    st.dataframe(metrics_df, width='stretch')

with tab2:
    st.header("Make Predictions")
    st.markdown(f"**Model:** {selected_model}")
    st.caption(f"Active dataset: **{results['data_source']}** ({results['row_count']:,} records)")

    st.markdown("### Prediction Summary")
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        st.metric("Total Samples", f"{len(predictions):,}")
    with col2:
        satisfied_count = int((predictions == 1).sum())
        st.metric("Satisfied", satisfied_count, f"{(satisfied_count / len(predictions)) * 100:.1f}%")
    with col3:
        not_satisfied_count = int((predictions == 0).sum())
        st.metric("Not Satisfied", not_satisfied_count, f"{(not_satisfied_count / len(predictions)) * 100:.1f}%")
    with col4:
        if has_labels:
            st.metric("Accuracy", f"{results['accuracy']:.2%}")
        else:
            st.metric("Accuracy", "N/A")

    st.markdown("---")
    st.subheader("Detailed Predictions")

    results_df = pd.DataFrame({
        'Satisfied_Probability': probabilities[:, 1].round(4),
        'Not_Satisfied_Probability': probabilities[:, 0].round(4),
        'Prediction': np.where(predictions == 1, 'Satisfied', 'Not Satisfied'),
    })

    if has_labels:
        results_df.insert(0, 'Actual', np.where(y_encoded == 1, 'Satisfied', 'Not Satisfied'))
        results_df['Correct'] = predictions == y_encoded

    display_limit = 200
    if len(results_df) > display_limit:
        st.caption(f"Showing first {display_limit} of {len(results_df):,} rows for faster loading.")
        st.dataframe(results_df.head(display_limit), width='stretch', height=400)
    else:
        st.dataframe(results_df, width='stretch', height=400)

with tab3:
    st.header("Confusion Matrix & Classification Report")
    st.caption(f"Active dataset: **{results['data_source']}** ({results['row_count']:,} records)")

    if has_labels:
        cm = confusion_matrix(y_encoded, predictions)

        st.markdown("### Sample Count Summary")
        col_counts = st.columns(5)
        with col_counts[0]:
            st.metric("Total Samples", f"{len(y_encoded):,}")
        with col_counts[1]:
            st.metric("True Negatives", f"{cm[0, 0]:,}")
        with col_counts[2]:
            st.metric("False Positives", f"{cm[0, 1]:,}")
        with col_counts[3]:
            st.metric("False Negatives", f"{cm[1, 0]:,}")
        with col_counts[4]:
            st.metric("True Positives", f"{cm[1, 1]:,}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Not Satisfied', 'Satisfied'],
                yticklabels=['Not Satisfied', 'Satisfied'],
                ax=ax,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'},
            )
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_title(f'Confusion Matrix - {selected_model}', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Classification Report")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Accuracy", f"{results['accuracy']:.2%}")
            with col2b:
                st.metric("F1-Score", f"{results['f1']:.4f}")

            report = classification_report(
                y_encoded,
                predictions,
                target_names=['Not Satisfied', 'Satisfied'],
                output_dict=True,
            )
            st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
    else:
        st.warning("Upload a CSV with a 'satisfaction' column to view the confusion matrix.")

with tab4:
    st.header("About This Project")
    st.markdown("""
    ### How to use this app
    - **Model Metrics**: Pre-computed scores from model training (fixed holdout split).
    - **Predictions / Confusion Matrix**: Results on the dataset you load (default 1000 rows or uploaded CSV).
    - Upload `data/test.csv` from the repo for predictions on ~26,000 rows.

    ### Models Implemented
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbor
    4. Naive Bayes
    5. Random Forest

    ### Dataset
    - 22 predictive features, binary satisfaction target
    - Default sample: 1,000 rows (`test_data.csv`)
  """)

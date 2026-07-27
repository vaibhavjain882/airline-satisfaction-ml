# Airline Passenger Satisfaction Classification

## Problem Statement

Airline passenger satisfaction is crucial for the success of any airline business. Airlines want to identify factors that lead to passenger satisfaction or dissatisfaction to improve their services. This project aims to build machine learning classification models that can predict whether a passenger will be satisfied with their airline experience based on various features such as age, travel type, cabin class, flight distance, and service ratings.

The goal is to develop multiple classification models and compare their performance to determine which model best predicts passenger satisfaction. This information can help airlines:
- Identify key factors affecting satisfaction
- Make targeted improvements in service delivery
- Predict customer satisfaction before it affects reputation
- Optimize operational decisions based on satisfaction patterns

---

## Dataset Description

**Dataset Name:** Airline Passenger Satisfaction

**Source:** [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

**Dataset Size:**
- **Total Instances:** 129,880 passengers
- **Total Features:** 22 features
- **Target Variable:** Satisfaction (Binary: Satisfied / Neutral or Dissatisfied)

**Class Distribution:**
- Satisfied: ~54% (70,000+ passengers)
- Neutral or Dissatisfied: ~46% (60,000+ passengers)

**Feature Descriptions:**

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Age of the passenger |
| Flight Distance | Numerical | Distance of the flight in miles |
| Inflight Wifi Service | Categorical | Satisfaction rating for wifi (0-5) |
| Departure/Arrival Time Convenient | Categorical | Rating of departure/arrival time (0-5) |
| Ease of Online Booking | Categorical | Ease of online booking (0-5) |
| Gate Location | Categorical | Satisfaction with gate location (0-5) |
| Food and Drink | Categorical | Satisfaction with food/drink (0-5) |
| Online Boarding | Categorical | Satisfaction with online boarding (0-5) |
| Seat Comfort | Categorical | Seat comfort rating (0-5) |
| Inflight Entertainment | Categorical | Entertainment quality (0-5) |
| On-air Service | Categorical | Quality of on-air service (0-5) |
| Leg Room Service | Categorical | Leg room satisfaction (0-5) |
| Baggage Handling | Categorical | Baggage handling satisfaction (0-5) |
| Checkin Service | Categorical | Check-in service quality (0-5) |
| Inflight Service | Categorical | Overall inflight service (0-5) |
| Cleanliness | Categorical | Cleanliness of aircraft (0-5) |
| Gender | Categorical | Passenger gender |
| Customer Type | Categorical | Loyal customer / Non-loyal |
| Type of Travel | Categorical | Business / Personal |
| Class | Categorical | Business / Eco / Eco Plus |
| Departure Delay (in minutes) | Numerical | Minutes delayed at departure |
| Arrival Delay (in minutes) | Numerical | Minutes delayed at arrival |

**Data Preprocessing:**
- Removed rows with missing values
- Encoded categorical variables using LabelEncoder
- Scaled numerical features using StandardScaler
- Applied 80-20 train-test split with stratification
- Total features used for modeling: 22

---

## GitHub Repository

**Repository Link:** [Airline Satisfaction ML](https://github.com/vaibhavjain882/airline-satisfaction-ml)

**Repository Structure:**
```
airline-satisfaction-project/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── test_data.csv                   # Test dataset (1000 samples for Streamlit app)
│
├── model/                          # Trained models directory & training code
│   ├── train_models.py             # Model training script
│   ├── logistic_regression.pkl     # Logistic Regression model
│   ├── decision_tree.pkl           # Decision Tree Classifier
│   ├── knn.pkl                     # K-Nearest Neighbors
│   ├── naive_bayes.pkl             # Naive Bayes Classifier
│   ├── random_forest.pkl           # Random Forest Classifier
│   ├── scaler.pkl                  # StandardScaler for features
│   └── metrics.csv                 # All metrics in CSV
│
└── data/
    ├── train.csv                   # Training dataset (from Kaggle)
    └── test.csv                    # Test dataset (from Kaggle)
```

---

## Models Used

Five machine learning classification models were implemented and evaluated on the same dataset. Each model was trained on 80% of the data and evaluated on 20% test data.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8650 | 0.9234 | 0.8612 | 0.8650 | 0.8625 | 0.7301 |
| Decision Tree | 0.9185 | 0.9180 | 0.9178 | 0.9185 | 0.9181 | 0.8371 |
| K-Nearest Neighbor | 0.9287 | 0.9674 | 0.9285 | 0.9287 | 0.9286 | 0.8575 |
| Naive Bayes | 0.8356 | 0.8945 | 0.8312 | 0.8356 | 0.8329 | 0.6706 |
| Random Forest | 0.9645 | 0.9952 | 0.9644 | 0.9645 | 0.9644 | 0.9291 |

### Model Observations and Analysis

#### 1. **Logistic Regression**
**Observation:** 
Logistic Regression achieved 86.50% accuracy with an AUC of 0.9234. This model assumes a linear relationship between features and the target variable. The model is fast to train and provides interpretable coefficients for understanding feature importance. However, it underperforms compared to more complex models, suggesting that the airline satisfaction problem has some non-linear patterns. The MCC score of 0.7301 indicates a moderate correlation between predictions and actual values.

**Why:** Linear models struggle when decision boundaries are complex. Airline satisfaction likely depends on non-linear interactions between service factors.

---

#### 2. **Decision Tree Classifier**
**Observation:**
Decision Tree achieved 91.85% accuracy with an AUC of 0.9180. Trees are excellent at capturing non-linear relationships and interactions between features. This model is highly interpretable - we can visualize the decision rules. However, decision trees are prone to overfitting on training data. The moderate F1 score (0.9181) suggests the model maintains balance between precision and recall. MCC of 0.8371 shows strong correlation.

**Why:** Trees excel at finding feature interactions but need pruning to avoid overfitting. The improvement over Logistic Regression indicates non-linear patterns exist in the data.

---

#### 3. **K-Nearest Neighbor (KNN)**
**Observation:**
KNN achieved 92.87% accuracy with the highest AUC of 0.9674 among traditional models. This instance-based learner works by finding the K nearest neighbors and using their majority class. KNN performs well because airline satisfaction likely has local patterns - passengers with similar profiles tend to have similar satisfaction levels. The high AUC (0.9674) indicates excellent separation between classes. MCC of 0.8575 confirms strong predictive power. The model's strength lies in capturing local patterns without making assumptions about the underlying distribution.

**Why:** KNN works well when similar passengers cluster together in feature space. The high AUC confirms good class separation. Requires feature scaling (which we applied).

---

#### 4. **Naive Bayes Classifier**
**Observation:**
Naive Bayes achieved 83.56% accuracy with AUC of 0.8945. This probabilistic model assumes feature independence, which is a strong assumption not always true in real-world data. Service ratings (wifi, food, seat comfort) are likely correlated with each other, violating this assumption. Despite this limitation, the model still performs reasonably well, suggesting that some features are indeed independently informative. The lower F1 score (0.8329) and MCC (0.6706) indicate this is the weakest model for this problem. However, Naive Bayes is very fast to train and good for baseline comparisons.

**Why:** The assumption of feature independence is violated when multiple service aspects are correlated. For example, good seat comfort often correlates with good overall service.

---

#### 5. **Random Forest (Ensemble)**
**Observation:**
Random Forest is the clear winner with **96.45% accuracy** and an exceptional **AUC of 0.9952**. This ensemble method combines multiple decision trees to reduce overfitting and improve generalization. The model achieves the highest scores across all metrics - Precision: 0.9644, Recall: 0.9645, F1: 0.9644, MCC: 0.9291. The near-perfect balance between precision and recall (both ~96.45%) indicates the model is excellent at both correctly identifying satisfied passengers and correctly identifying dissatisfied ones. The extraordinarily high AUC (0.9952) suggests nearly perfect class separation.

**Why:** Ensemble methods aggregate weak learners into a strong model. Random Forest's strength comes from:
- Multiple trees voting on predictions
- Random feature subsets reducing correlation between trees
- Robustness to overfitting
- Ability to capture complex non-linear patterns

---

### Overall Winner: **Random Forest Classifier**

The Random Forest model is the clear winner for this airline satisfaction prediction task because:

1. **Highest Accuracy (96.45%)**: Correctly predicts satisfaction for 96+ out of 100 passengers
2. **Exceptional AUC (0.9952)**: Almost perfect ability to distinguish between satisfied and dissatisfied passengers
3. **Perfect Balance**: Precision and Recall are nearly identical (0.9644 and 0.9645), meaning equally good at avoiding false positives and false negatives
4. **Best Generalization**: Ensemble approach reduces overfitting risk
5. **Robustness**: Works well with both numerical and categorical features without special tuning

---

## How to Start the Project Locally

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd airline-satisfaction-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (optional, if you want to regenerate trained models):**
   ```bash
   python model/train_models.py
   ```

4. **Run the Streamlit app:**
   ```bash
   python3 -m streamlit run app.py
   ```

5. **Access the app:**
   - The app will open automatically in your default browser at `http://localhost:8501`
   - If not, manually navigate to that URL

### What the Streamlit App Does
- Upload CSV files with test data
- Select from 5 different ML models
- View model evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Make predictions on new data
- Visualize confusion matrix for selected model
- Compare performance across all models

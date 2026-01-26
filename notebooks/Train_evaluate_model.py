

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, 
                             title="Model Evaluation", threshold=0.5, 
                             use_smote=False, apply_class_weight=False):
    """
    Trains and evaluates a model with optional SMOTE or class weighting.
    Returns performance metrics and displays confusion matrix.
    """

    # ----- 1. Handle class imbalance -----
    if use_smote:
        sm = SMOTE(random_state=0)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    # ----- 2. Apply class weighting if supported -----
    if apply_class_weight and hasattr(model, 'class_weight'):
        model.set_params(class_weight='balanced')

    # ----- 3. Train the model -----
    model.fit(X_train, y_train)

    # ----- 4. Predict and evaluate -----
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 {title} (Threshold = {threshold}):")
    print("----------------------------------------")
    print(f"Accuracy      : {acc:.2f}")
    print(f"Precision     : {prec:.2f}")
    print(f"Recall        : {rec:.2f}")
    print(f"F1 Score      : {f1:.2f}")
    print(f"AUC (ROC)     : {auc:.2f}")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Greens")

    return {
        "Model": title,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc
    }


from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(max_iter=1000)
train_and_evaluate_model(log_reg_model, X_train_full, y_train, X_test_full, y_test, title="Logistic Regression")




from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, random_state=0)
train_and_evaluate_model(rf_model, X_train_full, y_train, X_test_full, y_test, 
                         title="Random Forest + SMOTE", use_smote=True)




from xgboost import XGBClassifier

imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=imbalance_ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=0
)

train_and_evaluate_model(xgb_model, X_train_full, y_train, X_test_full, y_test, title="XGBoost")

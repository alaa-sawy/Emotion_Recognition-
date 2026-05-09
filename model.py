from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def train_seven_expressions_model(X_train, X_test, y_train, y_test):

    # Pipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),

        # SMOTE after PCA projection
        ('smote', SMOTE(random_state=42)),

        ('rf', RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Hyperparameter Search Space
    param_grid = {
        'rf__n_estimators': [300, 500],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'smote__k_neighbors': [3, 5]
    }

    print("\n" + "="*50)
    print("[TRAINING] Executing Optimized ImbPipeline...")
    print("="*50)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    # Train ONLY on train set
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"\n[OK] Best Parameters: {grid_search.best_params_}")
    print(f"[OK] Best CV Score (F1-Macro): {grid_search.best_score_:.4f}")

    # Final Evaluation
    y_pred = best_model.predict(X_test)

    # Visualization Support
    y_before = y_train

    best_smote = best_model.named_steps['smote']
    _, y_after = best_smote.fit_resample(X_train, y_train)

    return best_model, y_test, y_pred, y_before, y_after
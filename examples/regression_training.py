"""
ClearML Regression Training Example
====================================
하이퍼파라미터 최적화를 포함한 회귀 모델 학습 예제

Features:
- wget을 통한 데이터셋 다운로드 (California Housing)
- ClearML Hyperparameter Optimization
- 다양한 메트릭 시각화
- 원격 실행 지원 (Colab Agent)

사용법:
1. 로컬 실행: python regression_training.py
2. 원격 대기열 추가:
   clearml-task --project Manifest-Regression --name housing-price --script regression_training.py --queue default
"""

import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ClearML 임포트
from clearml import Task, Logger

# Scikit-learn 임포트
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ===========================================
# ClearML Task 초기화
# ===========================================
task = Task.init(
    project_name="Manifest-Regression",
    task_name="Housing-Price-Prediction",
    task_type=Task.TaskTypes.training,
    tags=["regression", "sklearn", "housing", "example"]
)

# 하이퍼파라미터 설정
params = {
    "model_type": "gradient_boosting",  # gradient_boosting, random_forest, ridge, lasso
    "test_size": 0.2,
    "random_state": 42,
    
    # Gradient Boosting 파라미터
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    
    # Cross-validation
    "cv_folds": 5
}
task.connect(params)

logger = Logger.current_logger()


# ===========================================
# 데이터셋 다운로드
# ===========================================
def download_housing_dataset():
    """California Housing 데이터셋 로드"""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="target")
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # 데이터셋 통계를 ClearML에 기록
    logger.report_table(
        title="Dataset Statistics",
        series="Features",
        table_plot=X.describe().round(3)
    )
    
    return X, y


# ===========================================
# 모델 생성
# ===========================================
def create_model(params):
    """파라미터에 따른 모델 생성"""
    model_type = params["model_type"]
    
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"]
        )
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"]
        )
    elif model_type == "ridge":
        return Ridge(alpha=1.0, random_state=params["random_state"])
    elif model_type == "lasso":
        return Lasso(alpha=0.1, random_state=params["random_state"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ===========================================
# 학습 및 평가
# ===========================================
def train_and_evaluate(model, X_train, X_test, y_train, y_test, params):
    """모델 학습 및 평가"""
    
    # 교차 검증
    print("\nCross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=params["cv_folds"], 
        scoring="r2"
    )
    
    logger.report_scalar("CV Score", "mean", value=cv_scores.mean(), iteration=1)
    logger.report_scalar("CV Score", "std", value=cv_scores.std(), iteration=1)
    
    print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 전체 학습
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 메트릭 계산
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    }
    
    # ClearML에 메트릭 기록
    for name, value in metrics.items():
        split = "train" if "train" in name else "test"
        metric = name.replace("train_", "").replace("test_", "")
        logger.report_scalar(metric.upper(), split, value=value, iteration=1)
    
    # 결과 테이블
    results_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "MAE", "R²"],
        "Train": [metrics["train_mse"], metrics["train_rmse"], 
                  metrics["train_mae"], metrics["train_r2"]],
        "Test": [metrics["test_mse"], metrics["test_rmse"], 
                 metrics["test_mae"], metrics["test_r2"]]
    }).round(4)
    
    logger.report_table(
        title="Model Performance",
        series="Metrics",
        table_plot=results_df
    )
    
    print("\n" + "="*40)
    print("Performance Summary")
    print("="*40)
    print(results_df.to_string(index=False))
    
    return model, metrics


# ===========================================
# Feature Importance 분석
# ===========================================
def analyze_feature_importance(model, feature_names, params):
    """Feature Importance 분석 및 시각화"""
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        # DataFrame 생성
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        
        # ClearML에 기록
        logger.report_table(
            title="Feature Importance",
            series="Analysis",
            table_plot=importance_df.round(4)
        )
        
        # 바 차트로도 기록
        for idx, row in importance_df.iterrows():
            logger.report_scalar(
                "Feature Importance", 
                row["Feature"], 
                value=row["Importance"], 
                iteration=1
            )
        
        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    return None


# ===========================================
# 메인 실행
# ===========================================
def main():
    print(f"Model type: {params['model_type']}")
    print("="*50)
    
    # 데이터 로드
    X, y = download_housing_dataset()
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params["test_size"],
        random_state=params["random_state"]
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 생성
    model = create_model(params)
    print(f"\nModel: {model.__class__.__name__}")
    
    # 학습 및 평가
    trained_model, metrics = train_and_evaluate(
        model, X_train_scaled, X_test_scaled, y_train, y_test, params
    )
    
    # Feature Importance 분석
    analyze_feature_importance(trained_model, X.columns.tolist(), params)
    
    # 모델 저장
    model_path = f"best_{params['model_type']}_model.joblib"
    joblib.dump(trained_model, model_path)
    task.upload_artifact("best_model", artifact_object=model_path)
    
    # Scaler도 저장
    scaler_path = "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    task.upload_artifact("scaler", artifact_object=scaler_path)
    
    # 최종 결과 기록
    logger.report_single_value("final_r2", metrics["test_r2"])
    logger.report_single_value("final_rmse", metrics["test_rmse"])
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Final Test R²: {metrics['test_r2']:.4f}")
    print(f"Final Test RMSE: {metrics['test_rmse']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()

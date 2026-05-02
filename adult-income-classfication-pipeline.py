from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    dataset_dir: Path = Path("dataset")
    processed_path: Path = Path("dataset/processed.csv")
    target_path: Path = Path("dataset/target.csv")
    feature_engineering_path: Path = Path("dataset/feature_engineering.csv")
    selected_features_path: Path = Path("dataset/selected_features.csv")
    test_size: float = 0.2
    random_state: int = 42
    variance_threshold: float = 0.01
    cv_folds: int = 5
    n_iter_search: int = 20
    iqr_multiplier: float = 1.5
    top_n_models: int = 2

    def __post_init__(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)


CONFIG = PipelineConfig()


class DataPreProcessing:
    def __init__(self) -> None:
        logger.info("Fetching Adult dataset from OpenML...")
        data = fetch_openml(name="adult", version=2, as_frame=True)
        self.df: pd.DataFrame = data.data.copy()
        self.df["income"] = data.target
        logger.info("Dataset loaded: %d rows, %d columns", *self.df.shape)

    def _handle_duplicates(self) -> None:
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info("Removed %d duplicate rows", before - len(self.df))

    def _handle_missing_values(self) -> None:
        self.df.replace("?", np.nan, inplace=True)
        for col in ["workclass", "occupation", "native-country"]:
            if col not in self.df.columns:
                continue
            if hasattr(self.df[col], "cat") and "Unknown" not in self.df[col].cat.categories:
                self.df[col] = self.df[col].cat.add_categories("Unknown")
            self.df[col] = self.df[col].fillna("Unknown")
        remaining = self.df.isnull().sum().sum()
        logger.info("Missing value imputation complete. Remaining nulls: %d", remaining)

    def _handle_outliers(self) -> None:
        before = len(self.df)
        self.df = OutlierDetection(CONFIG.iqr_multiplier).iqr(self.df, "hours-per-week")
        logger.info("Removed %d outlier rows from 'hours-per-week'", before - len(self.df))

    def _export(self) -> None:
        self.df.to_csv(CONFIG.processed_path, index=False)
        logger.info("Processed dataset saved → %s", CONFIG.processed_path)

    def execute(self) -> None:
        self._handle_duplicates()
        self._handle_missing_values()
        self._handle_outliers()
        self._export()


class OutlierDetection:
    def __init__(self, iqr_multiplier: float = 1.5) -> None:
        self.iqr_multiplier = iqr_multiplier

    def iqr(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr_range = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr_range
        upper = q3 + self.iqr_multiplier * iqr_range
        # Keep rows within bounds (AND, not OR)
        return df[(df[col] >= lower) & (df[col] <= upper)]


class FeatureEngineering:
    def __init__(self, path: Path = CONFIG.processed_path) -> None:
        self.df: pd.DataFrame = pd.read_csv(path)
        self._numerical_cols: List[str] = []
        logger.info("FeatureEngineering: loaded %d rows, %d columns from %s", *self.df.shape, path)

    def _derive_columns(self) -> None:
        # Create bins on the original scale before any log transformation
        self.df["age_bin"] = pd.cut(
            self.df["age"], bins=[0, 25, 40, 60, 120],
            labels=["young", "adult", "mid", "senior"], include_lowest=True,
        )
        self.df["hours_bin"] = pd.cut(
            self.df["hours-per-week"], bins=[0, 25, 40, 60, np.inf],
            labels=["low", "normal", "high", "very_high"], include_lowest=True,
        )
        self.df["capital-gain"] = np.log1p(self.df["capital-gain"])
        self.df["capital-loss"] = np.log1p(self.df["capital-loss"])
        self.df["hours-per-week"] = np.log1p(self.df["hours-per-week"].clip(lower=0))
        self.df["net-capital"] = self.df["capital-gain"] - self.df["capital-loss"]
        logger.info("Derived columns: age_bin, hours_bin, net-capital; log1p applied to skewed numerics")

    def _encode(self) -> None:
        # Capture numerical columns before encoding so _scaling targets only them
        self._numerical_cols = self.df.select_dtypes(include="number").columns.tolist()
        self.df[["income"]].to_csv(CONFIG.target_path, index=False)
        feature_cols = [col for col in self.df.columns if col != "income"]
        self.df = pd.get_dummies(self.df[feature_cols], drop_first=True, dtype=int)
        logger.info("Encoding complete: %d features after one-hot encoding", self.df.shape[1])

    def _scaling(self) -> None:
        num_cols = [col for col in self._numerical_cols if col in self.df.columns]
        self.df[num_cols] = RobustScaler().fit_transform(self.df[num_cols])
        logger.info("RobustScaler applied to %d numerical columns", len(num_cols))

    def _export(self) -> None:
        self.df.to_csv(CONFIG.feature_engineering_path, index=False)
        logger.info("Feature-engineered data saved → %s", CONFIG.feature_engineering_path)

    def execute(self) -> None:
        self._derive_columns()
        self._encode()
        self._scaling()
        self._export()


class FeatureSelection:
    def __init__(self, path: Path = CONFIG.feature_engineering_path) -> None:
        # income was already excluded during FeatureEngineering._encode
        self.X: pd.DataFrame = pd.read_csv(path)
        logger.info("FeatureSelection: loaded %d features", self.X.shape[1])

    def _select(self) -> None:
        initial_count = self.X.shape[1]
        selector = VarianceThreshold(threshold=CONFIG.variance_threshold)
        X_array = selector.fit_transform(self.X)
        selected_cols = self.X.columns[selector.get_support()]
        self.X = pd.DataFrame(X_array, columns=selected_cols)
        logger.info(
            "VarianceThreshold: %d → %d features retained (threshold=%.3f)",
            initial_count, len(selected_cols), CONFIG.variance_threshold,
        )
        self.X.to_csv(CONFIG.selected_features_path, index=False)
        logger.info("Selected features saved → %s", CONFIG.selected_features_path)

    def execute(self) -> None:
        self._select()


class HyperParameterTuning:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        param_distributions: Dict,
        classifier: object,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.param_distributions = param_distributions
        self.classifier = classifier

    def tune(self) -> Tuple[Dict, object]:
        search = RandomizedSearchCV(
            self.classifier,
            param_distributions=self.param_distributions,
            n_iter=CONFIG.n_iter_search,
            scoring="roc_auc",
            cv=CONFIG.cv_folds,
            verbose=1,
            n_jobs=-1,
            random_state=CONFIG.random_state,
        )
        search.fit(self.X_train, np.ravel(self.y_train))
        logger.info("Best params: %s", search.best_params_)
        return search.best_params_, search.best_estimator_


class ModelSelection:
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # SVC requires probability=True to support predict_proba
        self.models: Dict[str, object] = {
            "logistic": LogisticRegression(max_iter=1000),
            "decision_tree": DecisionTreeClassifier(random_state=CONFIG.random_state),
            "random_forest": RandomForestClassifier(random_state=CONFIG.random_state),
            "xgb_classifier": XGBClassifier(eval_metric="logloss", random_state=CONFIG.random_state)
        }
        self.models_params_dist: Dict[str, Dict] = {
            "logistic": {
                "penalty": ["l2", "l1"],
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "saga"],
                "class_weight": [None, "balanced"],
            },
            "decision_tree": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"],
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            "xgb_classifier": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
        }

    def _select_top_n_models(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for name, model in self.models.items():
            logger.info("Cross-validating '%s'...", name)
            cv_scores = cross_val_score(
                model, X=self.X_train, y=np.ravel(self.y_train),
                cv=CONFIG.cv_folds, scoring="roc_auc", n_jobs=-1,
            )
            scores[name] = cv_scores.mean()
            logger.info("  %s → CV ROC-AUC: %.4f ± %.4f", name, cv_scores.mean(), cv_scores.std())
        # Sort and take top-N using list slicing (dicts are not sliceable)
        top_n = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CONFIG.top_n_models])
        logger.info("Top %d models selected: %s", CONFIG.top_n_models, list(top_n.keys()))
        return top_n

    def execute(self) -> Tuple[str, object]:
        top_models = self._select_top_n_models()

        tuned: Dict[str, object] = {}
        for name in top_models:
            logger.info("Tuning hyperparameters for '%s'...", name)
            _, estimator = HyperParameterTuning(
                self.X_train, self.y_train,
                self.models_params_dist[name],
                self.models[name],
            ).tune()
            tuned[name] = estimator

        auc_scores: Dict[str, float] = {}
        for name, estimator in tuned.items():
            y_prob = estimator.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(np.ravel(self.y_test), y_prob)
            auc_scores[name] = auc
            logger.info("  %s → Test ROC-AUC: %.4f", name, auc)

        best_name = max(auc_scores, key=auc_scores.get)
        logger.info("Best model: %s (ROC-AUC=%.4f)", best_name, auc_scores[best_name])
        return best_name, tuned[best_name]


class Evaluation:
    def __init__(self, model_name: str, estimator: object) -> None:
        self.model_name = model_name
        self.estimator = estimator

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        y_true = np.ravel(y_test)
        y_pred = self.estimator.predict(X_test)
        y_prob = self.estimator.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_true, y_prob)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        sep = "=" * 60
        logger.info(sep)
        logger.info("Evaluation — %s", self.model_name)
        logger.info("ROC-AUC : %.4f", auc)
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("Classification Report:\n%s", classification_report(y_true, y_pred))
        logger.info(sep)

        return {
            "model": self.model_name,
            "roc_auc": auc,
            "accuracy": report_dict["accuracy"],
            "precision": report_dict["weighted avg"]["precision"],
            "recall": report_dict["weighted avg"]["recall"],
            "f1_score": report_dict["weighted avg"]["f1-score"],
        }


class Pipeline:
    def __init__(self, config: PipelineConfig = CONFIG) -> None:
        self.config = config

    def run(self) -> Dict[str, float]:
        sep = "=" * 60
        logger.info(sep)
        logger.info("Adult Income Classification Pipeline — START")
        logger.info(sep)

        logger.info("[1/4] Data Preprocessing")
        DataPreProcessing().execute()

        logger.info("[2/4] Feature Engineering")
        FeatureEngineering(self.config.processed_path).execute()

        logger.info("[3/4] Feature Selection")
        FeatureSelection(self.config.feature_engineering_path).execute()

        X = pd.read_csv(self.config.selected_features_path)
        y_raw = pd.read_csv(self.config.target_path)
        y = pd.DataFrame(
            LabelEncoder().fit_transform(np.ravel(y_raw)),
            columns=["income"],
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        logger.info(
            "Train/test split (stratified): %d train | %d test",
            len(X_train), len(X_test),
        )

        logger.info("[4/4] Model Selection & Hyperparameter Tuning")
        model_name, best_estimator = ModelSelection(
            X_train, X_test, y_train, y_test
        ).execute()

        metrics = Evaluation(model_name, best_estimator).evaluate(X_test, y_test)

        logger.info(sep)
        logger.info("Pipeline complete.")
        logger.info(sep)
        return metrics


if __name__ == "__main__":
    results = Pipeline().run()
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

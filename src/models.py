"""
Model training with walk-forward validation, purged cross-validation,
and SHAP-based feature importance.

Trains: Lasso (linear), XGBoost (tree), Random Forest.
Evaluation: walk-forward expanding window with purge + embargo gaps.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: Dict[str, float]
    preds: pd.Series
    y_test: pd.Series
    feature_importance: Optional[pd.Series] = None
    walk_forward_results: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# Deep Learning PyTorch Implementations
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=60):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32)
        else:
            self.y = torch.zeros(len(X), dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]

BaseModule = nn.Module if nn is not None else object

class LSTMModel(BaseModule):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        if nn is not None:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
            self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(1)

class PyTorchLSTMRegressor:
    def __init__(self, seq_len=60, hidden_dim=128, num_layers=2, epochs=5, batch_size=4096, lr=1e-3, device="cuda"):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device if torch and torch.cuda.is_available() else "cpu"
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y, eval_set=None, verbose=True):
        if not torch:
            raise ImportError("PyTorch is not installed. Please install it to use LSTM.")
            
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        self.model = LSTMModel(input_dim=X.shape[1], hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        dataset = TimeSeriesDataset(X_scaled_df, y, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if verbose:
                print(f"  LSTM Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.6f}")
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        dataset = TimeSeriesDataset(X_scaled_df, None, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                preds.append(pred.cpu().numpy())
                
        preds = np.concatenate(preds) if len(preds) > 0 else np.array([])
        
        # Pad the first `seq_len` samples with the first prediction to maintain length match with X
        pad_val = preds[0] if len(preds) > 0 else 0.0
        pad = np.full(self.seq_len, pad_val)
        return np.concatenate([pad, preds])


# ---------------------------------------------------------------------------
# Train / test helpers
# ---------------------------------------------------------------------------

def _train_test_time_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> Tuple:
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return (np.sign(y_true) == np.sign(y_pred)).mean()


def _information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Rank IC — Spearman correlation between predictions and actuals."""
    return y_true.corr(y_pred, method="spearman")


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": _directional_accuracy(y_true, y_pred),
        "information_coefficient": _information_coefficient(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Feature importance via SHAP (graceful fallback)
# ---------------------------------------------------------------------------

def _get_feature_importance(model, X_test: pd.DataFrame) -> Optional[pd.Series]:
    """Get SHAP-based or built-in feature importance."""
    try:
        import shap
        # Use TreeExplainer for tree models, LinearExplainer for linear
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.head(100))
            return pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X_test.columns,
                name="shap_importance",
            ).sort_values(ascending=False)
    except Exception:
        pass

    # Fallback: built-in importance for tree models
    if hasattr(model, "feature_importances_"):
        return pd.Series(
            model.feature_importances_,
            index=X_test.columns,
            name="importance",
        ).sort_values(ascending=False)

    # Fallback: coefficient magnitude for linear models
    if hasattr(model, "named_steps") and hasattr(model.named_steps.get("model", None), "coef_"):
        coefs = model.named_steps["model"].coef_
        return pd.Series(
            np.abs(coefs),
            index=X_test.columns,
            name="coef_magnitude",
        ).sort_values(ascending=False)

    return None


# ---------------------------------------------------------------------------
# Walk-forward validation (expanding window with purge + embargo)
# ---------------------------------------------------------------------------

def walk_forward_validate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_bars: int = 5,
    embargo_bars: int = 3,
) -> Tuple[pd.Series, pd.Series, List[dict]]:
    """Expanding-window walk-forward with purge gap between train/test.

    Parameters
    ----------
    model_factory : callable that returns a fresh model instance
    purge_bars : number of bars to drop between train end and test start
    embargo_bars : number of bars to drop after test end before next train window

    Returns
    -------
    all_preds, all_actuals, fold_results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_preds = []
    all_actuals = []
    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Apply purge: remove last `purge_bars` from train
        if purge_bars > 0 and len(train_idx) > purge_bars:
            train_idx = train_idx[:-purge_bars]

        # Apply embargo: skip first `embargo_bars` from test
        if embargo_bars > 0 and len(test_idx) > embargo_bars:
            test_idx = test_idx[embargo_bars:]

        if len(train_idx) < 30 or len(test_idx) < 10:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_factory()
        
        # Verbose terminal output for XGBoost progress
        try:
            import xgboost as xgb
            is_xgb = isinstance(model, (xgb.XGBRegressor, xgb.XGBRFRegressor))
        except ImportError:
            is_xgb = False
            
        if is_xgb:
            print(f"\n--- Training Fold {fold_num} [{model.__class__.__name__}] ---")
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
        elif isinstance(model, PyTorchLSTMRegressor):
            print(f"\n--- Training Fold {fold_num} [LSTM] ---")
            model.fit(X_train, y_train)
        else:
            print(f"\n--- Training Fold {fold_num} [{model.__class__.__name__}] ---")
            model.fit(X_train, y_train)
            
        preds = pd.Series(model.predict(X_test), index=y_test.index)

        metrics = _compute_metrics(y_test, preds)
        fold_results.append({"fold": fold_num, **metrics, "test_size": len(test_idx)})

        all_preds.append(preds)
        all_actuals.append(y_test)

    if not all_preds:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    return pd.concat(all_preds), pd.concat(all_actuals), fold_results


# ---------------------------------------------------------------------------
# Model training functions
# ---------------------------------------------------------------------------

# Max rows to use for Lasso hyperparameter search (LassoCV is O(n) but slow at n>100K)
_LASSO_SEARCH_MAX_ROWS = 100_000


def train_linear_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_walk_forward: bool = True,
) -> ModelResult:
    X = df.drop(columns=["target"])
    y = df["target"]
    _alphas = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    # Scale-aware alpha search: subsample to avoid hanging on large datasets
    if len(X) > _LASSO_SEARCH_MAX_ROWS:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(len(X), size=_LASSO_SEARCH_MAX_ROWS, replace=False))
        X_search, y_search = X.iloc[idx], y.iloc[idx]
    else:
        X_search, y_search = X, y

    # Find best alpha on subsample
    _scaler = StandardScaler()
    X_search_scaled = _scaler.fit_transform(X_search)
    _lasso_cv = LassoCV(alphas=_alphas, cv=3, max_iter=5000)
    _lasso_cv.fit(X_search_scaled, y_search)
    best_alpha = _lasso_cv.alpha_

    from sklearn.linear_model import Lasso

    if use_walk_forward:
        def factory():
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=best_alpha, max_iter=5000)),
            ])
        preds, y_test, wf_results = walk_forward_validate(factory, X, y, n_splits=5)
        # Train final model on all data for prediction
        final_model = factory()
        final_model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = _train_test_time_split(X, y, test_size)
        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=best_alpha, max_iter=5000)),
        ])
        final_model.fit(X_train, y_train)
        preds = pd.Series(final_model.predict(X_test), index=y_test.index)
        wf_results = None

    metrics = _compute_metrics(y_test, preds)
    importance = _get_feature_importance(final_model, X)

    return ModelResult(
        name="lasso",
        model=final_model,
        metrics=metrics,
        preds=preds,
        y_test=y_test,
        feature_importance=importance,
        walk_forward_results=wf_results,
    )


def train_tree_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_walk_forward: bool = True,
) -> ModelResult:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    X = df.drop(columns=["target"])
    y = df["target"]

    # Use GPU for parallel tree building
    n_estimators = 150 if len(df) < 200_000 else 200
    model_params = dict(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        random_state=random_state,
    )

    if use_walk_forward:
        def factory():
            return XGBRegressor(**model_params)
        preds, y_test, wf_results = walk_forward_validate(factory, X, y, n_splits=5)
        final_model = XGBRegressor(**model_params)
        print("\n--- Training Final XGBoost Model on Full Data ---")
        final_model.fit(X, y, eval_set=[(X, y)], verbose=50)
    else:
        X_train, X_test, y_train, y_test = _train_test_time_split(X, y, test_size)
        final_model = XGBRegressor(**model_params)
        final_model.fit(X_train, y_train)
        preds = pd.Series(final_model.predict(X_test), index=y_test.index)
        wf_results = None

    metrics = _compute_metrics(y_test, preds)
    importance = _get_feature_importance(final_model, X)

    return ModelResult(
        name="xgboost",
        model=final_model,
        metrics=metrics,
        preds=preds,
        y_test=y_test,
        feature_importance=importance,
        walk_forward_results=wf_results,
    )


def train_forest_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_walk_forward: bool = True,
) -> ModelResult:
    try:
        from xgboost import XGBRFRegressor
        base_params = {"tree_method": "hist", "device": "cuda", "random_state": random_state}
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor as XGBRFRegressor
        base_params = {"n_jobs": -1, "random_state": random_state}

    X = df.drop(columns=["target"])
    y = df["target"]

    # Lightweight hyperparam sweep via time series split
    candidate_params = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 7},
    ]

    # Subsample for fast hyperparameter search to avoid OOM/hanging
    search_sample_size = min(len(X), 100_000)
    if len(X) > search_sample_size:
        X_sel = X.sample(n=search_sample_size, random_state=random_state).sort_index()
        y_sel = y.loc[X_sel.index]
    else:
        X_sel, y_sel = X, y

    tscv = TimeSeriesSplit(n_splits=3)

    best_params = candidate_params[0]
    best_score = -np.inf
    for params in candidate_params:
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_sel):
            m = XGBRFRegressor(**base_params, **params)
            m.fit(X_sel.iloc[train_idx], y_sel.iloc[train_idx])
            p = m.predict(X_sel.iloc[val_idx])
            fold_scores.append(r2_score(y_sel.iloc[val_idx], p))
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_params = params

    if use_walk_forward:
        def factory():
            return XGBRFRegressor(**base_params, **best_params)
        preds, y_test, wf_results = walk_forward_validate(factory, X, y, n_splits=5)
        final_model = XGBRFRegressor(**base_params, **best_params)
        print("\n--- Training Final Random Forest Model on Full Data ---")
        try:
            final_model.fit(X, y, eval_set=[(X, y)], verbose=50)
        except TypeError:
            final_model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = _train_test_time_split(X, y, test_size)
        final_model = XGBRFRegressor(**base_params, **best_params)
        final_model.fit(X_train, y_train)
        preds = pd.Series(final_model.predict(X_test), index=y_test.index)
        wf_results = None

    metrics = _compute_metrics(y_test, preds)
    importance = _get_feature_importance(final_model, X)

    return ModelResult(
        name="random_forest",
        model=final_model,
        metrics=metrics,
        preds=preds,
        y_test=y_test,
        feature_importance=importance,
        walk_forward_results=wf_results,
    )


def train_lstm_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    use_walk_forward: bool = True,
) -> ModelResult:
    X = df.drop(columns=["target"])
    y = df["target"]
    
    if not torch:
        print("[WARN] PyTorch not found. Falling back to Lasso.")
        return train_linear_model(df, test_size, random_state, use_walk_forward)

    # Note: LSTM takes longer to train on CPU. Adjust epochs/batch for speed.
    model_params = dict(
        seq_len=60,       # 1 hour context window for 1min data
        hidden_dim=128,   
        num_layers=2,
        epochs=5,         # Fast baseline
        batch_size=8192,  # Large batch size for speed over 1.7M rows
        lr=2e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    if use_walk_forward:
        def factory():
            return PyTorchLSTMRegressor(**model_params)
        preds, y_test, wf_results = walk_forward_validate(factory, X, y, n_splits=5)
        final_model = factory()
        print("\n--- Training Final LSTM Model on Full Data ---")
        final_model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = _train_test_time_split(X, y, test_size)
        final_model = PyTorchLSTMRegressor(**model_params)
        final_model.fit(X_train, y_train)
        preds = pd.Series(final_model.predict(X_test), index=y_test.index)
        wf_results = None

    metrics = _compute_metrics(y_test, preds)
    importance = _get_feature_importance(final_model, X) # Usually None for deep learning without SHAP DeepExplainer

    return ModelResult(
        name="lstm",
        model=final_model,
        metrics=metrics,
        preds=preds,
        y_test=y_test,
        feature_importance=importance,
        walk_forward_results=wf_results,
    )


def predict_next(model, latest_features: pd.DataFrame) -> float:
    """Generate prediction for the most recent bar."""
    if hasattr(model, 'seq_len'):
        # LSTM Model needs context sequence
        seq_len = model.seq_len
        X_input = latest_features.tail(seq_len * 2) 
        if len(X_input) < seq_len:
            return 0.0 # Cannot predict without sequence
        preds = model.predict(X_input)
        return float(preds[-1])
    else:
        check_is_fitted(model)
        return float(model.predict(latest_features.tail(1))[0])

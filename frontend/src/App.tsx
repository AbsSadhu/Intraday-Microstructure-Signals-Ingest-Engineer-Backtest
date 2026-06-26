import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
  directional_accuracy: number;
  information_coefficient: number;
}

interface ModelPerf {
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  calmar: number;
  hit_rate: number;
  max_drawdown: number;
  final_equity: number;
  n_trades: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  avg_holding_bars: number;
  total_cost: number | string;
  gross_pnl: number;
  net_pnl: number;
}

interface ModelResult {
  metrics: ModelMetrics;
  perf: ModelPerf;
}

interface Metrics {
  lasso: ModelResult;
  xgboost: ModelResult;
  random_forest: ModelResult;
  lstm: ModelResult;
  latest_signal: number;
  best_model: "lasso" | "xgboost" | "random_forest" | "lstm";
  evaluation: {
    ic: number;
    icir: number;
    turnover_adjusted_alpha: number;
    calibration_monotonicity: number;
  };
  walk_forward: {
    wf_mae_mean: number;
    wf_mae_std: number;
    wf_mae_min: number;
    wf_mae_max: number;
    wf_rmse_mean: number;
    wf_rmse_std: number;
    wf_rmse_min: number;
    wf_rmse_max: number;
    wf_r2_mean: number;
    wf_r2_std: number;
    wf_r2_min: number;
    wf_r2_max: number;
    wf_directional_accuracy_mean: number;
    wf_directional_accuracy_std: number;
    wf_directional_accuracy_min: number;
    wf_directional_accuracy_max: number;
    wf_information_coefficient_mean?: number;
    wf_information_coefficient_std?: number;
    wf_information_coefficient_min?: number;
    wf_information_coefficient_max?: number;
    wf_n_folds: number;
    wf_total_test_samples: number;
  };
  [key: string]: any;
}

function App() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<boolean>(false);

  useEffect(() => {
    const checkServerAndLoadData = async () => {
      try {
        const healthCheck = await axios.get('http://localhost:8000/');
        if (healthCheck.data.status === 'ok') {
          setServerStatus(true);
          const response = await axios.get(`${API_BASE_URL}/metrics`);
          setMetrics(response.data);
        }
      } catch (err: any) {
        setServerStatus(false);
        setError(err.message || 'Failed to connect to API');
      } finally {
        setLoading(false);
      }
    };

    checkServerAndLoadData();
    const interval = setInterval(checkServerAndLoadData, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);


  return (
    <div style={{ padding: '20px 0' }}>
      <header style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 className="title">Quant Research Dashboard</h1>
        <p className="subtitle">
          <span className={serverStatus ? "status-indicator status-online" : "status-indicator"} style={{ backgroundColor: serverStatus ? '' : '#ef4444' }}></span>
          {serverStatus ? 'Inference Server Online' : 'Server Offline'}
        </p>
      </header>

      <div className="dashboard-grid">
        {loading ? (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '50px' }}>
            <div className="loader"></div>
            <p style={{ marginTop: '20px', color: 'var(--text-secondary)' }}>Loading model artifacts...</p>
          </div>
        ) : error && !metrics ? (
          <div className="glass-card" style={{ gridColumn: '1 / -1', borderLeft: '4px solid var(--danger)' }}>
            <h3>Connection Error</h3>
            <p>{error}</p>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Ensure the Python API is running (`uvicorn src.api:app --reload`) and models are trained.
            </p>
          </div>
        ) : metrics ? (
          <>
            <div className="glass-card">
              <div className="metric-label">Active Model</div>
              <div className="metric-value" style={{ textTransform: 'capitalize' }}>{metrics.best_model.replace('_', ' ')}</div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Winning model out of Lasso, XGBoost, and RF
              </div>
            </div>

            <div className="glass-card">
              <div className="metric-label">Directional Accuracy</div>
              <div className="metric-value" style={{ color: (metrics[metrics.best_model].metrics.directional_accuracy) > 0.5 ? 'var(--success)' : 'var(--text-primary)' }}>
                {((metrics[metrics.best_model].metrics.directional_accuracy) * 100).toFixed(2)}%
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Hit rate for correctly predicting market direction
              </div>
            </div>

            <div className="glass-card">
              <div className="metric-label">Net Profit (P&L)</div>
              <div className="metric-value" style={{ color: (metrics[metrics.best_model].perf.net_pnl) > 0 ? 'var(--success)' : 'var(--danger)' }}>
                {((metrics[metrics.best_model].perf.net_pnl) * 100).toFixed(2)}%
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Simulated profitability after 1.5bps trading costs
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

export default App;

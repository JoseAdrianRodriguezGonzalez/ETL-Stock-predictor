

import { useState } from "react";
import Sidebar from "../components/layout/Sidebar";
import Header from "../components/layout/Header";
import PredictionCard from "../components/dashboard/PredictionCard";
import ModelInfo from "../components/dashboard/ModelInfo";
import { predictNextSession } from "../services/predictionService";
import type { PredictionResponse } from "../types/prediction";

export default function Dashboard() {
  const [symbol, setSymbol] = useState("^GSPC");
  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePrediction() {
    setLoading(true);
    setError(null);

    try {
      const result = await predictNextSession({ symbol });
      setPrediction(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred.",
      );
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />

      <main className="main-content">
        <Header symbol={symbol} onSymbolChange={setSymbol} />

        <div className="dashboard-content">
          <section className="welcome-section">
            <div>
              <span className="eyebrow">MARKET INTELLIGENCE</span>
              <h2>Market insights, powered by AI.</h2>
              <p>
                Monitor market information and generate machine learning
                forecasts for the next trading session.
              </p>
            </div>
            <div className="market-tag">
              <span className="status-dot" />
              Forecasting platform
            </div>
          </section>

          <div className="dashboard-grid">
            <PredictionCard
              prediction={prediction}
              loading={loading}
              error={error}
              onPredict={handlePrediction}
            />

            <ModelInfo predictionExists={prediction !== null} />
          </div>

          <section className="panel history-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">MARKET DATA</span>
                <h2>Historical performance</h2>
              </div>
              <span className="coming-soon">DATA ENDPOINT PENDING</span>
            </div>

            <div className="history-empty">
              <div className="empty-icon">⌁</div>
              <h3>Historical visualization</h3>
              <p>
                This section will display historical market prices
                once the historical data endpoint is available.
              </p>
            </div>
          </section>

          <footer className="dashboard-footer">
            <span>MARKET PULSE · S&P 500 FORECASTING</span>
            <span>Research &amp; experimental platform</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

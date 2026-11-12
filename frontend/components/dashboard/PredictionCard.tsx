

import type { PredictionResponse } from "../../types/prediction";

interface PredictionCardProps {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string | null;
  onPredict: () => void;
}

export default function PredictionCard({
  prediction,
  loading,
  error,
  onPredict,
}: PredictionCardProps) {
  const positive = (prediction?.predicted_return ?? 0) >= 0;

  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MACHINE LEARNING</span>
          <h2>Next session forecast</h2>
        </div>
        <span className="model-badge">LSTM MODEL</span>
      </div>

      {loading ? (
        <div className="prediction-empty">
          <div className="loader" />
          <p>Running model inference...</p>
        </div>
      ) : error ? (
        <div className="prediction-empty error-state">
          <span>!</span>
          <p>{error}</p>
        </div>
      ) : prediction ? (
        <>
          <div className="forecast-value">
            <strong className={positive ? "positive" : "negative"}>
              {positive ? "+" : ""}
              {prediction.predicted_return_percent.toFixed(4)}%
            </strong>
            <span>Predicted return</span>
          </div>

          <div className="prediction-details">
            <div>
              <span>Last observation</span>
              <strong>{prediction.last_observed_date}</strong>
            </div>
            <div>
              <span>Forecast horizon</span>
              <strong>Next trading session</strong>
            </div>
            <div>
              <span>Input tensor</span>
              <strong>[{prediction.input_shape.join(", ")}]</strong>
            </div>
          </div>
        </>
      ) : (
        <div className="prediction-empty">
          <div className="empty-icon">↗</div>
          <p>Generate a forecast to see the model output.</p>
        </div>
      )}

      <button
        className="primary-button"
        onClick={onPredict}
        disabled={loading}
      >
        {loading ? "Processing..." : "Generate prediction"}
        <span>→</span>
      </button>

      <p className="disclaimer">
        Model output is an estimate, not a guaranteed market outcome.
      </p>
    </section>
  );
}

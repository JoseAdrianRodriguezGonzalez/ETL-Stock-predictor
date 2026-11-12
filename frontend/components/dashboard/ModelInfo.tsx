

interface ModelInfoProps {
  predictionExists: boolean;
}

export default function ModelInfo({ predictionExists }: ModelInfoProps) {
  return (
    <section className="panel model-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MODEL INFORMATION</span>
          <h2>Inference configuration</h2>
        </div>
        <span className="status-dot" />
      </div>

      <div className="model-grid">
        <div>
          <span>Architecture</span>
          <strong>Long Short-Term Memory</strong>
        </div>
        <div>
          <span>Inference engine</span>
          <strong>NVIDIA Triton</strong>
        </div>
        <div>
          <span>Sequence length</span>
          <strong>30 sessions</strong>
        </div>
        <div>
          <span>Input features</span>
          <strong>6 numerical features</strong>
        </div>
        <div>
          <span>Model version</span>
          <strong>1</strong>
        </div>
        <div>
          <span>Last inference</span>
          <strong>{predictionExists ? "Completed" : "Not executed"}</strong>
        </div>
      </div>
    </section>
  );
}

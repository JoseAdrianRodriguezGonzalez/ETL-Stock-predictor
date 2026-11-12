

const navigation = [
  { label: "Overview", icon: "◫", active: true },
  { label: "Market data", icon: "⌁", active: false },
  { label: "Predictions", icon: "⌁", active: false },
  { label: "Model monitoring", icon: "◉", active: false },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">M</div>
        <div>
          <strong>MARKET PULSE</strong>
          <span>QUANTITATIVE ANALYTICS</span>
        </div>
      </div>

      <div className="nav-section">
        <span className="nav-heading">WORKSPACE</span>
        {navigation.map((item) => (
          <a
            href="#"
            key={item.label}
            className={`nav-item ${item.active ? "active" : ""}`}
            onClick={(event) => event.preventDefault()}
          >
            <span>{item.icon}</span>
            {item.label}
          </a>
        ))}
      </div>

      <div className="sidebar-footer">
        <span className="status-dot" />
        <div>
          <strong>Development</strong>
          <span>Local environment</span>
        </div>
      </div>
    </aside>
  );
}

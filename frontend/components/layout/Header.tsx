

interface HeaderProps {
  symbol: string;
  onSymbolChange: (symbol: string) => void;
}

export default function Header({
  symbol,
  onSymbolChange,
}: HeaderProps) {
  return (
    <header className="topbar">
      <div>
        <span className="eyebrow">WORKSPACE / DASHBOARD</span>
        <h1>Market overview</h1>
      </div>

      <div className="header-actions">
        <label className="symbol-selector">
          <span>SYMBOL</span>
          <select
            value={symbol}
            onChange={(event) => onSymbolChange(event.target.value)}
          >
            <option value="^GSPC">S&P 500 · ^GSPC</option>
          </select>
        </label>
        <div className="avatar">MP</div>
      </div>
    </header>
  );
}

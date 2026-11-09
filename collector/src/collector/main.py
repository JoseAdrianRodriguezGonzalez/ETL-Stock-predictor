from collector.config import get_settings
from collector.normalization.market_data import MarketDataNormalizer
from collector.providers.fmp import FMPClient


def main():
    #Se obtienen las variables de entorno
    settings = get_settings()
    # se piden los datos
    with FMPClient(settings) as fmp:
        records = fmp.get_historical_prices(
            symbol="^GSPC", start_date="2020-01-01", end_date="2026-11-09"
        )
    #validar info recibida
    print(f"Records received: {len(records)}")
    if records:
        print(f"Available fields: {list(records[0].keys())}")
        print(f"Sample record: {records[0]}")
    # Normaliza 
    normalized_records=MarketDataNormalizer.normalize_many(records)
    print(f"Normalized records: {len(normalized_records)}")
    print(f"Sample: {normalized_records[0]}")
if __name__=="__main__":
    main()

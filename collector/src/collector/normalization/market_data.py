from datetime import date
from typing import Any, Dict, TypedDict

class MarketData(TypedDict):
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    change: float
    change_percent: float
    vwap: float
class MarketDataNormalizer:
    FIELD_MAPPING  = {
        "changePercent" : "change_percent"
    }
    REQUIRED_FILEDS= {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "change",
        "changePercent",
        "vwap"
    }
    @classmethod
    def normalize(cls, record:dict[str,Any])->MarketData:
        missing= cls.REQUIRED_FILEDS-record.keys()
        if missing:
            raise ValueError(f"Missing some required market data: {sorted(missing)}")
        try:
            normalized = {
                cls.FIELD_MAPPING.get(key,key): value 
                for key,value in record.items()
            }
            normalized["date"]=date.fromisoformat(normalized["date"])
            for field in ("open","high","low","close","change","change_percent","vwap"):
                normalized[field]= float(normalized[field])
            normalized["volume"]=int(normalized["volume"])
            if not normalized["symbol"].strip():
                raise ValueError("Symbol cannot be empty")

            if normalized["volume"]<0:
                raise ValueError("volume cannot be negative")
            if normalized["high"] < max(normalized["open"],normalized["close"],normalized["low"]):
                raise ValueError("high must be >= open, close and low")

            if normalized["low"] > min(normalized["open"],normalized["close"],normalized["high"]):
                raise ValueError("low must be <= open,close and low")
            return MarketData(**normalized)
        except (TypeError,ValueError,KeyError) as exc:
            raise ValueError(f"INvalid market data record: {exc}") from exc 
    @classmethod
    def normalize_many(cls,records:list[dict[str,Any]])->list[MarketData]:
        return [cls.normalize(record) for record in records]

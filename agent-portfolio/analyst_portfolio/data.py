from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["rev_yoy", "op_margin", "fcf_margin_ttm", "debt_to_assets",
            "cash_to_assets", "op_margin_chg_yoy", "earnings_yield_ttm",
            "fcf_yield_ttm", "momentum_21", "momentum_63", "momentum_252",
            "volatility_63", "drawdown_63", "size_pct_sector", "turnover", "news_count"]
DATE_COLUMNS = {"returns_daily": "date", "market_daily": "date",
                "fundamentals_q": "available_date", "valuation_daily": "date",
                "news_events": "usable_date", "consensus": "asof_date",
                "eps_revisions": "snapshot_date", "eps_surprises": "announce_date",
                "price_target_distance": "asof_date"}

def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)): return str(pd.Timestamp(value).date())
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value, (float, np.floating)): return round(float(value), 8) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value

def dumps(value): return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False))

class DataStore:
    """Predictor access is separate from forward-return evaluation."""
    def __init__(self, root, config, development=False):
        self.root, self.config, self.development = Path(root), config, development
        self.tables = {}

    def table(self, name):
        if name not in self.tables:
            frame = pd.read_parquet(self.root / f"{name}.parquet")
            if self.development and name in DATE_COLUMNS:
                col = DATE_COLUMNS[name]
                # The supplied calendar/universe contain future metadata; never use those as features.
                frame = frame[frame[col] <= pd.Timestamp(self.config["cutoff"])].copy()
            self.tables[name] = frame
        return self.tables[name]

    def dates(self, split):
        d = self.table("decision_dates")
        d = d[d.split == split].copy()
        if split == "dev":
            d = d[d.exit_date <= pd.Timestamp(self.config["cutoff"])]
        else:
            if not (d.decision_date > pd.Timestamp(self.config["cutoff"])).all():
                raise ValueError("Test decisions must be strictly after the model cutoff")
        return d

    def universe(self, role, t):
        u = self.table("universe")
        u = u[u.role == role].copy()
        r = self.table("returns_daily")
        counts = r[(r.date <= t) & r.ret_total.notna()].groupby("gvkey").size()
        # Recompute eligibility using history visible at this date; do not use future counts/lifetimes.
        return u[u.gvkey.map(counts).fillna(0) >= 252].sort_values("gvkey")

    def latest(self, name, key, date_col, t, strict=False, max_age=None, extra=None):
        d = self.table(name)
        mask = (d.gvkey == key) & ((d[date_col] < t) if strict else (d[date_col] <= t))
        if max_age: mask &= d[date_col] >= t - pd.Timedelta(days=max_age)
        if extra is not None: mask &= extra(d)
        d = d[mask].sort_values(date_col)
        return {} if d.empty else clean(d.iloc[-1].to_dict())

    def pack(self, key, t, price_cutoff=None):
        t = pd.Timestamp(t); pt = pd.Timestamp(price_cutoff) if price_cutoff is not None else t
        f = self.latest("fundamentals_q", key, "available_date", t, strict=True, max_age=180)
        v = self.latest("valuation_daily", key, "date", pt, max_age=7)
        r = self.table("returns_daily")
        r = r[(r.gvkey == key) & (r.date <= pt)].sort_values("date").tail(253)
        tech = {}
        for n in [21, 63, 252]:
            x = r.ret_total.tail(n)
            tech[f"momentum_{n}"] = float((1+x).prod()-1) if len(x)==n and x.notna().all() else None
        x = r.ret_total.tail(63)
        wealth = (1+x).cumprod()
        tech.update(volatility_63=float(x.std()*np.sqrt(252)),
                    drawdown_63=float((wealth/wealth.cummax()-1).min()))
        if not r.empty:
            tech.update(r.iloc[-1][["size_pct_sector", "turnover"]].to_dict())
        news = self.table("news_events")
        news = news[(news.gvkey == key) & (news.usable_date < t) &
                    (news.usable_date >= t-pd.Timedelta(days=self.config["news_window_days"])) &
                    ~news.is_analyst_action & ~news.modified_later & news.headline_text.notna()]
        news = news.sort_values(["usable_date", "event_id"], ascending=[False, True])
        events = []
        for _, row in news.head(self.config["news_limit"]).iterrows():
            events.append({"id": "N:"+str(row.event_id), "date": str(row.usable_date.date()),
                           "category": row.event_type, "headline": row.headline_text,
                           "text": str(row.situation_text or "")[:1100]})
        # Strip identifiers, future outcome fields, analyst-derived information from blind evidence.
        fundamental = {k:v for k,v in f.items() if k not in ["gvkey", "fiscal_year", "fiscal_qtr"]}
        valuation = {k:v for k,v in v.items() if k not in ["gvkey"]}
        features = {k: fundamental.get(k, valuation.get(k, tech.get(k))) for k in FEATURES}
        features["news_count"] = len(news)
        evidence = []
        for prefix, obj in [("F", fundamental), ("V", valuation), ("T", tech)]:
            evidence += [{"id": f"{prefix}:{key}:{k}", "field": k, "value": val}
                         for k,val in obj.items() if val is not None]
        evidence += events
        return clean({"id":key, "decision_date":t, "price_cutoff":pt,
                      "fundamental":fundamental, "valuation":valuation, "technical":tech,
                      "news":events, "features":features, "evidence":evidence})

    def analyst(self, key, t):
        t = pd.Timestamp(t)
        c = self.latest("consensus",key,"asof_date",t,max_age=0,
            extra=lambda d: (d.asof_date==t) & (d.definition=="B") &
                (d.max_age==self.config["consensus_max_age"]) &
                (d.age_basis==self.config["consensus_age_basis"]))
        if c.get("n_brokers",0) < self.config["min_brokers"]: c = {}
        p = self.latest("price_target_distance",key,"asof_date",t,
                        extra=lambda d:(d.asof_date==t)&(d.max_age==self.config["consensus_max_age"]))
        e = self.latest("eps_revisions",key,"snapshot_date",t,max_age=35,
                        extra=lambda d:(d.horizon=="FY1") & ~d.small_base)
        s = self.latest("eps_surprises",key,"announce_date",t,strict=True,max_age=120,
                        extra=lambda d:~d.small_base)
        return clean({"consensus":c, "price_target":p, "eps_revisions":e, "eps_surprise":s})

    def history(self, keys, t):
        r = self.table("returns_daily")
        r = r[(r.date<=pd.Timestamp(t)) & r.gvkey.isin(keys)]
        return r.pivot(index="date",columns="gvkey",values="ret_total").sort_index().tail(self.config["covariance_window"])

    def file_hashes(self):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.glob("*.parquet")}

    def forward_returns(self, keys, entry, exit):
        # Close-to-close: entry day's return is deliberately excluded.
        r = self.table("returns_daily")
        sessions = self.table("sessions").date
        dates = sessions[(sessions>entry)&(sessions<=exit)]
        r = r[(r.date>entry)&(r.date<=exit)&r.gvkey.isin(keys)]
        wide = r.pivot(index="date",columns="gvkey",values="ret_total").reindex(index=dates,columns=keys)
        # No silently dropping delisted / missing securities.
        ok = wide.notna().all() & (len(dates)==self.config["horizon_sessions"])
        return ((1+wide).prod()-1).where(ok), wide

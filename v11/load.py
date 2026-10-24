"""Load one split into a tidy table: one row per transaction."""
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

XRP_TOKEN = "0x1111222211112222111122221111222211112222"


def _asset_units(wire):
    """Amount JSON -> (asset key, units as float). XRP comes as a string of drops."""
    if isinstance(wire, str):
        return "XRP", int(wire) / 1e6
    return f"{wire['issuer']}:{wire['currency']}", float(Decimal(wire["value"]))


def _rates(directory: Path) -> pd.Series:
    r = pd.read_csv(directory / "market_rates.csv")
    return pd.Series(r["PRICE_MICRO_USD"].to_numpy() / 1e6,
                     index=pd.MultiIndex.from_arrays([r["TIMESTAMP"].astype(int), r["ASSET"]]))


def _usd(rates: pd.Series, ts: np.ndarray, assets: pd.Series, units: np.ndarray) -> np.ndarray:
    hour = (ts // 3600) * 3600
    idx = pd.MultiIndex.from_arrays([hour, assets.to_numpy()])
    return units * rates.reindex(idx).to_numpy()


def load_split(directory) -> pd.DataFrame:
    directory = Path(directory)
    tx = pd.read_parquet(directory / "transactions.parquet")
    tr = pd.read_parquet(directory / "transfers.parquet")
    rates = _rates(directory)

    df = pd.DataFrame({
        "hash": tx["TRANSACTION_HASH"].to_numpy(),
        "ts": tx["TIMESTAMP"].astype("int64").to_numpy(),
        "ledger": tx["LEDGER_INDEX"].astype("int64").to_numpy(),
        "tx_index": tx["TRANSACTION_INDEX"].astype("int64").to_numpy(),
        "sender": tx["ACCOUNT"].to_numpy(),
        "fee": tx["SOURCE_FEE"].astype("int64").to_numpy(),
        "seq": tx["SEQUENCE"].astype("int64").to_numpy(),
    })
    extra = tx["EXTRA_TRANSACTION_FIELDS"].map(json.loads)
    df["receiver"] = extra.map(lambda e: e["Destination"]).to_numpy()
    d = tx["DELIVERED_AMOUNT"].map(lambda s: _asset_units(json.loads(s)))
    df["d_asset"] = d.map(lambda x: x[0]).to_numpy()
    df["d_units"] = d.map(lambda x: x[1]).to_numpy()
    sm = extra.map(lambda e: _asset_units(e["SendMax"]) if "SendMax" in e else (None, np.nan))
    df["sm_asset"] = sm.map(lambda x: x[0]).to_numpy()
    df["sm_units"] = sm.map(lambda x: x[1]).to_numpy()
    df["has_paths"] = extra.map(lambda e: "Paths" in e).astype(int).to_numpy()

    # Transfers: first leg = actual source debit; count legs
    tr = tr.assign(TRANSFER_INDEX=tr["TRANSFER_INDEX"].astype(int)).sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    tr["asset"] = np.where(tr["TOKEN_ADDRESS"] == XRP_TOKEN, "XRP", tr["TOKEN_ADDRESS"] + ":" + tr["SYMBOL"])
    first = tr.groupby("TRANSACTION_HASH").first()
    legs = tr.groupby("TRANSACTION_HASH").size()
    df["n_legs"] = legs.reindex(df["hash"]).to_numpy()
    df["src_asset"] = first["asset"].reindex(df["hash"]).to_numpy()
    df["src_units"] = first["VALUE"].reindex(df["hash"]).to_numpy()
    df["lp1"] = np.where(df["n_legs"] > 1, first["TO_ADDRESS"].reindex(df["hash"]).to_numpy(), None)

    df["d_usd"] = _usd(rates, df["ts"].to_numpy(), df["d_asset"], df["d_units"].to_numpy())
    df["src_usd"] = _usd(rates, df["ts"].to_numpy(), df["src_asset"], df["src_units"].to_numpy())
    df["conv_cost"] = 1 - df["d_usd"] / df["src_usd"]          # spread paid on conversion
    df["sm_slack"] = df["sm_units"] / df["src_units"]          # SendMax ceiling vs actual debit
    df["cross"] = (df["src_asset"] != df["d_asset"]).astype(int)

    labels = directory / "labels.csv"
    if labels.exists():
        lab = pd.read_csv(labels).rename(columns={"TRANSACTION_HASH": "hash"})
        df = df.merge(lab, on="hash", how="left")
    return df.sort_values(["ts", "tx_index"], kind="mergesort").reset_index(drop=True)

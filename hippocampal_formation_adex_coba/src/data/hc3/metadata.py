from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from zipfile import ZipFile

import pandas as pd


@dataclass
class UnitMeta:
    region: str
    cell_type: str


class HC3Metadata:
    """Lightweight lookup helper for hc-3 metadata tables."""

    def __init__(self, table):
        self.table = table
        self._norm_cols = {c.lower(): c for c in table.columns}

    def _pick_col(self, candidates: Iterable[str]) -> Optional[str]:
        for cand in candidates:
            c_lower = cand.lower()
            if c_lower in self._norm_cols:
                return self._norm_cols[c_lower]
            for col_l, orig in self._norm_cols.items():
                if col_l.endswith(c_lower):
                    return orig
        return None

    def lookup(self, topdir: str, electrode: int, cluster: int) -> Optional[UnitMeta]:
        df = self.table
        topdir_col = self._pick_col(["topdir", "animal", "subject"])
        ele_col = self._pick_col(["ele", "electrode", "shank", "tet"])
        clu_col = self._pick_col(["clu", "cluster", "unit"])
        region_col = self._pick_col(["region", "area", "structure"])
        celltype_col = self._pick_col(["celltype", "cell_type", "type", "cell_class"])

        subset = df
        if topdir_col is not None and topdir in df[topdir_col].astype(str).unique():
            subset = subset[subset[topdir_col].astype(str) == str(topdir)]
        if ele_col is not None:
            subset = subset[subset[ele_col].astype(int) == int(electrode)]
        if clu_col is not None:
            subset = subset[subset[clu_col].astype(int) == int(cluster)]

        if subset.empty:
            return None

        region_val = str(subset.iloc[0][region_col]) if region_col is not None else "unknown"
        celltype_val = str(subset.iloc[0][celltype_col]) if celltype_col is not None else "unknown"
        return UnitMeta(region=region_val, cell_type=celltype_val)


def _clean_metadata_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [str(c).strip().lower() for c in df_raw.columns]
    if any(c in {"topdir", "animal", "subject"} for c in cols_lower):
        return df_raw

    header_row = None
    for idx, row in df_raw.iterrows():
        vals = [str(v).strip().lower() for v in row.tolist() if pd.notna(v)]
        if any(v in {"topdir", "animal", "subject"} for v in vals):
            header_row = idx
            break
    if header_row is None:
        header_row = 0

    header_vals = df_raw.iloc[header_row].fillna("").map(lambda v: str(v).strip())
    df = df_raw.iloc[header_row + 1 :].copy()
    df.columns = header_vals
    df = df.loc[:, ~df.columns.isna()]
    df = df.dropna(how="all")
    return df


def _select_excel_sheet(sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for name, df in sheets.items():
        if str(name).strip().lower() == "cell":
            return df
    for name, df in sheets.items():
        if "cell" in str(name).strip().lower():
            return df
    return next(iter(sheets.values()))


def _load_excel(source) -> pd.DataFrame:
    sheets = pd.read_excel(source, sheet_name=None, header=None)
    if isinstance(sheets, dict):
        df_raw = _select_excel_sheet(sheets)
    else:
        df_raw = sheets
    return _clean_metadata_table(df_raw)


def _load_sqlite_table(path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cur.fetchall()]
        table_name = None
        for name in table_names:
            if name.lower() == "cell":
                table_name = name
                break
        if table_name is None:
            for name in table_names:
                if "cell" in name.lower():
                    table_name = name
                    break
        if table_name is None and table_names:
            table_name = table_names[0]
        if table_name is None:
            raise ValueError("No tables found in metadata database")
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def load_metadata_table(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()

    if suffix == ".zip":
        with ZipFile(p, "r") as zf:
            xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            db_names = [n for n in zf.namelist() if n.lower().endswith(".db")]
            csv_names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".tsv"))]
            if xlsx_names:
                with zf.open(xlsx_names[0]) as f:
                    return _load_excel(f)
            if db_names:
                with zf.open(db_names[0]) as f:
                    db_bytes = f.read()
                tmp_db = p.with_suffix(".tmp.db")
                tmp_db.write_bytes(db_bytes)
                try:
                    return _load_sqlite_table(tmp_db)
                finally:
                    tmp_db.unlink(missing_ok=True)
            if csv_names:
                with zf.open(csv_names[0]) as f:
                    sep = "\t" if csv_names[0].lower().endswith(".tsv") else ","
                    return _clean_metadata_table(pd.read_csv(f, sep=sep))

    if suffix in {".xlsx", ".xls"}:
        return _load_excel(p)

    if suffix == ".db":
        return _load_sqlite_table(p)

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return _clean_metadata_table(pd.read_csv(p, sep=sep))

    raise ValueError(f"Unsupported metadata format: {p}")


def load_metadata(path: str | Path) -> HC3Metadata:
    table = load_metadata_table(path)
    return HC3Metadata(table)

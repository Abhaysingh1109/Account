import os
import re
import socket
import importlib
import warnings
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from numpy.typing import NDArray


app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

DATE_HINTS = ["date", "transaction_date", "posting_date", "invoice_date", "entry_date"]
AMOUNT_HINTS = ["amount", "value", "net_amount", "total", "balance", "debit", "credit"]
TYPE_HINTS = ["type", "transaction_type", "entry_type", "nature", "flow"]
CATEGORY_HINTS = ["category", "account", "expense_category", "gl", "ledger"]
CUSTOMER_HINTS = ["customer", "client", "party", "name"]
AR_HINTS = ["receivable", "ar", "accounts_receivable"]
AP_HINTS = ["payable", "ap", "accounts_payable"]


@dataclass
class ColumnMap:
    date: Optional[str]
    amount: Optional[str]
    txn_type: Optional[str]
    category: Optional[str]
    customer: Optional[str]
    receivable: Optional[str]
    payable: Optional[str]


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def find_column(columns: list[str], hints: list[str]) -> Optional[str]:
    norm_to_original = {_normalize_col(c): c for c in columns}
    norm_cols = list(norm_to_original.keys())

    for hint in hints:
        hint_norm = _normalize_col(hint)
        for col in norm_cols:
            if hint_norm == col:
                return norm_to_original[col]

    for hint in hints:
        hint_norm = _normalize_col(hint)
        for col in norm_cols:
            if hint_norm in col:
                return norm_to_original[col]

    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    cols = [str(c) for c in df.columns.tolist()]
    mapping = ColumnMap(
        date=find_column(cols, DATE_HINTS),
        amount=find_column(cols, AMOUNT_HINTS),
        txn_type=find_column(cols, TYPE_HINTS),
        category=find_column(cols, CATEGORY_HINTS),
        customer=find_column(cols, CUSTOMER_HINTS),
        receivable=find_column(cols, AR_HINTS),
        payable=find_column(cols, AP_HINTS),
    )
    return infer_required_columns(df, mapping)


def _valid_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.notna().mean())


def infer_required_columns(df: pd.DataFrame, mapping: ColumnMap) -> ColumnMap:
    """Fallback inference based on values when header hints are missing."""
    cols = [str(c) for c in df.columns.tolist()]
    working = ColumnMap(
        date=mapping.date,
        amount=mapping.amount,
        txn_type=mapping.txn_type,
        category=mapping.category,
        customer=mapping.customer,
        receivable=mapping.receivable,
        payable=mapping.payable,
    )

    if working.date is None:
        best_date_col = None
        best_date_ratio = 0.0
        for col in cols:
            parsed = cast(pd.Series, pd.to_datetime(df[col], errors="coerce"))
            ratio = _valid_ratio(parsed)
            if ratio > best_date_ratio:
                best_date_ratio = ratio
                best_date_col = col
        # Require a meaningful proportion of parseable dates to avoid false matches.
        if best_date_col is not None and best_date_ratio >= 0.5:
            working.date = best_date_col

    if working.amount is None:
        best_amount_col = None
        best_amount_ratio = 0.0
        for col in cols:
            numeric = coerce_numeric(df[col])
            ratio = _valid_ratio(numeric)
            if ratio > best_amount_ratio:
                best_amount_ratio = ratio
                best_amount_col = col
        # Require sufficient numeric coverage for amount inference.
        if best_amount_col is not None and best_amount_ratio >= 0.5:
            working.amount = best_amount_col

    return working


def coerce_numeric(series: pd.Series) -> pd.Series:
    as_text = series.astype(str).str.strip()
    is_parenthesized = as_text.str.match(r"^\(.*\)$")
    cleaned = as_text.str.replace(r"[,$]", "", regex=True)
    cleaned = cleaned.str.replace(r"[()]", "", regex=True)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = pd.Series(numeric, index=series.index)
    mask = is_parenthesized.fillna(False).astype(bool)
    numeric = numeric.mask(mask, -numeric.abs())
    return numeric


def classify_transaction(row: pd.Series, txn_type_col: Optional[str], category_col: Optional[str]) -> str:
    text_parts = []
    if txn_type_col and pd.notna(row.get(txn_type_col)):
        text_parts.append(str(row[txn_type_col]).lower())
    if category_col and pd.notna(row.get(category_col)):
        text_parts.append(str(row[category_col]).lower())
    text = " ".join(text_parts)

    compact = re.sub(r"[^a-z0-9]+", "", text)
    tokens = set(re.findall(r"[a-z0-9]+", text))

    if any(k in text for k in ["revenue", "sale", "income", "inflow", "receipt"]):
        return "Revenue"
    if any(k in text for k in ["expense", "cost", "rent", "salary", "marketing", "outflow", "payment"]):
        return "Expense"
    if "receivable" in text or "accounts receivable" in text or "ar" in tokens or compact.endswith("ar"):
        return "Receivable"
    if "payable" in text or "accounts payable" in text or "ap" in tokens or compact.endswith("ap"):
        return "Payable"

    return "Other"


def signed_amount(amount: float, label: str) -> float:
    if label in {"Expense", "Payable"}:
        return -abs(amount)
    return float(amount)


def preprocess(df: pd.DataFrame, mapping: ColumnMap) -> pd.DataFrame:
    if mapping.date is None or mapping.amount is None:
        raise ValueError(
            "Could not find required columns. File needs at least one Date column and one Amount column."
        )

    data = df.copy()
    data["_date"] = pd.to_datetime(data[mapping.date], errors="coerce")
    data["_amount_raw"] = coerce_numeric(data[mapping.amount])

    if mapping.txn_type or mapping.category:
        data["_class"] = data.apply(lambda r: classify_transaction(r, mapping.txn_type, mapping.category), axis=1)
    else:
        data["_class"] = np.where(data["_amount_raw"] < 0, "Expense", "Revenue")

    data["_amount"] = [
        signed_amount(a, c) if pd.notna(a) else np.nan
        for a, c in zip(data["_amount_raw"], data["_class"])
    ]

    data = data.dropna(subset=["_date", "_amount"]).copy()
    data["_month"] = data["_date"].dt.to_period("M").dt.to_timestamp()

    if mapping.customer and mapping.customer in data.columns:
        data["_customer"] = data[mapping.customer].astype(str)
    else:
        data["_customer"] = "Unknown"

    if mapping.category and mapping.category in data.columns:
        data["_category"] = data[mapping.category].astype(str)
    else:
        data["_category"] = "Other"

    return data


def monthly_metrics(data: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        data.groupby(["_month", "_class"], as_index=False)["_amount"]
        .sum()
        .pivot(index="_month", columns="_class", values="_amount")
        .fillna(0)
    )

    for col in ["Revenue", "Expense", "Receivable", "Payable"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Revenue"] = pivot["Revenue"].abs()
    pivot["Expense"] = pivot["Expense"].abs()
    pivot["Profit"] = pivot["Revenue"] - pivot["Expense"]
    pivot["NetCashFlow"] = pivot["Profit"]

    return pivot.sort_index()


def period_metrics(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    grouped = (
        data.groupby([pd.Grouper(key="_date", freq=freq), "_class"], as_index=False)["_amount"]
        .sum()
        .pivot(index="_date", columns="_class", values="_amount")
        .fillna(0)
    )

    for col in ["Revenue", "Expense", "Receivable", "Payable"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["Revenue"] = grouped["Revenue"].abs()
    grouped["Expense"] = grouped["Expense"].abs()
    grouped["Profit"] = grouped["Revenue"] - grouped["Expense"]
    grouped["NetCashFlow"] = grouped["Profit"]
    return grouped.sort_index()


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)))


def _predict_naive(train: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, train[-1], dtype=float)


def _predict_moving_average(train: np.ndarray, horizon: int, window: int = 4) -> np.ndarray:
    w = max(1, min(window, len(train)))
    return np.full(horizon, float(np.mean(train[-w:])), dtype=float)


def _predict_drift(train: np.ndarray, horizon: int) -> np.ndarray:
    if len(train) < 2:
        return _predict_naive(train, horizon)
    slope = (train[-1] - train[0]) / (len(train) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return train[-1] + slope * steps


def _predict_linear_trend(train: np.ndarray, horizon: int) -> np.ndarray:
    if len(train) < 2:
        return _predict_naive(train, horizon)
    x = np.arange(len(train), dtype=float)
    coef = np.polyfit(x, train, 1)
    x_future = np.arange(len(train), len(train) + horizon, dtype=float)
    return coef[0] * x_future + coef[1]


def _predict_seasonal_naive(train: np.ndarray, horizon: int, season_len: int) -> np.ndarray:
    if season_len <= 0 or len(train) < season_len:
        return _predict_naive(train, horizon)
    base = train[-season_len:]
    repeats = int(np.ceil(horizon / season_len))
    return np.tile(base, repeats)[:horizon].astype(float)


def _predict_holt_winters(
    train: NDArray[np.float64], horizon: int, season_len: int
) -> Optional[NDArray[np.float64]]:
    try:
        hw_module = importlib.import_module("statsmodels.tsa.holtwinters")
        ExponentialSmoothing = getattr(hw_module, "ExponentialSmoothing")
    except Exception:
        return None
    if len(train) < max(8, season_len * 2):
        return None
    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if season_len >= 2 else None,
            seasonal_periods=season_len if season_len >= 2 else None,
            damped_trend=True,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True, use_brute=True)
        fc = np.asarray(fitted.forecast(horizon), dtype=float)
        return fc
    except Exception:
        return None


def _predict_auto_ar(
    train: NDArray[np.float64], horizon: int, max_lag: int = 8
) -> Optional[NDArray[np.float64]]:
    n = len(train)
    if n < 6:
        return None
    lag = max(2, min(max_lag, n - 1))
    try:
        y = train.copy()
        preds: list[float] = []
        for _ in range(horizon):
            x = np.arange(lag, dtype=float)
            window = y[-lag:]
            coef = np.polyfit(x, window, 1)
            nxt = float(coef[0] * lag + coef[1])
            preds.append(nxt)
            y = np.append(y, nxt)
        return np.asarray(preds, dtype=float)
    except Exception:
        return None


def _predict_sarima(
    train: NDArray[np.float64], horizon: int, season_len: int
) -> Optional[NDArray[np.float64]]:
    if len(train) < max(12, season_len * 2):
        return None
    try:
        sarimax_module = importlib.import_module("statsmodels.tsa.statespace.sarimax")
        SARIMAX = getattr(sarimax_module, "SARIMAX")
    except Exception:
        return None

    # Small bounded search to balance quality and speed.
    candidate_orders = [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
    seasonal_orders = [(0, 1, 1, season_len), (1, 1, 1, season_len)] if season_len >= 2 else [(0, 0, 0, 0)]
    best_aic = float("inf")
    best_result = None

    for order in candidate_orders:
        for seasonal_order in seasonal_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False)
                aic = float(result.aic)
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_result = result
            except Exception:
                continue

    if best_result is None:
        return None

    try:
        pred = np.asarray(best_result.forecast(steps=horizon), dtype=float)
        return pred
    except Exception:
        return None


def _fit_and_forecast(series: pd.Series, horizon: int, season_len: int) -> dict:
    clean = np.asarray(series.astype(float).to_numpy(), dtype=float)
    n = len(clean)
    if n < 3:
        fallback = _predict_naive(clean, horizon) if n else np.zeros(horizon, dtype=float)
        return {
            "model": "insufficient_history_naive",
            "mae": None,
            "forecast": fallback,
            "lower": fallback,
            "upper": fallback,
        }

    validation = int(min(max(2, int(n // 4)), 8))
    train = clean[:-validation]
    val = clean[-validation:]
    if len(train) < 2:
        train = clean[:-1]
        val = clean[-1:]

    candidates: dict[str, NDArray[np.float64]] = {
        "naive": _predict_naive(train, len(val)),
        "moving_average": _predict_moving_average(train, len(val), window=4),
        "drift": _predict_drift(train, len(val)),
        "linear_trend": _predict_linear_trend(train, len(val)),
        "seasonal_naive": _predict_seasonal_naive(train, len(val), season_len=season_len),
    }
    hw_pred = _predict_holt_winters(train, len(val), season_len)
    if hw_pred is not None:
        candidates["holt_winters"] = hw_pred
    ar_pred = _predict_auto_ar(train, len(val))
    if ar_pred is not None:
        candidates["autoregressive_trend"] = ar_pred
    sarima_pred = _predict_sarima(train, len(val), season_len=season_len)
    if sarima_pred is not None:
        candidates["sarima"] = sarima_pred

    scores = {name: _safe_mae(val, pred) for name, pred in candidates.items()}
    best_model = min(scores, key=lambda model_name: scores[model_name])
    best_mae = float(scores[best_model])

    full_candidates: dict[str, NDArray[np.float64]] = {
        "naive": _predict_naive(clean, horizon),
        "moving_average": _predict_moving_average(clean, horizon, window=4),
        "drift": _predict_drift(clean, horizon),
        "linear_trend": _predict_linear_trend(clean, horizon),
        "seasonal_naive": _predict_seasonal_naive(clean, horizon, season_len=season_len),
    }
    full_hw = _predict_holt_winters(clean, horizon, season_len)
    if full_hw is not None:
        full_candidates["holt_winters"] = full_hw
    full_ar = _predict_auto_ar(clean, horizon)
    if full_ar is not None:
        full_candidates["autoregressive_trend"] = full_ar
    full_sarima = _predict_sarima(clean, horizon, season_len=season_len)
    if full_sarima is not None:
        full_candidates["sarima"] = full_sarima

    if best_model not in full_candidates:
        available = [name for name in scores.keys() if name in full_candidates]
        if available:
            best_model = min(available, key=lambda model_name: scores[model_name])
        else:
            best_model = "naive"
    forecast = full_candidates[best_model]

    residual_std = float(np.std(val - candidates[best_model])) if len(val) else 0.0
    z_80 = 1.28
    margin = z_80 * residual_std
    lower = forecast - margin
    upper = forecast + margin

    return {
        "model": best_model,
        "mae": best_mae,
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
    }


def build_forecast(data: pd.DataFrame) -> dict:
    monthly = period_metrics(data, "MS")
    weekly = period_metrics(data, "W-MON")

    monthly_horizon = 3
    weekly_horizon = 8
    monthly_future_idx = pd.date_range(
        monthly.index.max() + pd.offsets.MonthBegin(1), periods=monthly_horizon, freq="MS"
    )
    weekly_future_idx = pd.date_range(
        weekly.index.max() + pd.offsets.Week(1), periods=weekly_horizon, freq="W-MON"
    )

    monthly_series = ["Revenue", "Expense", "Profit", "NetCashFlow"]
    weekly_series = ["Revenue", "Expense", "Profit", "NetCashFlow"]

    monthly_out: dict[str, dict] = {"models": {}, "history": {}, "forecast": {}, "bands": {}}
    weekly_out: dict[str, dict] = {"models": {}, "history": {}, "forecast": {}, "bands": {}}

    for col in monthly_series:
        fit = _fit_and_forecast(monthly[col], monthly_horizon, season_len=12)
        monthly_out["history"][col] = [float(v) for v in monthly[col].to_list()]
        monthly_out["forecast"][col] = [float(v) for v in fit["forecast"]]
        monthly_out["bands"][col] = {
            "lower": [float(v) for v in fit["lower"]],
            "upper": [float(v) for v in fit["upper"]],
        }
        monthly_out["models"][col] = {"name": fit["model"], "mae": fit["mae"]}

    for col in weekly_series:
        fit = _fit_and_forecast(weekly[col], weekly_horizon, season_len=4)
        weekly_out["history"][col] = [float(v) for v in weekly[col].to_list()]
        weekly_out["forecast"][col] = [float(v) for v in fit["forecast"]]
        weekly_out["bands"][col] = {
            "lower": [float(v) for v in fit["lower"]],
            "upper": [float(v) for v in fit["upper"]],
        }
        weekly_out["models"][col] = {"name": fit["model"], "mae": fit["mae"]}

    return {
        "monthly": {
            "historyLabels": [d.strftime("%Y-%m") for d in monthly.index.to_list()],
            "futureLabels": [d.strftime("%Y-%m") for d in monthly_future_idx.to_list()],
            **monthly_out,
        },
        "weekly": {
            "historyLabels": [d.strftime("%Y-%m-%d") for d in weekly.index.to_list()],
            "futureLabels": [d.strftime("%Y-%m-%d") for d in weekly_future_idx.to_list()],
            **weekly_out,
        },
    }


def generate_insights(monthly: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if monthly.empty:
        return ["Not enough data to generate insights."]

    latest = monthly.iloc[-1]
    margin = (latest["Profit"] / latest["Revenue"] * 100) if latest["Revenue"] else 0

    if margin < 10:
        insights.append("Low profit margin: review high-cost categories and set monthly spending caps.")
    elif margin > 25:
        insights.append("Healthy margin: consider reinvesting part of profit into channels with measurable ROI.")

    if len(monthly) >= 3 and monthly["Revenue"].iloc[-1] < monthly["Revenue"].iloc[-3]:
        insights.append("Revenue softened in recent months: weekly forecasting flags pressure and should guide pipeline goals.")

    expense_by_cat = (
        data[data["_class"] == "Expense"].groupby("_category")["_amount"].sum().abs().sort_values(ascending=False)
    )
    if not expense_by_cat.empty:
        top_cat = expense_by_cat.index[0]
        top_val = expense_by_cat.iloc[0]
        total_expense = expense_by_cat.sum()
        share = (top_val / total_expense * 100) if total_expense else 0
        if share > 35:
            insights.append(f"Expense concentration: {top_cat} is {share:.1f}% of all expenses.")

    ar = monthly["Receivable"].abs().iloc[-1] if "Receivable" in monthly.columns else 0
    rev = monthly["Revenue"].iloc[-1]
    if rev > 0 and ar / rev > 0.6:
        insights.append("High receivables vs revenue: tighten collection cycles and send payment reminders earlier.")

    if latest["Profit"] < 0:
        insights.append("Current month is loss-making: freeze non-essential spend and review runway weekly.")

    if not insights:
        insights.append("Performance looks stable. Keep monthly budgeting and compare planned vs actuals.")

    return insights


def generate_structure_recommendations(
    raw_df: pd.DataFrame,
    data: pd.DataFrame,
    monthly: pd.DataFrame,
    mapping: ColumnMap,
) -> tuple[list[dict], dict]:
    recommendations: list[dict] = []
    raw_rows = max(int(len(raw_df)), 1)
    used_rows = int(len(data))
    valid_ratio = used_rows / raw_rows

    unknown_category_ratio = float((data["_category"] == "Other").mean()) if used_rows else 1.0
    unknown_customer_ratio = float((data["_customer"] == "Unknown").mean()) if used_rows else 1.0
    other_class_ratio = float((data["_class"] == "Other").mean()) if used_rows else 1.0

    scores = {
        "Data Quality": max(0.0, min(100.0, valid_ratio * 100)),
        "Categorization": max(0.0, min(100.0, (1 - unknown_category_ratio) * 100)),
        "Transaction Mapping": max(0.0, min(100.0, (1 - other_class_ratio) * 100)),
        "Customer Coverage": max(0.0, min(100.0, (1 - unknown_customer_ratio) * 100)),
        "Cash Discipline": 70.0,
    }

    if monthly.empty:
        scores["Cash Discipline"] = 50.0
    else:
        latest = monthly.iloc[-1]
        rev = float(latest["Revenue"])
        ar = float(abs(latest["Receivable"]))
        payable = float(abs(latest["Payable"]))
        profit = float(latest["Profit"])
        cash_score = 72.0
        if rev > 0:
            ar_pressure = ar / rev
            if ar_pressure > 0.8:
                cash_score -= 30
            elif ar_pressure > 0.5:
                cash_score -= 15
        if profit < 0:
            cash_score -= 15
        if payable > rev * 0.7 and rev > 0:
            cash_score -= 12
        scores["Cash Discipline"] = max(0.0, min(100.0, cash_score))

    avg_score = float(np.mean(list(scores.values())))
    maturity = (
        "Advanced"
        if avg_score >= 80
        else "Developing"
        if avg_score >= 60
        else "Foundation Needed"
    )

    if valid_ratio < 0.85:
        dropped_rows = raw_rows - used_rows
        recommendations.append(
            {
                "priority": "High",
                "title": "Improve Raw Data Quality",
                "finding": f"{dropped_rows:,} rows were dropped during cleaning because date/amount parsing failed.",
                "action": "Standardize export format to ISO date (YYYY-MM-DD) and numeric amount columns before upload.",
                "impact": "Cleaner source data improves trend reliability and forecasting accuracy.",
            }
        )

    if mapping.category is None or unknown_category_ratio > 0.3:
        recommendations.append(
            {
                "priority": "High",
                "title": "Strengthen Chart of Accounts Mapping",
                "finding": "A large share of transactions are uncategorized or mapped to generic buckets.",
                "action": "Define a fixed account taxonomy (Revenue, COGS, Opex, AR, AP) and enforce category picklists.",
                "impact": "Better category hygiene unlocks tighter budget control and more useful variance analysis.",
            }
        )

    if mapping.customer is None or unknown_customer_ratio > 0.4:
        recommendations.append(
            {
                "priority": "Medium",
                "title": "Increase Customer Tagging Coverage",
                "finding": "Many rows do not carry a customer/client identifier.",
                "action": "Require customer codes on revenue entries and keep a simple customer master reference.",
                "impact": "Customer-level profitability and concentration risks become visible.",
            }
        )

    if mapping.txn_type is None:
        recommendations.append(
            {
                "priority": "Medium",
                "title": "Add Transaction Type Field",
                "finding": "The dataset lacks an explicit transaction type column (inflow/outflow/AR/AP).",
                "action": "Include transaction type in exports and validate values with dropdown-like controlled labels.",
                "impact": "Classification quality improves and receivable/payable reporting becomes more precise.",
            }
        )

    if not monthly.empty:
        latest = monthly.iloc[-1]
        rev = float(latest["Revenue"])
        ar = float(abs(latest["Receivable"]))
        if rev > 0 and (ar / rev) > 0.6:
            recommendations.append(
                {
                    "priority": "High",
                    "title": "Reduce Receivable Pressure",
                    "finding": "Accounts receivable is high compared with current revenue.",
                    "action": "Implement staged collection reminders and set invoice due-date SLAs by customer tier.",
                    "impact": "Lower DSO improves liquidity and reduces cash crunch risk.",
                }
            )

    if not recommendations:
        recommendations.append(
            {
                "priority": "Low",
                "title": "Maintain Current Structure",
                "finding": "Your current accounting structure appears consistent and well tagged.",
                "action": "Keep monthly account review rituals and monitor the score trend over time.",
                "impact": "Sustains reporting quality as volume grows.",
            }
        )

    structure_scores = [{"dimension": k, "score": round(float(v), 1)} for k, v in scores.items()]
    return recommendations[:5], {
        "scores": structure_scores,
        "overallScore": round(avg_score, 1),
        "maturity": maturity,
    }


def build_payload(raw_df: pd.DataFrame, data: pd.DataFrame, monthly: pd.DataFrame, mapping: ColumnMap) -> dict:
    months = [d.strftime("%Y-%m") for d in monthly.index.to_list()]
    revenue = [float(v) for v in monthly["Revenue"].to_list()]
    expenses = [float(v) for v in monthly["Expense"].to_list()]
    profit = [float(v) for v in monthly["Profit"].to_list()]
    receivable = [float(v) for v in monthly["Receivable"].abs().to_list()]
    payable = [float(v) for v in monthly["Payable"].abs().to_list()]
    net_cash = [float(v) for v in monthly["NetCashFlow"].to_list()]

    latest = monthly.iloc[-1]
    latest_revenue = float(latest["Revenue"])
    latest_expense = float(latest["Expense"])
    latest_profit = float(latest["Profit"])
    latest_margin = (latest_profit / latest_revenue * 100) if latest_revenue else 0.0
    latest_cash = float(latest["NetCashFlow"])

    expense_cat = (
        data[data["_class"] == "Expense"]
        .groupby("_category", as_index=False)["_amount"]
        .sum()
        .assign(_amount=lambda d: d["_amount"].abs())
        .sort_values(by="_amount", ascending=False)
        .head(8)
    )

    top_customers = (
        data.loc[data["_class"] == "Revenue", ["_customer", "_amount"]]
        .groupby("_customer", as_index=False)
        .agg({"_amount": "sum"})
        .sort_values(by="_amount", ascending=False)
        .head(8)
    )

    structure_recommendations, structure_profile = generate_structure_recommendations(raw_df, data, monthly, mapping)
    forecast = build_forecast(data)

    return {
        "months": months,
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "receivable": receivable,
        "payable": payable,
        "netCash": net_cash,
        "metrics": {
            "revenue": latest_revenue,
            "expense": latest_expense,
            "profit": latest_profit,
            "margin": latest_margin,
            "netCash": latest_cash,
        },
        "expenseBreakdown": {
            "labels": [str(v) for v in expense_cat["_category"].to_list()],
            "values": [float(v) for v in expense_cat["_amount"].to_list()],
        },
        "topCustomers": [
            {"name": str(r["_customer"]), "revenue": float(r["_amount"])} for _, r in top_customers.iterrows()
        ],
        "insights": generate_insights(monthly, data),
        "structureAdvice": structure_recommendations,
        "structureProfile": structure_profile,
        "forecast": forecast,
        "summary": {
            "rowsUploaded": int(len(raw_df)),
            "rowsUsed": int(len(data)),
            "startDate": data["_date"].min().strftime("%Y-%m-%d"),
            "endDate": data["_date"].max().strftime("%Y-%m-%d"),
            "detectedColumns": {
                "date": mapping.date,
                "amount": mapping.amount,
                "type": mapping.txn_type,
                "category": mapping.category,
                "customer": mapping.customer,
            },
        },
    }


@app.get("/")
def dashboard() -> str:
    return render_template("dashboard.html")


@app.post("/api/analyze")
def analyze_csv():
    if "file" not in request.files:
        return jsonify({"error": "Missing file field. Please upload a CSV/XLS/XLSX file."}), 400

    uploaded = request.files["file"]
    filename = uploaded.filename or ""

    if filename == "":
        return jsonify({"error": "No file selected."}), 400

    file_lower = filename.lower()
    allowed_extensions = (".csv", ".xls", ".xlsx")
    if not file_lower.endswith(allowed_extensions):
        return jsonify({"error": "Only .csv, .xls, and .xlsx files are supported."}), 400

    try:
        if file_lower.endswith(".csv"):
            raw_df = pd.read_csv(uploaded)
        else:
            raw_df = pd.read_excel(uploaded)
    except Exception:
        return jsonify({"error": "Could not read this file. Check that it is a valid, unprotected CSV or Excel export."}), 400

    if raw_df.empty or len(raw_df.columns) == 0:
        return jsonify({"error": "This file does not contain any transaction rows."}), 400

    try:
        mapping = detect_columns(raw_df)
        data = preprocess(raw_df, mapping)
    except ValueError as exc:
        return jsonify({"error": str(exc), "columnsFound": list(raw_df.columns)}), 400
    except Exception:
        app.logger.exception("Analysis failed while preparing uploaded file")
        return jsonify({"error": "We could not process this spreadsheet. Check that its columns contain valid dates and amounts."}), 422

    if data.empty:
        return jsonify({"error": "No valid rows found after cleaning date/amount values."}), 400

    try:
        monthly = monthly_metrics(data)
        payload = build_payload(raw_df, data, monthly, mapping)
        return jsonify(payload)
    except Exception:
        app.logger.exception("Analysis failed while building dashboard payload")
        return jsonify({"error": "We could not complete the analysis for this spreadsheet. Check the file structure and try again."}), 422


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "This file is larger than the 16 MB upload limit."}), 413


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    start_port = int(os.environ.get("PORT", "8501"))

    def _port_is_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.connect_ex(("127.0.0.1", port)) != 0

    port = start_port
    for _ in range(20):
        if _port_is_available(port):
            break
        port += 1

    if port != start_port:
        print(f"Port {start_port} is in use. Starting on port {port} instead.")

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

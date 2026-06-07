from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


DEBUG_MODE = False
GOOGLE_SHEETS_HEADERS = {"User-Agent": "Mozilla/5.0"}
REPORTING_DATA_SOURCE_VERSION = "reporting-sources-with-offline-v2"

CSV_SOURCES = {
    "online": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRTopkhW4vXgkThdRYpoK1bMEkmBbN_Tx3mKtwT4xFL5wDiNUDFqdSFYBTMh5A4KoVnjQaOPVbRGV4f/pub?gid=324204648&single=true&output=csv",
    "product_analysis": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRTopkhW4vXgkThdRYpoK1bMEkmBbN_Tx3mKtwT4xFL5wDiNUDFqdSFYBTMh5A4KoVnjQaOPVbRGV4f/pub?gid=662528227&single=true&output=csv",
    "payouts": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRTopkhW4vXgkThdRYpoK1bMEkmBbN_Tx3mKtwT4xFL5wDiNUDFqdSFYBTMh5A4KoVnjQaOPVbRGV4f/pub?gid=1523304948&single=true&output=csv",
    "offline": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRTopkhW4vXgkThdRYpoK1bMEkmBbN_Tx3mKtwT4xFL5wDiNUDFqdSFYBTMh5A4KoVnjQaOPVbRGV4f/pub?gid=112717234&single=true&output=csv",
}

REPORTING_TABLES = (
    "online",
    "product_analysis",
    "payouts",
    "offline",
)

MONEY_COLUMN_KEYWORDS = (
    "amount",
    "sale",
    "sales",
    "revenue",
    "gross",
    "net",
    "payout",
    "commission",
    "tax",
    "fee",
    "charge",
    "discount",
    "deduction",
    "refund",
    "total",
    "value",
    "paid",
)

NON_MONEY_COLUMN_KEYWORDS = (
    "date",
    "time",
    "id",
    "order",
    "orders",
    "quantity",
    "qty",
    "count",
    "items",
    "units",
)

PRODUCT_QUANTITY_KEYWORDS = (
    "qty",
    "quantity",
    "count",
    "orders",
    "order_count",
    "items",
    "units",
    "sold",
    "online",
    "offline",
    "total",
)

PRODUCT_TEXT_KEYWORDS = (
    "name",
    "category",
    "sku",
)

PRODUCT_MONEY_COLUMN_KEYWORDS = (
    "amount",
    "sale",
    "sales",
    "revenue",
    "gross",
    "net",
    "value",
)


def empty_reporting_data() -> Dict[str, pd.DataFrame]:
    return {table_name: pd.DataFrame() for table_name in REPORTING_TABLES}


def clean_column_name(column_name: object) -> str:
    cleaned = str(column_name).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [clean_column_name(column) for column in frame.columns]
    return frame


def parse_money_value(value: object) -> float | None:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = (
        text.replace("€", "")
        .replace("\u00a0", "")
        .replace(" ", "")
        .replace("−", "-")
    )
    text = re.sub(r"[^0-9,.\-]", "", text)

    if not text or text in {"-", ".", ",", "-.", "-,"}:
        return None

    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        whole, _, decimals = text.rpartition(",")
        if len(decimals) in (1, 2):
            text = f"{whole}.{decimals}"
        else:
            text = text.replace(",", "")
    elif text.count(".") > 1:
        whole, _, decimals = text.rpartition(".")
        text = f"{whole.replace('.', '')}.{decimals}"

    try:
        return float(text)
    except ValueError:
        return None


def is_money_column(column_name: str, series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(25)
    if sample.str.contains("€", regex=False).any():
        return True

    if any(keyword in column_name for keyword in NON_MONEY_COLUMN_KEYWORDS):
        return False

    if any(keyword in column_name for keyword in MONEY_COLUMN_KEYWORDS):
        return True

    return False


def clean_money_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if is_money_column(column, frame[column]):
            frame[column] = frame[column].map(parse_money_value).astype("float64")
    return frame


def is_product_text_column(column_name: str) -> bool:
    if column_name in {"product", "item"}:
        return True

    if column_name.endswith("_name"):
        return True

    return any(keyword in column_name for keyword in PRODUCT_TEXT_KEYWORDS)


def clean_product_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if is_product_text_column(column):
            frame[column] = frame[column].astype("string")
        elif any(keyword in column for keyword in PRODUCT_QUANTITY_KEYWORDS):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        elif any(keyword in column for keyword in PRODUCT_MONEY_COLUMN_KEYWORDS):
            frame[column] = frame[column].map(parse_money_value).astype("float64")
    return frame


def clean_reporting_frame(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    frame = clean_column_names(frame)

    if table_name in {"online", "payouts", "offline"}:
        return clean_money_columns(frame)

    if table_name == "product_analysis":
        return clean_product_analysis(frame)

    return frame


def response_looks_like_html(text: str) -> bool:
    stripped = text.lstrip().lower()
    return stripped.startswith("<!doctype html") or stripped.startswith("<html")


def debug_test_frame(name: str) -> pd.DataFrame:
    if name == "product_analysis":
        return pd.DataFrame(
            {
                "Product": ["Debug Product A", "Debug Product B"],
                "Quantity": [1, 2],
            }
        )

    if name == "payouts":
        return pd.DataFrame(
            {
                "Partner": ["Debug Partner A", "Debug Partner B"],
                "Gross Sale": ["€10.00", "€20.00"],
            }
        )

    return pd.DataFrame(
        {
            "Partner": ["Debug Partner A", "Debug Partner B"],
            "Order ID": ["debug-1", "debug-2"],
            "Subtotal": ["€10.00", "€20.00"],
        }
    )


def load_csv_safely(name: str, url: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    debug: dict[str, Any] = {
        "sheet": name,
        "url": url,
        "status_code": None,
        "response_text_length": 0,
        "response_preview": "",
        "raw_shape": None,
        "cleaned_columns": [],
        "warning": "",
        "error": "",
        "used_debug_fallback": False,
    }

    try:
        response = requests.get(url, headers=GOOGLE_SHEETS_HEADERS, timeout=20)
        csv_text = response.content.decode("utf-8-sig", errors="replace")
        debug["status_code"] = response.status_code
        debug["response_text_length"] = len(csv_text)
        debug["response_preview"] = csv_text[:300]
        response.raise_for_status()

        if not csv_text.strip():
            debug["warning"] = "CSV response is empty."
            st.warning(f"{name}: CSV response is empty.")
            return fallback_or_empty(name, debug)

        if response_looks_like_html(csv_text):
            debug["warning"] = (
                "Google Sheets returned HTML, not CSV. Check publish settings or URL."
            )
            st.warning(f"{name}: {debug['warning']}")
            return fallback_or_empty(name, debug)

        try:
            frame = pd.read_csv(StringIO(csv_text))
        except Exception as exc:
            debug["error"] = f"pd.read_csv failed: {exc}"
            st.warning(f"{name}: pd.read_csv failed: {exc}")
            return fallback_or_empty(name, debug)

        debug["raw_shape"] = frame.shape
        return frame, debug
    except Exception as exc:
        debug["error"] = str(exc)
        st.warning(f"Could not load {name}: {exc}")
        return fallback_or_empty(name, debug)


def fallback_or_empty(name: str, debug: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if DEBUG_MODE:
        debug["used_debug_fallback"] = True
        frame = debug_test_frame(name)
        debug["raw_shape"] = frame.shape
        st.warning(f"{name}: using DEBUG_MODE fallback data.")
        return frame, debug

    return pd.DataFrame(), debug


def load_and_clean_sheet(table_name: str, url: str) -> pd.DataFrame:
    try:
        response = requests.get(url, headers=GOOGLE_SHEETS_HEADERS, timeout=20)
        csv_text = response.content.decode("utf-8-sig", errors="replace")
        response.raise_for_status()

        if not csv_text.strip():
            st.warning(f"{table_name}: CSV response is empty.")
            return pd.DataFrame()

        if response_looks_like_html(csv_text):
            st.warning(
                f"{table_name}: Google Sheets returned HTML, not CSV. "
                "Check publish settings or URL."
            )
            return pd.DataFrame()

        frame = pd.read_csv(StringIO(csv_text))
        if frame.empty:
            return pd.DataFrame()

        return clean_reporting_frame(table_name, frame)
    except Exception as exc:
        st.warning(f"Could not load {table_name}: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_reporting_data(
    source_version: str = REPORTING_DATA_SOURCE_VERSION,
) -> Dict[str, pd.DataFrame]:
    _ = source_version
    data = empty_reporting_data()

    with ThreadPoolExecutor(max_workers=len(CSV_SOURCES)) as executor:
        future_to_table = {
            executor.submit(load_and_clean_sheet, table_name, url): table_name
            for table_name, url in CSV_SOURCES.items()
        }

        for future in as_completed(future_to_table):
            table_name = future_to_table[future]
            try:
                data[table_name] = future.result()
            except Exception as exc:
                st.warning(f"Could not load {table_name}: {exc}")
                data[table_name] = pd.DataFrame()

    return data


@st.cache_data(show_spinner=False)
def load_reporting_data_with_debug(
    source_version: str = REPORTING_DATA_SOURCE_VERSION,
) -> tuple[Dict[str, pd.DataFrame], list[dict[str, Any]]]:
    _ = source_version
    data = empty_reporting_data()
    debug_details: list[dict[str, Any]] = []

    for table_name, url in CSV_SOURCES.items():
        frame, debug = load_csv_safely(table_name, url)

        if frame.empty:
            data[table_name] = pd.DataFrame()
            debug["cleaned_columns"] = []
            debug_details.append(debug)
            continue

        try:
            data[table_name] = clean_reporting_frame(table_name, frame)
            debug["cleaned_columns"] = list(data[table_name].columns)
        except Exception:
            debug["error"] = f"Could not clean {table_name} data."
            data[table_name] = pd.DataFrame()
            debug["cleaned_columns"] = []

        debug_details.append(debug)

    return data, debug_details


def clear_reporting_data_cache() -> None:
    load_reporting_data.clear()
    load_reporting_data_with_debug.clear()

from html import escape
import math
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import (
    REPORTING_DATA_SOURCE_VERSION,
    clear_reporting_data_cache,
    load_reporting_data,
)


DASHBOARD_TITLE = "IBC - Delivery Partner Analysis"
PAGES = ["Overall Analysis", "Online Analysis"]
VALID_PARTNERS = ["Wolt", "Uber Eats", "Lieferando"]
VALID_PARTNER_SET = set(VALID_PARTNERS)
FILTER_DATE_COLUMN = "_filter_date"
FILTER_YEAR_COLUMN = "_filter_year"
FILTER_MONTH_COLUMN = "_filter_month"
FILTER_MONTH_NUMBER_COLUMN = "_filter_month_number"
ONLINE_CHART_TITLE_FONT_FAMILY = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
OVERALL_BUSINESS_SOURCE_VERSION = "payouts-gross-offline-total-v2"


def empty_data_status() -> dict:
    return {
        "online": [],
        "product_analysis": [],
        "payouts": [],
        "offline": [],
    }


def row_count(data: dict, table_name: str) -> int:
    return len(data.get(table_name, []))


def render_data_status(data: dict) -> None:
    st.caption("Data status")
    st.write(f"online rows count: {row_count(data, 'online')}")
    st.write(f"product_analysis rows count: {row_count(data, 'product_analysis')}")
    st.write(f"payouts rows count: {row_count(data, 'payouts')}")
    st.write(f"offline rows count: {row_count(data, 'offline')}")


def format_euro(value: float) -> str:
    return f"€{value:,.2f}"


def format_whole_number(value: int) -> str:
    return f"{value:,}"


def find_first_column(frame: pd.DataFrame, column_names: list[str]) -> str | None:
    for column_name in column_names:
        if column_name in frame.columns:
            return column_name
    return None


def parse_date_series(series: pd.Series) -> pd.Series:
    series_text = series.astype("string").str.strip()
    iso_like = series_text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").fillna(False)

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso_like.any():
        parsed.loc[iso_like] = pd.to_datetime(
            series.loc[iso_like],
            errors="coerce",
            dayfirst=False,
        )

    non_iso = ~iso_like
    if non_iso.any():
        parsed.loc[non_iso] = pd.to_datetime(
            series.loc[non_iso],
            errors="coerce",
            dayfirst=True,
        )

    unresolved = parsed.isna() & series.notna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            series.loc[unresolved],
            errors="coerce",
            dayfirst=False,
        )
    return parsed


def prepare_dynamic_reporting_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    prepared = frame.copy()
    date_column = find_first_column(prepared, ["order_date", "date"])
    year_column = find_first_column(prepared, ["fy", "year"])
    month_column = find_first_column(prepared, ["month"])

    parsed_dates = pd.Series(pd.NaT, index=prepared.index, dtype="datetime64[ns]")
    if date_column:
        parsed_dates = parse_date_series(prepared[date_column])

    prepared[FILTER_DATE_COLUMN] = parsed_dates
    prepared[FILTER_YEAR_COLUMN] = parsed_dates.dt.year.astype("Int64").astype("string")
    prepared[FILTER_MONTH_COLUMN] = parsed_dates.dt.month_name()
    prepared[FILTER_MONTH_NUMBER_COLUMN] = parsed_dates.dt.month.astype("Int64")

    if year_column and prepared[FILTER_YEAR_COLUMN].isna().all():
        prepared[FILTER_YEAR_COLUMN] = prepared[year_column].astype(str).str.strip()

    if month_column:
        missing_month = prepared[FILTER_MONTH_COLUMN].isna()
        if missing_month.any():
            month_text = prepared.loc[missing_month, month_column].astype(str).str.strip()
            prepared.loc[missing_month, FILTER_MONTH_COLUMN] = month_text
            month_dates = pd.to_datetime(month_text, format="%B", errors="coerce")
            unresolved = month_dates.isna()
            if unresolved.any():
                month_dates.loc[unresolved] = pd.to_datetime(
                    month_text.loc[unresolved],
                    format="%b",
                    errors="coerce",
                )
            prepared.loc[missing_month, FILTER_MONTH_NUMBER_COLUMN] = (
                month_dates.dt.month.astype("Int64")
            )

    for column in (FILTER_YEAR_COLUMN, FILTER_MONTH_COLUMN):
        prepared[column] = prepared[column].astype("string")
        invalid_values = prepared[column].str.lower().isin(
            ["", "nan", "nat", "none", "null"]
        ).fillna(False)
        prepared.loc[invalid_values, column] = pd.NA

    return prepared


def prepare_dynamic_online_data(online_df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_dynamic_reporting_frame(online_df)
    partner_column = find_first_column(prepared, ["partner", "patner"])
    if not prepared.empty and partner_column:
        prepared["_partner_display"] = prepared[partner_column].map(valid_partner_display_name)
        prepared = prepared[prepared["_partner_display"].notna()]
    return prepared


def prepare_dynamic_product_data(product_analysis_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_dynamic_reporting_frame(product_analysis_df)


def prepare_overall_payout_frame(payouts_df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_dynamic_reporting_frame(payouts_df)
    if prepared.empty:
        return prepared

    month_start = payout_month_start_series(prepared, "All Years")
    prepared[FILTER_DATE_COLUMN] = month_start
    prepared[FILTER_YEAR_COLUMN] = month_start.dt.year.astype("Int64").astype("string")
    prepared[FILTER_MONTH_COLUMN] = month_start.dt.month_name()
    prepared[FILTER_MONTH_NUMBER_COLUMN] = month_start.dt.month.astype("Int64")

    for column in (FILTER_YEAR_COLUMN, FILTER_MONTH_COLUMN):
        prepared[column] = prepared[column].astype("string")
        invalid_values = prepared[column].str.lower().isin(
            ["", "nan", "nat", "none", "null"]
        ).fillna(False)
        prepared.loc[invalid_values, column] = pd.NA

    return prepared


def infer_single_reporting_year(frame: pd.DataFrame) -> str | None:
    if frame.empty or FILTER_YEAR_COLUMN not in frame.columns:
        return None

    years = (
        frame[FILTER_YEAR_COLUMN]
        .dropna()
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .dropna()
        .unique()
        .tolist()
    )
    return years[0] if len(years) == 1 else None


def prepare_overall_offline_frame(
    offline_df: pd.DataFrame,
    fallback_year: str | None,
) -> pd.DataFrame:
    prepared = prepare_dynamic_reporting_frame(offline_df)
    if prepared.empty:
        return prepared

    if not fallback_year or not re.fullmatch(r"\d{4}", str(fallback_year)):
        return prepared

    month_column = usable_column(prepared, FILTER_MONTH_COLUMN) or find_first_column(
        prepared,
        ["month"],
    )
    if not month_column:
        return prepared

    month_numbers = prepared[FILTER_MONTH_NUMBER_COLUMN].copy()
    unresolved_months = month_numbers.isna()
    if unresolved_months.any():
        month_numbers.loc[unresolved_months] = prepared.loc[
            unresolved_months,
            month_column,
        ].map(month_number_from_text)

    valid_months = month_numbers.notna()
    if not valid_months.any():
        return prepared

    month_start = pd.to_datetime(
        {
            "year": int(str(fallback_year)),
            "month": month_numbers.loc[valid_months].astype(int),
            "day": 1,
        },
        errors="coerce",
    )
    prepared.loc[valid_months, FILTER_DATE_COLUMN] = month_start
    prepared.loc[valid_months, FILTER_YEAR_COLUMN] = str(fallback_year)
    prepared.loc[valid_months, FILTER_MONTH_COLUMN] = month_start.dt.month_name().values
    prepared.loc[valid_months, FILTER_MONTH_NUMBER_COLUMN] = month_start.dt.month.astype(
        "Int64"
    ).values

    return prepared


def filter_options(frame: pd.DataFrame, column_name: str, all_label: str) -> list[str]:
    if frame.empty or column_name not in frame.columns:
        return [all_label]

    values = frame[column_name].dropna().astype(str).str.strip()
    values = [value for value in values.unique() if value and value.lower() not in {"nan", "nat"}]
    return [all_label, *sorted(values)]


def year_filter_options(frame: pd.DataFrame) -> list[str]:
    if frame.empty or FILTER_YEAR_COLUMN not in frame.columns:
        return ["All Years"]

    years = frame[FILTER_YEAR_COLUMN].dropna().astype(str).str.strip()
    year_values = sorted(
        {year for year in years if year and year.lower() not in {"nan", "nat"}},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    return ["All Years", *year_values]


def month_filter_options(frame: pd.DataFrame) -> list[str]:
    if frame.empty or FILTER_MONTH_COLUMN not in frame.columns:
        return ["All Months"]

    month_frame = frame[[FILTER_MONTH_COLUMN, FILTER_MONTH_NUMBER_COLUMN]].dropna(
        subset=[FILTER_MONTH_COLUMN]
    )
    if month_frame.empty:
        return ["All Months"]

    month_frame[FILTER_MONTH_COLUMN] = month_frame[FILTER_MONTH_COLUMN].astype(str).str.strip()
    month_frame = month_frame[
        month_frame[FILTER_MONTH_COLUMN].ne("")
        & ~month_frame[FILTER_MONTH_COLUMN].str.lower().isin(["nan", "nat"])
    ]
    if month_frame.empty:
        return ["All Months"]

    month_order = (
        month_frame.groupby(FILTER_MONTH_COLUMN, dropna=True)[FILTER_MONTH_NUMBER_COLUMN]
        .min()
        .sort_values()
    )
    return ["All Months", *month_order.index.tolist()]


def ensure_selectbox_state(key: str, options: list[str]) -> None:
    if not options:
        return

    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def usable_column(frame: pd.DataFrame, column_name: str | None) -> str | None:
    if not column_name or column_name not in frame.columns:
        return None
    return column_name if frame[column_name].notna().any() else None


def calculate_online_kpis(online_df: pd.DataFrame) -> dict:
    if online_df.empty:
        return {
            "gross_sales": 0.0,
            "orders": 0,
            "average_order_value": 0.0,
            "best_partner": "N/A",
        }

    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    partner_column = find_first_column(online_df, ["partner", "patner"])

    sales = (
        pd.to_numeric(online_df[sales_column], errors="coerce").fillna(0)
        if sales_column
        else pd.Series(dtype="float64")
    )
    gross_sales = float(sales.sum()) if not sales.empty else 0.0

    orders = (
        int(online_df[order_column].dropna().nunique())
        if order_column
        else int(len(online_df.index))
    )
    average_order_value = gross_sales / orders if orders else 0.0

    best_partner = "N/A"
    if partner_column and sales_column:
        partner_sales = (
            online_df.assign(_online_sales=sales)
            .groupby(partner_column, dropna=True)["_online_sales"]
            .sum()
            .sort_values(ascending=False)
        )
        if not partner_sales.empty:
            best_partner = str(partner_sales.index[0])

    return {
        "gross_sales": gross_sales,
        "orders": orders,
        "average_order_value": average_order_value,
        "best_partner": best_partner,
    }


def partner_display_name(partner_name: object) -> str:
    if pd.isna(partner_name):
        return ""

    text = str(partner_name).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    known_names = {
        "wolt": "Wolt",
        "uber eats": "Uber Eats",
        "ubereats": "Uber Eats",
        "lefrando": "Lieferando",
        "lieferando": "Lieferando",
        "lieferando.de": "Lieferando",
    }
    return known_names.get(text.lower(), text.title())


def valid_partner_display_name(partner_name: object) -> str | None:
    display_name = partner_display_name(partner_name)
    if display_name in VALID_PARTNER_SET:
        return display_name
    return None


def partner_accent_color(partner_name: str) -> str:
    name = partner_name.lower()
    if "wolt" in name:
        return "#4A9EFF"
    if "uber" in name:
        return "#2ECC71"
    if "lieferando" in name or "lefrando" in name:
        return "#FF6B35"
    return "#7A7A7A"


def partner_avatar_style(partner_name: str) -> tuple[str, str, str]:
    name = partner_display_name(partner_name)
    if name == "Wolt":
        return "WL", "#d7eaff", "#2f6fae"
    if name == "Uber Eats":
        return "UE", "#f5e1d0", "#965832"
    if name == "Lieferando":
        return "LF", "#ffd9c2", "#c85825"

    initials = (
        "".join(part[0] for part in name.split()[:2]).upper()
        if len(name.split()) > 1
        else name[:2].upper()
    ) or "NA"
    return initials[:2], "#e4e0d9", "#5f6368"


def select_options(frame: pd.DataFrame, column_name: str | None, all_label: str) -> list:
    if not column_name or column_name not in frame.columns:
        return [all_label]

    values = frame[column_name].dropna().astype(str).str.strip()
    values = sorted(value for value in values.unique() if value)
    return [all_label, *values]


def apply_online_filters(
    online_df: pd.DataFrame,
    partner_column: str | None,
    year_column: str | None,
    month_column: str | None,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    filtered = online_df.copy()
    filter_partner_column = "_partner_display" if "_partner_display" in filtered.columns else partner_column
    filter_year_column = usable_column(filtered, FILTER_YEAR_COLUMN) or year_column
    filter_month_column = usable_column(filtered, FILTER_MONTH_COLUMN) or month_column

    if filter_partner_column and selected_partner != "All Partners":
        selected_display = partner_display_name(selected_partner)
        filtered = filtered[
            filtered[filter_partner_column].map(partner_display_name) == selected_display
        ]
    if filter_year_column and selected_year != "All Years":
        filtered = filtered[
            filtered[filter_year_column].map(
                lambda value: year_matches_value(value, selected_year)
            )
        ]
    if filter_month_column and selected_month != "All Months":
        filtered = filtered[
            filtered[filter_month_column].map(
                lambda value: month_matches_period(value, selected_month)
            )
        ]

    return filtered


def month_matches_period(period_value: object, selected_month: str) -> bool:
    if not selected_month or selected_month == "All Months":
        return True

    period_text = str(period_value).lower()
    month_text = str(selected_month).strip().lower()
    if not month_text:
        return True

    month_abbrev = month_text[:3]
    return month_text in period_text or month_abbrev in period_text


def year_matches_value(year_value: object, selected_year: str) -> bool:
    if not selected_year or selected_year == "All Years":
        return True

    year_text = str(year_value).strip()
    return year_text == selected_year or selected_year in year_text


def fiscal_quarter_for_month(month_number: object) -> str | None:
    value = pd.to_numeric(pd.Series([month_number]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    month = int(value)
    if month in (4, 5, 6):
        return "Q1"
    if month in (7, 8, 9):
        return "Q2"
    if month in (10, 11, 12):
        return "Q3"
    if month in (1, 2, 3):
        return "Q4"
    return None


def build_overall_month_options(online_df: pd.DataFrame, offline_df: pd.DataFrame, channel: str) -> list[str]:
    if channel == "Online":
        source = online_df
    elif channel == "Offline":
        source = offline_df
    else:
        frames = [frame for frame in (online_df, offline_df) if not frame.empty]
        source = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return month_filter_options(source)


def apply_overall_time_filters(
    frame: pd.DataFrame,
    selected_year: str,
    selected_quarter: str,
    selected_month: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = frame.copy()
    year_column = usable_column(filtered, FILTER_YEAR_COLUMN) or find_first_column(filtered, ["fy", "year"])
    month_column = usable_column(filtered, FILTER_MONTH_COLUMN) or find_first_column(filtered, ["month"])
    month_number_column = usable_column(filtered, FILTER_MONTH_NUMBER_COLUMN)

    if year_column and selected_year != "All Years":
        filtered = filtered[
            filtered[year_column].map(lambda value: year_matches_value(value, selected_year))
        ]

    if month_column and selected_month != "All Months":
        filtered = filtered[
            filtered[month_column].map(lambda value: month_matches_period(value, selected_month))
        ]

    if selected_quarter != "All Quarters":
        if month_number_column:
            quarter_series = filtered[month_number_column].map(fiscal_quarter_for_month)
        elif month_column:
            month_numbers = pd.to_datetime(filtered[month_column].astype(str), format="%B", errors="coerce")
            unresolved = month_numbers.isna()
            if unresolved.any():
                month_numbers.loc[unresolved] = pd.to_datetime(
                    filtered.loc[unresolved, month_column].astype(str),
                    format="%b",
                    errors="coerce",
                )
            quarter_series = month_numbers.dt.month.map(fiscal_quarter_for_month)
        else:
            quarter_series = pd.Series(index=filtered.index, dtype="object")
        filtered = filtered[quarter_series == selected_quarter]

    return filtered


def apply_payout_filters(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    filtered = prepare_dynamic_reporting_frame(payouts_df)

    partner_column = find_first_column(filtered, ["partner", "patner"])
    year_column = usable_column(filtered, FILTER_YEAR_COLUMN) or find_first_column(filtered, ["fy", "year"])
    month_column = usable_column(filtered, FILTER_MONTH_COLUMN) or find_first_column(filtered, ["month"])
    invoice_period_column = find_first_column(
        filtered,
        ["invoice_period", "period", "payout_period"],
    )

    if partner_column and selected_partner != "All Partners":
        selected_display = partner_display_name(selected_partner)
        filtered = filtered[
            filtered[partner_column].map(partner_display_name) == selected_display
        ]

    if year_column and selected_year != "All Years":
        filtered = filtered[
            filtered[year_column].map(lambda value: year_matches_value(value, selected_year))
        ]

    if selected_month != "All Months":
        if month_column:
            filtered = filtered[
                filtered[month_column].map(
                    lambda value: month_matches_period(value, selected_month)
                )
            ]
        elif invoice_period_column:
            filtered = filtered[
                filtered[invoice_period_column].map(
                    lambda value: month_matches_period(value, selected_month)
                )
            ]

    return filtered


def apply_partner_year_month_filters(
    frame: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    filtered = prepare_dynamic_reporting_frame(frame)
    partner_column = find_first_column(filtered, ["partner", "patner"])
    year_column = usable_column(filtered, FILTER_YEAR_COLUMN) or find_first_column(filtered, ["fy", "year"])
    month_column = usable_column(filtered, FILTER_MONTH_COLUMN) or find_first_column(filtered, ["month"])
    period_column = find_first_column(filtered, ["invoice_period", "period", "payout_period"])

    if partner_column and selected_partner != "All Partners":
        selected_display = partner_display_name(selected_partner)
        filtered = filtered[
            filtered[partner_column].map(partner_display_name) == selected_display
        ]

    if year_column and selected_year != "All Years":
        filtered = filtered[
            filtered[year_column].map(lambda value: year_matches_value(value, selected_year))
        ]

    if selected_month != "All Months":
        if month_column:
            filtered = filtered[
                filtered[month_column].map(
                    lambda value: month_matches_period(value, selected_month)
                )
            ]
        elif period_column:
            filtered = filtered[
                filtered[period_column].map(
                    lambda value: month_matches_period(value, selected_month)
                )
            ]

    return filtered


def calculate_net_sale(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> float:
    if payouts_df.empty:
        return 0.0

    net_sale_column = find_first_column(
        payouts_df,
        [
            "net_sale_payout",
            "net_sale",
            "net_payout",
            "net_revenue",
            "net_revenue_after_delivery_partner_deductions",
            "payout",
        ],
    )
    if not net_sale_column:
        return 0.0

    filtered_payouts = apply_payout_filters(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    net_sale = pd.to_numeric(filtered_payouts[net_sale_column], errors="coerce").fillna(0)
    return float(net_sale.sum())


def calculate_payout_gross_sales(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> float:
    if payouts_df.empty:
        return 0.0

    gross_sale_column = find_first_column(
        payouts_df,
        [
            "gross_sale",
            "gross_sales",
            "gross_revenue",
            "gross_revenue_before_partner_deductions",
            "gross",
            "sale",
            "sales",
        ],
    )
    if not gross_sale_column:
        return 0.0

    filtered_payouts = apply_payout_filters(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    gross_sale = pd.to_numeric(filtered_payouts[gross_sale_column], errors="coerce").fillna(0)
    return float(gross_sale.sum())


def calculate_deduction_percentage(total_gross_sales: float, net_sale: float) -> float:
    if total_gross_sales <= 0:
        return 0.0

    return ((total_gross_sales - net_sale) / total_gross_sales) * 100.0


def calculate_total_deductions(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
    total_gross_sales: float,
    net_sale: float,
) -> float:
    if not payouts_df.empty:
        filtered_payouts = apply_payout_filters(
            payouts_df,
            selected_partner,
            selected_year,
            selected_month,
        )
        deduction_column = find_first_column(
            filtered_payouts,
            [
                "total_deductions",
                "total_deduction",
                "deductions",
                "deduction",
                "partner_deduction",
                "delivery_partner_deduction",
                "commission",
                "commission_amount",
                "fees",
                "fee",
                "charges",
                "charge",
                "platform_fee",
                "service_fee",
            ],
        )
        if deduction_column:
            return float(
                pd.to_numeric(filtered_payouts[deduction_column], errors="coerce")
                .fillna(0)
                .sum()
            )

    return max(0.0, total_gross_sales - net_sale)


def calculate_partner_payout_metrics(
    payouts_df: pd.DataFrame,
    partner_name: str,
    selected_year: str,
    selected_month: str,
) -> tuple[float, float]:
    if payouts_df.empty:
        return 0.0, 0.0

    filtered_payouts = apply_payout_filters(
        payouts_df,
        partner_name,
        selected_year,
        selected_month,
    )
    if filtered_payouts.empty:
        return 0.0, 0.0

    net_payout_column = find_first_column(
        filtered_payouts,
        [
            "net_sale_payout",
            "net_sale",
            "net_payout",
            "net_revenue",
            "net_revenue_after_delivery_partner_deductions",
            "payout",
        ],
    )
    gross_sale_column = find_first_column(
        filtered_payouts,
        [
            "gross_sale",
            "gross_sales",
            "gross_revenue",
            "gross_revenue_before_partner_deductions",
            "gross",
            "sale",
            "sales",
        ],
    )
    deduction_column = find_first_column(
        filtered_payouts,
        [
            "total_deductions",
            "total_deduction",
            "deductions",
            "deduction",
            "partner_deduction",
            "delivery_partner_deduction",
            "commission",
            "commission_amount",
            "fees",
            "fee",
            "charges",
            "charge",
            "platform_fee",
            "service_fee",
        ],
    )

    net_payout = (
        float(
            pd.to_numeric(filtered_payouts[net_payout_column], errors="coerce")
            .fillna(0)
            .sum()
        )
        if net_payout_column
        else 0.0
    )

    if deduction_column:
        total_deductions = float(
            pd.to_numeric(filtered_payouts[deduction_column], errors="coerce")
            .fillna(0)
            .sum()
        )
    elif gross_sale_column and net_payout_column:
        gross_sales = float(
            pd.to_numeric(filtered_payouts[gross_sale_column], errors="coerce")
            .fillna(0)
            .sum()
        )
        total_deductions = gross_sales - net_payout
    else:
        total_deductions = 0.0

    return total_deductions, net_payout


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def format_partner_date_range(partner_df: pd.DataFrame, date_column: str | None) -> str:
    if not date_column or date_column not in partner_df.columns:
        return "Selected period"

    dates = pd.to_datetime(partner_df[date_column], errors="coerce", dayfirst=True)
    dates = dates.dropna()
    if dates.empty:
        return "Selected period"

    start_month = dates.min().to_period("M").to_timestamp()
    end_month = dates.max().to_period("M").to_timestamp()

    if start_month == end_month:
        return start_month.strftime("%b %Y")

    return f"{start_month.strftime('%b %Y')} to {end_month.strftime('%b %Y')}"


def online_chart_title(
    title_text: str,
    subtitle_text: str | None = None,
    subtitle_color: str = "#CFC7BD",
) -> dict:
    title = (
        f"<b>{escape(title_text)}</b>"
        if title_text
        else ""
    )
    if subtitle_text:
        title += (
            f"<br><span style='font-size:16px;color:{subtitle_color}'>"
            + escape(subtitle_text)
            + "</span>"
        )
    return {
        "text": title,
        "font": {
            "family": ONLINE_CHART_TITLE_FONT_FAMILY,
            "size": 22,
            "color": "#FFFFFF",
        },
        "x": 0.0,
        "xanchor": "left",
        "y": 0.95,
        "yanchor": "top",
    }


def render_summary_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="online-summary-card">
            <div class="online-card-label">{escape(label)}</div>
            <div class="online-card-value">{escape(value)}</div>
            <div class="online-card-subtext">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card_with_secondary(
    label: str,
    value: str,
    secondary_value: str,
    subtext: str,
) -> None:
    st.markdown(
        f"""
        <div class="online-summary-card">
            <div class="online-card-label">{escape(label)}</div>
            <div class="online-card-value">{escape(value)}</div>
            <div class="online-card-secondary-value">{escape(secondary_value)}</div>
            <div class="online-card-subtext">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_split_channel_row(
    channel_label: str,
    value: float | int | None,
    percent: float | None,
    value_type: str,
    color_class: str,
) -> str:
    safe_label = escape(channel_label)
    safe_color = escape(color_class)
    safe_percent = escape(format_split_percent(percent))
    safe_value = escape(format_split_value(value, value_type))
    width = split_progress_width(percent)
    return (
        '<div class="overall-split-row">'
        '<div class="overall-split-row-main">'
        '<div class="overall-split-row-header">'
        f"<span>{safe_label}</span>"
        f'<strong class="{safe_color}">{safe_percent}</strong>'
        "</div>"
        '<div class="overall-split-track">'
        f'<div class="overall-split-fill {safe_color}" style="width: {width:.1f}%;"></div>'
        "</div>"
        "</div>"
        f'<div class="overall-split-value-box {safe_color}">'
        f"<span>{safe_label}</span>"
        f"<strong>{safe_value}</strong>"
        "</div>"
        "</div>"
    )


def render_overall_split_card(title: str, split_metrics: dict[str, object]) -> None:
    value_type = str(split_metrics["value_type"])
    offline_row = render_split_channel_row(
        "Offline",
        split_metrics["offline_value"],
        split_metrics["offline_percent"],
        value_type,
        "offline",
    )
    online_row = render_split_channel_row(
        "Online",
        split_metrics["online_value"],
        split_metrics["online_percent"],
        value_type,
        "online",
    )
    card_html = (
        '<div class="overall-split-card">'
        f'<div class="overall-split-title">{escape(title)}</div>'
        '<div class="overall-split-content">'
        f"{offline_row}{online_row}"
        "</div>"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_overall_info_card(title: str, message: str, class_name: str = "") -> None:
    extra_class = f" {escape(class_name)}" if class_name else ""
    st.markdown(
        f"""
        <div class="overall-info-card{extra_class}">
            <div class="overall-info-title">{escape(title)}</div>
            <div class="overall-info-message">{escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_hour_bucket(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    compact = text.replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::\d{2})?[-–](\d{1,2})(?::\d{2})?", compact)
    if match:
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        return f"{start_hour:02d}:00–{end_hour:02d}:00"

    parsed = pd.to_datetime(pd.Series([text]), format="%H:%M", errors="coerce").iloc[0]
    if pd.isna(parsed):
        parsed = pd.to_datetime(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return text

    start_hour = int(parsed.hour)
    end_hour = (start_hour + 1) % 24
    return f"{start_hour:02d}:00–{end_hour:02d}:00"


def build_grain_kpi_frame(online_df: pd.DataFrame, grain: str) -> pd.DataFrame:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    if online_df.empty or not sales_column:
        return pd.DataFrame(columns=["label", "orders", "sales"])

    grain_frame = online_df.copy()
    grain_frame["_grain_sales"] = pd.to_numeric(
        grain_frame[sales_column],
        errors="coerce",
    ).fillna(0)

    if grain == "day":
        weekday_column = find_first_column(grain_frame, ["weekday", "day", "day_name"])
        date_column = usable_column(grain_frame, FILTER_DATE_COLUMN) or find_first_column(
            grain_frame,
            ["order_date", "date"],
        )
        if weekday_column:
            grain_frame["_grain_label"] = grain_frame[weekday_column].astype("string").str.strip()
        elif date_column:
            grain_frame["_grain_label"] = parse_date_series(grain_frame[date_column]).dt.day_name()
        else:
            return pd.DataFrame(columns=["label", "orders", "sales"])
    else:
        hour_column = find_first_column(
            grain_frame,
            ["hour_slot", "hour_bucket", "hour", "time_slot"],
        )
        time_column = find_first_column(grain_frame, ["time", "order_time"])
        if hour_column:
            grain_frame["_grain_label"] = grain_frame[hour_column].map(normalize_hour_bucket)
        elif time_column:
            grain_frame["_grain_label"] = grain_frame[time_column].map(normalize_hour_bucket)
        else:
            return pd.DataFrame(columns=["label", "orders", "sales"])

    grain_frame["_grain_label"] = grain_frame["_grain_label"].astype("string").str.strip()
    grain_frame = grain_frame[
        grain_frame["_grain_label"].notna()
        & grain_frame["_grain_label"].ne("")
        & ~grain_frame["_grain_label"].str.lower().isin(["nan", "nat", "none", "null"])
    ]
    if grain_frame.empty:
        return pd.DataFrame(columns=["label", "orders", "sales"])

    if order_column:
        grouped = grain_frame.groupby("_grain_label", dropna=True).agg(
            orders=(order_column, "nunique"),
            sales=("_grain_sales", "sum"),
        )
    else:
        grouped = grain_frame.groupby("_grain_label", dropna=True).agg(
            orders=("_grain_sales", "size"),
            sales=("_grain_sales", "sum"),
        )

    return (
        grouped.reset_index()
        .rename(columns={"_grain_label": "label"})
        .sort_values(["orders", "sales"], ascending=[False, False])
        .reset_index(drop=True)
    )


def render_grain_kpi_card(title: str, value: str, orders: int, sales: float) -> None:
    st.markdown(
        f"""
        <div class="grain-kpi-card">
            <div class="online-card-label">{escape(title)}</div>
            <div class="grain-kpi-value">{escape(value)}</div>
            <div class="grain-kpi-detail">{escape(format_whole_number(orders))} Orders</div>
            <div class="grain-kpi-detail">{escape(format_euro(sales))} Sales</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_grain_analysis_kpis(online_df: pd.DataFrame) -> None:
    peak_day_frame = build_grain_kpi_frame(online_df, "day")
    peak_hour_frame = build_grain_kpi_frame(online_df, "hour")

    columns = st.columns(2, gap="medium")
    with columns[0]:
        if peak_day_frame.empty:
            render_grain_kpi_card("PEAK DAY", "N/A", 0, 0.0)
        else:
            peak_day = peak_day_frame.iloc[0]
            render_grain_kpi_card(
                "PEAK DAY",
                str(peak_day["label"]),
                int(peak_day["orders"]),
                float(peak_day["sales"]),
            )
    with columns[1]:
        if peak_hour_frame.empty:
            render_grain_kpi_card("PEAK HOUR", "N/A", 0, 0.0)
        else:
            peak_hour = peak_hour_frame.iloc[0]
            render_grain_kpi_card(
                "PEAK HOUR",
                str(peak_hour["label"]),
                int(peak_hour["orders"]),
                float(peak_hour["sales"]),
            )


def build_weekday_order_revenue_frame(online_df: pd.DataFrame) -> pd.DataFrame:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    weekday_column = find_first_column(online_df, ["weekday", "day", "day_name"])
    date_column = usable_column(online_df, FILTER_DATE_COLUMN) or find_first_column(
        online_df,
        ["order_date", "date"],
    )
    columns = ["weekday", "weekday_short", "orders", "revenue", "average_order_value"]
    if online_df.empty or not sales_column:
        return pd.DataFrame(columns=columns)

    weekday_frame = online_df.copy()
    weekday_frame["_weekday_revenue"] = pd.to_numeric(
        weekday_frame[sales_column],
        errors="coerce",
    ).fillna(0)
    if weekday_column:
        weekday_frame["_weekday_name"] = weekday_frame[weekday_column].astype("string").str.strip()
    elif date_column:
        weekday_frame["_weekday_name"] = parse_date_series(weekday_frame[date_column]).dt.day_name()
    else:
        return pd.DataFrame(columns=columns)

    weekday_frame["_weekday_name"] = weekday_frame["_weekday_name"].astype("string").str.strip()
    weekday_frame = weekday_frame[
        weekday_frame["_weekday_name"].notna()
        & weekday_frame["_weekday_name"].ne("")
        & ~weekday_frame["_weekday_name"].str.lower().isin(["nan", "nat", "none", "null"])
    ]
    if weekday_frame.empty:
        return pd.DataFrame(columns=columns)

    weekday_aliases = {
        "mon": "Monday",
        "monday": "Monday",
        "tue": "Tuesday",
        "tues": "Tuesday",
        "tuesday": "Tuesday",
        "wed": "Wednesday",
        "wednesday": "Wednesday",
        "thu": "Thursday",
        "thur": "Thursday",
        "thurs": "Thursday",
        "thursday": "Thursday",
        "fri": "Friday",
        "friday": "Friday",
        "sat": "Saturday",
        "saturday": "Saturday",
        "sun": "Sunday",
        "sunday": "Sunday",
    }
    weekday_frame["_weekday_name"] = weekday_frame["_weekday_name"].map(
        lambda value: weekday_aliases.get(str(value).strip().lower(), str(value).strip())
    )

    if order_column:
        grouped = weekday_frame.groupby("_weekday_name", dropna=True).agg(
            orders=(order_column, "nunique"),
            revenue=("_weekday_revenue", "sum"),
        )
    else:
        grouped = weekday_frame.groupby("_weekday_name", dropna=True).agg(
            orders=("_weekday_revenue", "size"),
            revenue=("_weekday_revenue", "sum"),
        )

    weekday_order = [
        ("Monday", "Mon"),
        ("Tuesday", "Tue"),
        ("Wednesday", "Wed"),
        ("Thursday", "Thu"),
        ("Friday", "Fri"),
        ("Saturday", "Sat"),
        ("Sunday", "Sun"),
    ]
    ordered_rows: list[dict] = []
    for weekday_name, weekday_short in weekday_order:
        if weekday_name in grouped.index:
            orders = int(grouped.loc[weekday_name, "orders"])
            revenue = float(grouped.loc[weekday_name, "revenue"])
        else:
            orders = 0
            revenue = 0.0
        ordered_rows.append(
            {
                "weekday": weekday_name,
                "weekday_short": weekday_short,
                "orders": orders,
                "revenue": revenue,
                "average_order_value": revenue / orders if orders else 0.0,
            }
        )

    return pd.DataFrame(ordered_rows, columns=columns)


def format_grain_period_subtitle(online_df: pd.DataFrame) -> str:
    date_column = usable_column(online_df, FILTER_DATE_COLUMN) or find_first_column(
        online_df,
        ["order_date", "date"],
    )
    if not date_column or date_column not in online_df.columns:
        return "Selected Period Combined"

    dates = parse_date_series(online_df[date_column]).dropna()
    if dates.empty:
        return "Selected Period Combined"

    start_month = dates.min().to_period("M").to_timestamp()
    end_month = dates.max().to_period("M").to_timestamp()
    if start_month == end_month:
        return f"{start_month.strftime('%b %Y')} Combined"
    if start_month.year == end_month.year:
        return f"{start_month.strftime('%b')}–{end_month.strftime('%b %Y')} Combined"
    return f"{start_month.strftime('%b %Y')}–{end_month.strftime('%b %Y')} Combined"


def weekday_order_revenue_empty_chart(subtitle: str) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.update_layout(
        title=online_chart_title(
            "Orders & Revenue by Day of Week",
            subtitle,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        annotations=[
            {
                "text": "No weekday order data available for selected filters.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": "#9A9A9A", "size": 14},
            }
        ],
        margin={"l": 42, "r": 48, "t": 92, "b": 52},
        height=520,
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 14, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickfont={"size": 12, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        tickfont={"size": 12, "color": "#9A9A9A"},
        showgrid=False,
        zeroline=False,
        rangemode="tozero",
        secondary_y=True,
    )
    return figure


def build_weekday_order_revenue_chart(
    weekday_frame: pd.DataFrame,
    subtitle: str,
) -> go.Figure:
    if weekday_frame.empty or float(weekday_frame["orders"].sum()) <= 0:
        return weekday_order_revenue_empty_chart(subtitle)

    best_revenue = float(weekday_frame["revenue"].max())
    best_mask = weekday_frame["revenue"] == best_revenue
    customdata = weekday_frame[
        ["weekday", "orders", "revenue", "average_order_value"]
    ].to_numpy()
    bar_colors = ["#79B8FF" if is_best else "#5B9DFF" for is_best in best_mask.tolist()]
    marker_sizes = [13 if is_best else 9 for is_best in best_mask.tolist()]
    order_labels = [
        format_whole_number(int(value)) if int(value) > 0 else ""
        for value in weekday_frame["orders"].tolist()
    ]

    revenue_labels = [
        f"€{float(value) / 1_000:.1f}k"
        if float(value) >= 1_000
        else f"€{float(value):.0f}"
        if float(value) > 0
        else ""
        for value in weekday_frame["revenue"].tolist()
    ]

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            name="Orders",
            x=weekday_frame["weekday_short"].tolist(),
            y=weekday_frame["orders"].tolist(),
            marker={"color": bar_colors, "line": {"color": "#5B9DFF", "width": 0}},
            width=0.58,
            customdata=customdata,
            text=order_labels,
            texttemplate="<b>%{text}</b>",
            textposition="inside",
            textfont={"color": "#FFFFFF", "size": 12},
            insidetextanchor="start",
            cliponaxis=False,
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "Orders: %{customdata[1]:,.0f}<br>"
            + "Revenue: €%{customdata[2]:,.2f}<br>"
            + "Avg Order Value: €%{customdata[3]:,.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            name="Revenue",
            x=weekday_frame["weekday_short"].tolist(),
            y=weekday_frame["revenue"].tolist(),
            mode="lines+markers+text",
            line={"color": "#38D27A", "width": 3, "shape": "spline", "smoothing": 0.9},
            marker={
                "size": marker_sizes,
                "color": "#38D27A",
                "line": {"color": "#38D27A", "width": 1},
            },
            customdata=customdata,
            text=revenue_labels,
            textposition="top center",
            textfont={"color": "#38D27A", "size": 11},
            texttemplate="<b>%{text}</b>",
            cliponaxis=False,
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + "Orders: %{customdata[1]:,.0f}<br>"
            + "Revenue: €%{customdata[2]:,.2f}<br>"
            + "Avg Order Value: €%{customdata[3]:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    max_orders = float(weekday_frame["orders"].max()) if not weekday_frame.empty else 0.0
    max_revenue = float(weekday_frame["revenue"].max()) if not weekday_frame.empty else 0.0
    figure.update_layout(
        title=online_chart_title(
            "Orders & Revenue by Day of Week",
            subtitle,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 42, "r": 48, "t": 92, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.935,
            "yref": "container",
            "xanchor": "left",
            "x": 0.34,
            "xref": "container",
            "font": {"color": "#CFC7BD", "size": 14},
            "tracegroupgap": 8,
            "itemwidth": 30,
        },
        height=520,
        hovermode="x unified",
        bargap=0.36,
    )
    figure.update_xaxes(
        title_text="",
        categoryorder="array",
        categoryarray=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        tickfont={"size": 14, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        range=[0, max_orders * 1.28 if max_orders > 0 else 1],
        tickfont={"size": 12, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="",
        range=[0, max_revenue * 1.28 if max_revenue > 0 else 1],
        tickprefix="€",
        tickformat="~s",
        tickfont={"size": 12, "color": "#9A9A9A"},
        showgrid=False,
        zeroline=False,
        rangemode="tozero",
        secondary_y=True,
    )
    return figure


def render_weekday_order_revenue_section(online_df: pd.DataFrame) -> None:
    subtitle = format_grain_period_subtitle(online_df)
    weekday_frame = build_weekday_order_revenue_frame(online_df)
    with st.container(key="grain_weekday_order_revenue_card"):
        st.plotly_chart(
            build_weekday_order_revenue_chart(weekday_frame, subtitle),
            use_container_width=True,
            config={"displayModeBar": False},
        )


def hour_bucket_sort_key(value: object) -> int:
    text = str(value).strip()
    match = re.match(r"^(\d{1,2}):", text)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d{1,2})", text)
    if match:
        return int(match.group(1))
    return 99


def build_hourly_revenue_frame(online_df: pd.DataFrame) -> pd.DataFrame:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    hour_column = find_first_column(
        online_df,
        ["hour_slot", "hour_bucket", "hour", "time_slot"],
    )
    time_column = find_first_column(online_df, ["time", "order_time"])
    columns = ["hour_bucket", "orders", "revenue", "average_order_value", "sort_key"]
    if online_df.empty or not sales_column:
        return pd.DataFrame(columns=columns)

    hourly_frame = online_df.copy()
    hourly_frame["_hourly_revenue"] = pd.to_numeric(
        hourly_frame[sales_column],
        errors="coerce",
    ).fillna(0)
    if hour_column:
        hourly_frame["_hour_bucket"] = hourly_frame[hour_column].map(normalize_hour_bucket)
    elif time_column:
        hourly_frame["_hour_bucket"] = hourly_frame[time_column].map(normalize_hour_bucket)
    else:
        return pd.DataFrame(columns=columns)

    hourly_frame["_hour_bucket"] = hourly_frame["_hour_bucket"].astype("string").str.strip()
    hourly_frame = hourly_frame[
        hourly_frame["_hour_bucket"].notna()
        & hourly_frame["_hour_bucket"].ne("")
        & ~hourly_frame["_hour_bucket"].str.lower().isin(["nan", "nat", "none", "null"])
    ]
    if hourly_frame.empty:
        return pd.DataFrame(columns=columns)

    if order_column:
        grouped = hourly_frame.groupby("_hour_bucket", dropna=True).agg(
            orders=(order_column, "nunique"),
            revenue=("_hourly_revenue", "sum"),
        )
    else:
        grouped = hourly_frame.groupby("_hour_bucket", dropna=True).agg(
            orders=("_hourly_revenue", "size"),
            revenue=("_hourly_revenue", "sum"),
        )

    hourly_revenue = grouped.reset_index().rename(columns={"_hour_bucket": "hour_bucket"})
    hourly_revenue["orders"] = hourly_revenue["orders"].astype(int)
    hourly_revenue["revenue"] = pd.to_numeric(hourly_revenue["revenue"], errors="coerce").fillna(0)
    hourly_revenue["average_order_value"] = hourly_revenue.apply(
        lambda row: float(row["revenue"]) / int(row["orders"])
        if int(row["orders"]) > 0
        else 0.0,
        axis=1,
    )
    hourly_revenue["sort_key"] = hourly_revenue["hour_bucket"].map(hour_bucket_sort_key)
    return hourly_revenue.sort_values(["sort_key", "hour_bucket"]).reset_index(drop=True)[columns]


def compact_euro_one_decimal(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:.1f}m"
    return f"€{value / 1_000:.1f}k"


def hourly_revenue_empty_chart(subtitle: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=online_chart_title(
            "Revenue Heatmap by Hour",
            subtitle,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        annotations=[
            {
                "text": "No hourly revenue data available for selected filters.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": "#9A9A9A", "size": 14},
            }
        ],
        margin={"l": 96, "r": 62, "t": 92, "b": 52},
        height=520,
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        tickfont={"size": 12, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    figure.update_yaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    return figure


def build_hourly_revenue_heatmap_chart(hourly_frame: pd.DataFrame, subtitle: str) -> go.Figure:
    if hourly_frame.empty or float(hourly_frame["revenue"].sum()) <= 0:
        return hourly_revenue_empty_chart(subtitle)

    display_frame = hourly_frame.copy()
    max_revenue = float(display_frame["revenue"].max()) if not display_frame.empty else 0.0
    min_revenue = float(display_frame["revenue"].min()) if not display_frame.empty else 0.0
    revenue_span = max(max_revenue - min_revenue, 1.0)

    bar_colors: list[str] = []
    for revenue in display_frame["revenue"].tolist():
        if float(revenue) == max_revenue:
            bar_colors.append("#38D27A")
            continue
        intensity = (float(revenue) - min_revenue) / revenue_span
        bar_colors.append(blend_hex_colors("#263345", "#5B9DFF", 0.22 + intensity * 0.66))

    display_frame["revenue_label"] = display_frame["revenue"].map(
        lambda value: compact_euro_one_decimal(float(value))
    )
    customdata = display_frame[["revenue", "orders", "average_order_value"]].to_numpy()

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Revenue",
            x=display_frame["revenue"].tolist(),
            y=display_frame["hour_bucket"].tolist(),
            orientation="h",
            marker={"color": bar_colors, "line": {"color": "#1A1A1A", "width": 1}},
            text=display_frame["revenue_label"].tolist(),
            texttemplate="<b>%{text}</b>",
            textposition="outside",
            textfont={"color": "#FFFFFF", "size": 12},
            customdata=customdata,
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>"
            + "Revenue: €%{customdata[0]:,.2f}<br>"
            + "Orders: %{customdata[1]:,.0f}<br>"
            + "Avg Order Value: €%{customdata[2]:,.2f}<extra></extra>",
            width=0.62,
        )
    )

    figure.update_layout(
        title=online_chart_title(
            "Revenue Heatmap by Hour",
            subtitle,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 96, "r": 62, "t": 92, "b": 52},
        height=520,
        showlegend=False,
        bargap=0.32,
    )
    figure.update_xaxes(
        title_text="",
        range=[0, max_revenue * 1.22 if max_revenue > 0 else 1],
        tickprefix="€",
        tickformat="~s",
        tickfont={"size": 12, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    figure.update_yaxes(
        title_text="",
        autorange="reversed",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    return figure


def render_hourly_revenue_heatmap_section(online_df: pd.DataFrame) -> None:
    subtitle = format_grain_period_subtitle(online_df)
    hourly_frame = build_hourly_revenue_frame(online_df)
    with st.container(key="grain_hourly_revenue_heatmap_card"):
        st.plotly_chart(
            build_hourly_revenue_heatmap_chart(hourly_frame, subtitle),
            use_container_width=True,
            config={"displayModeBar": False},
        )


def format_ticket_period_label(online_df: pd.DataFrame) -> str:
    date_column = usable_column(online_df, FILTER_DATE_COLUMN) or find_first_column(
        online_df,
        ["order_date", "date"],
    )
    if not date_column or date_column not in online_df.columns:
        return "Selected period"

    dates = parse_date_series(online_df[date_column]).dropna()
    if dates.empty:
        return "Selected period"

    month_starts = dates.dt.to_period("M").dt.to_timestamp().drop_duplicates().sort_values()
    if len(month_starts) == 1:
        return month_starts.iloc[0].strftime("%b %Y")
    if len(month_starts) == 2 and month_starts.iloc[0].year == month_starts.iloc[1].year:
        return (
            f"{month_starts.iloc[0].strftime('%b')} & "
            f"{month_starts.iloc[1].strftime('%b %Y')}"
        )

    start_month = month_starts.iloc[0]
    end_month = month_starts.iloc[-1]
    if start_month.year == end_month.year:
        return f"{start_month.strftime('%b')}–{end_month.strftime('%b %Y')}"
    return f"{start_month.strftime('%b %Y')}–{end_month.strftime('%b %Y')}"


def build_ticket_size_distribution_frame(online_df: pd.DataFrame) -> pd.DataFrame:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    columns = ["bucket", "orders", "percentage", "revenue", "color"]
    if online_df.empty or not sales_column:
        return pd.DataFrame(columns=columns)

    ticket_frame = online_df.copy()
    ticket_frame["_ticket_value"] = pd.to_numeric(
        ticket_frame[sales_column],
        errors="coerce",
    ).fillna(0)
    if order_column:
        order_values = (
            ticket_frame.groupby(order_column, dropna=True)["_ticket_value"]
            .sum()
            .reset_index(drop=True)
            .rename("ticket_value")
            .to_frame()
        )
    else:
        order_values = ticket_frame[["_ticket_value"]].rename(
            columns={"_ticket_value": "ticket_value"}
        )

    order_values = order_values[pd.to_numeric(order_values["ticket_value"], errors="coerce").notna()]
    if order_values.empty:
        return pd.DataFrame(columns=columns)

    bucket_specs = [
        ("€0–15", 0.0, 15.0, "#214F8B"),
        ("€15–25", 15.0, 25.0, "#5B9DFF"),
        ("€25–35", 25.0, 35.0, "#38D27A"),
        ("€35–50", 35.0, 50.0, "#F59E0B"),
        ("€50+", 50.0, math.inf, "#9B7CF6"),
    ]
    total_orders = int(len(order_values.index))
    rows: list[dict] = []
    values = pd.to_numeric(order_values["ticket_value"], errors="coerce").fillna(0)
    for index, (bucket, lower, upper, color) in enumerate(bucket_specs):
        if index == 0:
            mask = (values >= lower) & (values <= upper)
        elif math.isinf(upper):
            mask = values > lower
        else:
            mask = (values > lower) & (values <= upper)
        bucket_values = values[mask]
        orders = int(len(bucket_values.index))
        revenue = float(bucket_values.sum())
        rows.append(
            {
                "bucket": bucket,
                "orders": orders,
                "percentage": orders / total_orders * 100.0 if total_orders else 0.0,
                "revenue": revenue,
                "color": color,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def render_ticket_size_distribution_summary(ticket_frame: pd.DataFrame) -> None:
    if ticket_frame.empty or int(ticket_frame["orders"].sum()) <= 0:
        st.markdown(
            '<div class="ticket-size-empty">No ticket size data available for selected filters.</div>',
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []
    for row in ticket_frame.itertuples(index=False):
        progress_width = max(0.0, min(100.0, float(row.percentage)))
        rows.append(
            '<div class="ticket-size-bucket-card" '
            f'style="border-left-color:{escape(str(row.color))};">'
            f'<div class="ticket-size-bucket">{escape(str(row.bucket))}</div>'
            f'<div class="ticket-size-orders">{escape(format_whole_number(int(row.orders)))} orders</div>'
            '<div class="ticket-size-progress-track">'
            '<div class="ticket-size-progress-fill" '
            f'style="width:{progress_width:.1f}%; background:{escape(str(row.color))};"></div>'
            "</div>"
            f'<div class="ticket-size-percent">{float(row.percentage):.1f}% of orders</div>'
            "</div>"
        )
    st.markdown(
        '<div class="ticket-size-bucket-grid">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_ticket_size_distribution_section(online_df: pd.DataFrame) -> None:
    ticket_frame = build_ticket_size_distribution_frame(online_df)
    total_orders = int(ticket_frame["orders"].sum()) if not ticket_frame.empty else 0
    period_label = format_ticket_period_label(online_df)
    subtitle = (
        "Number of orders by order value bucket — "
        f"{period_label} · {format_whole_number(total_orders)} orders total"
    )

    with st.container(key="grain_ticket_size_distribution_card"):
        st.markdown(
            '<div class="online-chart-title">Ticket Size Distribution</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="ticket-size-card-subtitle">{escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )
        render_ticket_size_distribution_summary(ticket_frame)


def render_metric_row(label: str, value: str) -> str:
    return (
        '<div class="detailed-metric-row">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        "</div>"
    )


def render_detailed_partner_card(
    partner_name: str,
    date_range: str,
    orders: int,
    gross_sales: float,
    total_deductions: float,
    net_payout: float,
    deduction_rate: float,
    average_order_value: float,
) -> None:
    display_name = partner_display_name(partner_name)
    initials, avatar_background, avatar_text = partner_avatar_style(partner_name)
    metrics = "".join(
        [
            render_metric_row("Total orders", format_whole_number(orders)),
            render_metric_row("Gross sales", format_euro(gross_sales)),
            render_metric_row("Total deductions", format_euro(total_deductions)),
            render_metric_row("Net payout", format_euro(net_payout)),
            render_metric_row("Deduction rate", format_percent(deduction_rate)),
            render_metric_row("Avg order value", format_euro(average_order_value)),
        ]
    )

    html = (
        '<div class="detailed-partner-card">'
        '<div class="detailed-card-header">'
        f'<div class="partner-avatar" style="background: {avatar_background}; color: {avatar_text};">'
        f"{escape(initials)}</div>"
        "<div>"
        f'<div class="detailed-partner-name">{escape(display_name)}</div>'
        f'<div class="detailed-date-range">{escape(date_range)}</div>'
        "</div>"
        "</div>"
        f'<div class="detailed-metrics">{metrics}</div>'
        "</div>"
    )
    st.html(html)


def render_partner_card(partner_name: str, gross_sales: float, orders: int, aov: float) -> None:
    accent = partner_accent_color(partner_name)
    st.markdown(
        f"""
        <div class="partner-card" style="border-left-color: {accent};">
            <div class="partner-name">{escape(partner_display_name(partner_name))}</div>
            <div class="partner-value">{escape(format_euro(gross_sales))}</div>
            <div class="partner-subtext">
                {escape(format_whole_number(orders))} orders &bull; AOV {escape(format_euro(aov))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_compact_euro(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"€{value / 1_000_000:.1f}m"
    if absolute >= 1_000:
        scaled = value / 1_000
        return f"€{scaled:.1f}k" if absolute >= 10_000 else f"€{scaled:.2f}k"
    return f"€{value:.0f}"


def build_top_ordered_products_frame(
    product_analysis_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    if product_analysis_df.empty:
        return pd.DataFrame(columns=["product", "ordered_count"])

    filtered_products = apply_partner_year_month_filters(
        product_analysis_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    if filtered_products.empty:
        return pd.DataFrame(columns=["product", "ordered_count"])

    product_candidates = [
        "product",
        "item",
        "product_name",
        "item_name",
        "menu_item",
        "product_title",
        "product_title_name",
        "dish_name",
        "article_name",
        "product_nm",
        "sku",
        "name",
    ]

    product_column = find_first_column(filtered_products, product_candidates)
    if not product_column:
        for column in filtered_products.columns:
            lowered = str(column).lower()
            if any(
                token in lowered for token in ["product", "item", "dish", "article", "name"]
            ):
                product_column = column
                break
    if not product_column:
        return pd.DataFrame(columns=["product", "ordered_count"])

    quantity_keywords = ("qty", "quantity", "count", "orders", "items", "units", "sold")
    partner_tokens = {
        "Wolt": ("wolt",),
        "Uber Eats": ("uber", "ubereats"),
        "Lieferando": ("lieferando", "lefrando"),
    }

    def has_quantity_keyword(column_name: str) -> bool:
        lowered = column_name.lower()
        return any(token in lowered for token in quantity_keywords)

    def score_quantity_column(column_name: str) -> int:
        lowered = column_name.lower()
        score = 0
        for token in ("total", "online", "ordered", "orders", "quantity", "qty"):
            if token in lowered:
                score += 1
        return score

    def best_column(candidates: list[str]) -> str | None:
        if not candidates:
            return None
        return max(candidates, key=score_quantity_column)

    selected_partner_name = partner_display_name(selected_partner)
    selected_tokens = partner_tokens.get(selected_partner_name, ())
    partner_quantity_candidates: list[str] = []
    if selected_partner_name != "All Partners":
        partner_quantity_candidates = [
            column
            for column in filtered_products.columns
            if has_quantity_keyword(str(column))
            and any(token in str(column).lower() for token in selected_tokens)
        ]

    total_quantity_column = find_first_column(
        filtered_products,
        [
            "total_online_quantity",
            "total_online_qty",
            "total_ordered_quantity",
            "online_ordered_quantity",
            "online_quantity",
            "online_qty",
            "total_quantity",
            "total_qty",
            "ordered_quantity",
            "order_quantity",
            "quantity",
            "qty",
            "total",
            "count",
            "orders",
            "units",
            "items",
        ],
    )

    if not total_quantity_column:
        non_partner_quantity_candidates = [
            column
            for column in filtered_products.columns
            if has_quantity_keyword(str(column))
            and not any(
                partner_key in str(column).lower()
                for partner_key in ("wolt", "uber", "ubereats", "lieferando", "lefrando")
            )
        ]
        total_quantity_column = best_column(non_partner_quantity_candidates)

    quantity_column = (
        best_column(partner_quantity_candidates)
        if selected_partner_name != "All Partners"
        else total_quantity_column
    )
    if not quantity_column:
        quantity_column = total_quantity_column

    if not quantity_column:
        return pd.DataFrame(columns=["product", "ordered_count"])

    product_frame = filtered_products.copy()
    product_frame["_product_name"] = (
        product_frame[product_column].astype(str).str.strip()
    )
    product_frame = product_frame[
        product_frame["_product_name"].ne("")
        & product_frame["_product_name"].str.lower().ne("nan")
    ]
    if product_frame.empty:
        return pd.DataFrame(columns=["product", "ordered_count"])

    product_frame["_ordered_count"] = pd.to_numeric(
        product_frame[quantity_column], errors="coerce"
    ).fillna(0)
    grouped = (
        product_frame.groupby("_product_name", dropna=True)["_ordered_count"]
        .sum()
        .reset_index()
    )

    return (
        grouped.rename(columns={"_product_name": "product", "_ordered_count": "ordered_count"})
        .query("ordered_count > 0")
        .sort_values("ordered_count", ascending=False)
        .reset_index(drop=True)
    )


def build_top_ordered_products_chart(
    product_orders: pd.DataFrame, top_n: int | str
) -> go.Figure:
    figure = go.Figure()
    if product_orders.empty:
        figure.update_layout(
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            margin={"l": 24, "r": 24, "t": 12, "b": 22},
            height=420,
            annotations=[
                {
                    "text": "No product order data available for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
        )
        return figure

    ranked = product_orders.sort_values("ordered_count", ascending=False).copy()
    total_orders = float(ranked["ordered_count"].sum())
    if str(top_n).lower() == "all":
        top = ranked.copy()
    else:
        top = ranked.head(int(top_n)).copy()
    top["orders_share_pct"] = (
        top["ordered_count"] / total_orders * 100.0 if total_orders else 0.0
    )
    display = top.iloc[::-1]
    max_orders = float(display["ordered_count"].max()) if not display.empty else 0.0

    figure.add_trace(
        go.Bar(
            x=display["ordered_count"].tolist(),
            y=display["product"].tolist(),
            orientation="h",
            marker={"color": "#5A8DEE", "cornerradius": 8},
            width=0.44,
            text=[format_whole_number(int(round(value))) for value in display["ordered_count"]],
            texttemplate="<b>%{text}</b>",
            textposition="auto",
            textfont={"size": 14, "color": "#FFFFFF"},
            insidetextanchor="end",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>"
            + "Total ordered: %{x:,.0f}<br>"
            + "% of total orders: %{customdata:.1f}%<extra></extra>",
            customdata=display["orders_share_pct"].tolist(),
        )
    )

    figure.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 24, "r": 52, "t": 2, "b": 20},
        height=420,
        showlegend=False,
        bargap=0.52,
    )
    figure.update_xaxes(
        title_text="",
        tickformat=",.0f",
        tickfont={"size": 12, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        range=[0, max_orders * 1.14 if max_orders > 0 else 1],
    )
    figure.update_yaxes(
        title_text="",
        tickfont={"size": 13, "color": "#FFFFFF"},
        automargin=True,
    )

    return figure


def render_top_ordered_products_section(
    product_analysis_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> None:
    with st.container(key="top_ordered_products_card"):
        header_left, header_right = st.columns([2.85, 0.72], vertical_alignment="top")
        with header_left:
            st.markdown(
                '<div class="online-chart-title">Top Ordered Products</div>',
                unsafe_allow_html=True,
            )
        with header_right:
            st.markdown('<div class="top-ordered-filter-label">SHOW</div>', unsafe_allow_html=True)
            top_n = st.selectbox(
                "Show",
                [3, 5, 8, 10, "All"],
                index=0,
                key="top_ordered_products_limit",
                label_visibility="collapsed",
            )

        product_orders = build_top_ordered_products_frame(
            product_analysis_df,
            selected_partner,
            selected_year,
            selected_month,
        )
        top_chart = build_top_ordered_products_chart(product_orders, top_n or 3)
        st.plotly_chart(
            top_chart,
            use_container_width=True,
            config={"displayModeBar": False},
        )


def partner_color(partner_name: str) -> str:
    name = partner_display_name(partner_name)
    if name == "Wolt":
        return "#4A9EFF"
    if name == "Uber Eats":
        return "#2ECC71"
    if name == "Lieferando":
        return "#FF6B35"
    return "#7A7A7A"


def blend_hex_colors(base_color: str, blend_with: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    base = base_color.lstrip("#")
    blend = blend_with.lstrip("#")
    if len(base) != 6 or len(blend) != 6:
        return base_color

    def channel(hex_value: str, index: int) -> int:
        return int(hex_value[index : index + 2], 16)

    mixed = []
    for idx in (0, 2, 4):
        start = channel(base, idx)
        end = channel(blend, idx)
        mixed.append(round(start + (end - start) * ratio))

    return f"#{mixed[0]:02X}{mixed[1]:02X}{mixed[2]:02X}"


def format_compact_value(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}m"
    if absolute >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.0f}"


def build_revenue_breakdown_frame(
    online_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    partner_column = find_first_column(online_df, ["partner", "patner"])

    if online_df.empty or not sales_column or not partner_column:
        return pd.DataFrame(columns=["partner", "gross_sales", "net_payout"])

    online_frame = online_df.copy()
    online_frame["_partner_display"] = online_frame[partner_column].map(
        valid_partner_display_name
    )
    online_frame = online_frame[online_frame["_partner_display"].notna()]
    online_frame["_gross_sales"] = pd.to_numeric(
        online_frame[sales_column], errors="coerce"
    ).fillna(0)
    gross_by_partner = (
        online_frame.groupby("_partner_display", dropna=True)["_gross_sales"]
        .sum()
        .rename("gross_sales")
        .reset_index()
        .rename(columns={"_partner_display": "partner"})
    )

    filtered_payouts = apply_payout_filters(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    payout_partner_column = find_first_column(filtered_payouts, ["partner", "patner"])
    net_payout_column = find_first_column(
        filtered_payouts,
        [
            "net_sale_payout",
            "net_sale",
            "net_payout",
            "net_revenue",
            "net_revenue_after_delivery_partner_deductions",
            "payout",
        ],
    )

    if (
        filtered_payouts.empty
        or not payout_partner_column
        or not net_payout_column
    ):
        revenue_frame = gross_by_partner.copy()
        revenue_frame["net_payout"] = 0.0
    else:
        payouts_frame = filtered_payouts.copy()
        payouts_frame["_partner_display"] = payouts_frame[payout_partner_column].map(
            valid_partner_display_name
        )
        payouts_frame = payouts_frame[payouts_frame["_partner_display"].notna()]
        payouts_frame["_net_payout"] = pd.to_numeric(
            payouts_frame[net_payout_column], errors="coerce"
        ).fillna(0)
        net_by_partner = (
            payouts_frame.groupby("_partner_display", dropna=True)["_net_payout"]
            .sum()
            .rename("net_payout")
            .reset_index()
            .rename(columns={"_partner_display": "partner"})
        )
        revenue_frame = gross_by_partner.merge(net_by_partner, on="partner", how="outer")
        revenue_frame[["gross_sales", "net_payout"]] = revenue_frame[
            ["gross_sales", "net_payout"]
        ].fillna(0.0)

    if revenue_frame.empty:
        return revenue_frame

    revenue_frame["partner"] = revenue_frame["partner"].astype(str)
    revenue_frame = revenue_frame.sort_values("gross_sales", ascending=False)
    return revenue_frame


def build_gross_net_chart(revenue_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    if revenue_frame.empty:
        figure.update_layout(
            title=online_chart_title("Gross vs Net by Partner (€)"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No data for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 30, "r": 20, "t": 60, "b": 40},
            height=410,
        )
        return figure

    partners = revenue_frame["partner"].tolist()
    gross_values = revenue_frame["gross_sales"].tolist()
    net_values = revenue_frame["net_payout"].tolist()
    max_chart_value = max([0.0, *gross_values, *net_values])
    y_range_top = max_chart_value * 1.15 if max_chart_value > 0 else 1.0

    def nice_tick_step(raw_step: float) -> float:
        if raw_step <= 0:
            return 1.0
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / (10**exponent)
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
        return nice_fraction * (10**exponent)

    if y_range_top <= 7_000:
        y_tick_step = 1_000.0
    else:
        y_tick_step = nice_tick_step(y_range_top / 4)
    y_top_tick = math.ceil(y_range_top / y_tick_step) * y_tick_step
    base_colors = [partner_color(partner) for partner in partners]
    gross_colors = [blend_hex_colors(color, "#FFFFFF", 0.35) for color in base_colors]
    net_colors = [blend_hex_colors(color, "#000000", 0.18) for color in base_colors]

    figure.add_trace(
        go.Bar(
            name="Gross",
            x=partners,
            y=gross_values,
            marker_color=gross_colors,
            text=[format_compact_value(value) for value in gross_values],
            textposition="auto",
            insidetextanchor="middle",
            textfont={"size": 13, "color": "#111111"},
            hovertemplate="%{x}<br>Gross: €%{y:,.2f}<extra></extra>",
            width=0.30,
            offsetgroup="gross",
            cliponaxis=False,
        )
    )
    figure.add_trace(
        go.Bar(
            name="Net",
            x=partners,
            y=net_values,
            marker_color=net_colors,
            text=[format_compact_value(value) for value in net_values],
            textposition="auto",
            insidetextanchor="middle",
            textfont={"size": 13, "color": "#FFFFFF"},
            hovertemplate="%{x}<br>Net: €%{y:,.2f}<extra></extra>",
            width=0.30,
            offsetgroup="net",
            cliponaxis=False,
        )
    )

    figure.update_layout(
        title=online_chart_title("Gross vs Net by Partner (€)"),
        barmode="group",
        bargap=0.36,
        bargroupgap=0.12,
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 42, "r": 20, "t": 90, "b": 82},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
            "font": {"color": "#CFC7BD", "size": 12},
        },
        height=470,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 14, "color": "#CFC7BD"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        dtick=y_tick_step,
        tickprefix="€",
        tickformat="~s",
        range=[0, y_top_tick],
        tickfont={"size": 12, "color": "#CFC7BD"},
        gridcolor="rgba(42,42,42,0.9)",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
        ticklen=0,
    )

    return figure


def build_gross_share_chart(revenue_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()

    if revenue_frame.empty or float(revenue_frame["gross_sales"].sum()) <= 0:
        figure.update_layout(
            title=online_chart_title("Gross Sales Share"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No gross sales for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 20, "r": 20, "t": 60, "b": 20},
            height=410,
        )
        return figure

    labels = revenue_frame["partner"].tolist()
    values = revenue_frame["gross_sales"].tolist()
    colors = [partner_color(partner) for partner in labels]

    figure.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.64,
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont={"color": "#FFFFFF", "size": 14},
            marker={"colors": colors, "line": {"color": "#1A1A1A", "width": 2}},
            hovertemplate="%{label}<br>Gross sales: €%{value:,.2f}<br>Share: %{percent}<extra></extra>",
            sort=False,
            domain={"x": [0.26, 0.98], "y": [0.04, 0.96]},
        )
    )

    figure.update_layout(
        title=online_chart_title("Gross Sales Share"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        legend={
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.50,
            "xanchor": "left",
            "x": 0.01,
            "font": {"color": "#9A9A9A", "size": 14},
            "tracegroupgap": 14,
        },
        margin={"l": 20, "r": 20, "t": 95, "b": 26},
        height=470,
    )

    return figure


def build_monthly_gross_sales_chart(online_df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    partner_column = find_first_column(online_df, ["partner", "patner"])
    date_column = find_first_column(online_df, ["order_date", "date"])
    year_column = find_first_column(online_df, ["fy", "year"])
    month_column = find_first_column(online_df, ["month"])

    if online_df.empty or not sales_column or not partner_column:
        figure.update_layout(
            title=online_chart_title("Monthly Gross Sales (€)"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No data for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 24, "r": 20, "t": 90, "b": 28},
            height=430,
        )
        return figure

    trend_frame = online_df.copy()
    trend_frame["_partner_display"] = trend_frame[partner_column].map(partner_display_name)
    trend_frame["_gross_sales"] = pd.to_numeric(trend_frame[sales_column], errors="coerce").fillna(0)

    month_dates = pd.Series(pd.NaT, index=trend_frame.index, dtype="datetime64[ns]")
    if date_column:
        month_dates = pd.to_datetime(
            trend_frame[date_column], errors="coerce", dayfirst=True
        ).dt.to_period("M").dt.to_timestamp()

    if month_dates.isna().all() and month_column:
        month_text = trend_frame[month_column].astype(str).str.strip()
        if year_column and year_column in trend_frame.columns:
            year_text = trend_frame[year_column].astype(str).str.strip()
            combined = month_text + " " + year_text
            month_dates = pd.to_datetime(
                combined, format="%B %Y", errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            unresolved = month_dates.isna()
            if unresolved.any():
                month_dates.loc[unresolved] = pd.to_datetime(
                    combined.loc[unresolved], format="%b %Y", errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
        else:
            month_dates = pd.to_datetime(month_text, format="%B", errors="coerce")
            unresolved = month_dates.isna()
            if unresolved.any():
                month_dates.loc[unresolved] = pd.to_datetime(
                    month_text.loc[unresolved], format="%b", errors="coerce"
                )
            month_dates = month_dates.dt.to_period("M").dt.to_timestamp()

    trend_frame["_month_date"] = month_dates
    trend_frame = trend_frame.dropna(subset=["_month_date"])

    if trend_frame.empty:
        figure.update_layout(
            title=online_chart_title("Monthly Gross Sales (€)"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No monthly timeline for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 24, "r": 20, "t": 90, "b": 28},
            height=430,
        )
        return figure

    monthly_totals = (
        trend_frame.groupby(["_month_date", "_partner_display"], dropna=True)["_gross_sales"]
        .sum()
        .reset_index()
        .sort_values("_month_date")
    )
    month_order = monthly_totals["_month_date"].drop_duplicates().sort_values()
    month_labels = [month.strftime("%b %Y") for month in month_order]
    monthly_totals["_month_label"] = monthly_totals["_month_date"].dt.strftime("%b %Y")
    range_label = f"{month_order.iloc[0].strftime('%b %Y')} – {month_order.iloc[-1].strftime('%b %Y')}"

    partners_order = (
        monthly_totals.groupby("_partner_display")["_gross_sales"].sum().sort_values(ascending=False).index
    )

    area_fill_color_by_partner = {
        "Wolt": "rgba(74, 158, 255, 0.18)",
        "Uber Eats": "rgba(46, 204, 113, 0.18)",
        "Lieferando": "rgba(255, 107, 53, 0.18)",
    }

    for partner_name in partners_order:
        partner_rows = monthly_totals[monthly_totals["_partner_display"] == partner_name]
        values_by_month = (
            partner_rows.set_index("_month_label")["_gross_sales"].reindex(month_labels)
        )
        partner_display = str(partner_name)
        color = partner_color(partner_display)
        area_fill = area_fill_color_by_partner.get(partner_display, "rgba(122, 122, 122, 0.18)")
        figure.add_trace(
            go.Scatter(
                name=partner_display,
                x=month_labels,
                y=values_by_month.tolist(),
                mode="lines+markers",
                line={"color": color, "width": 4, "shape": "spline", "smoothing": 1.0},
                marker={"size": 10, "color": color, "line": {"width": 1, "color": color}},
                fill="tozeroy",
                fillcolor=area_fill,
                hovertemplate="%{x}<br>"
                + escape(partner_display)
                + ": €%{y:,.2f}<extra></extra>",
            )
        )

    max_value = float(monthly_totals["_gross_sales"].max()) if not monthly_totals.empty else 0.0
    y_top = max_value * 1.18 if max_value > 0 else 1.0
    y_step = 2000 if y_top <= 10000 else 5000

    figure.update_layout(
        title=online_chart_title("Monthly Gross Sales (€)", range_label),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 34, "r": 18, "t": 92, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.935,
            "yref": "container",
            "xanchor": "left",
            "x": 0.34,
            "xref": "container",
            "font": {"color": "#CFC7BD", "size": 14},
            "tracegroupgap": 8,
            "itemwidth": 30,
        },
        height=520,
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 16, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        dtick=y_step,
        range=[0, y_top],
        tickfont={"size": 16, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )

    return figure


def build_monthly_sales_mom_chart(online_df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    date_column = find_first_column(online_df, ["order_date", "date"])
    year_column = find_first_column(online_df, ["fy", "year"])
    month_column = find_first_column(online_df, ["month"])

    if online_df.empty or not sales_column:
        figure.update_layout(
            title=online_chart_title("Monthly Sales & MoM Change"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No data for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 28, "r": 28, "t": 90, "b": 40},
            height=520,
        )
        return figure

    trend_frame = online_df.copy()
    trend_frame["_gross_sales"] = pd.to_numeric(trend_frame[sales_column], errors="coerce").fillna(0)

    month_dates = pd.Series(pd.NaT, index=trend_frame.index, dtype="datetime64[ns]")
    if date_column:
        month_dates = pd.to_datetime(
            trend_frame[date_column], errors="coerce", dayfirst=True
        ).dt.to_period("M").dt.to_timestamp()

    if month_dates.isna().all() and month_column:
        month_text = trend_frame[month_column].astype(str).str.strip()
        if year_column and year_column in trend_frame.columns:
            year_text = trend_frame[year_column].astype(str).str.strip()
            combined = month_text + " " + year_text
            month_dates = pd.to_datetime(
                combined, format="%B %Y", errors="coerce"
            ).dt.to_period("M").dt.to_timestamp()
            unresolved = month_dates.isna()
            if unresolved.any():
                month_dates.loc[unresolved] = pd.to_datetime(
                    combined.loc[unresolved], format="%b %Y", errors="coerce"
                ).dt.to_period("M").dt.to_timestamp()
        else:
            month_dates = pd.to_datetime(month_text, format="%B", errors="coerce")
            unresolved = month_dates.isna()
            if unresolved.any():
                month_dates.loc[unresolved] = pd.to_datetime(
                    month_text.loc[unresolved], format="%b", errors="coerce"
                )
            month_dates = month_dates.dt.to_period("M").dt.to_timestamp()

    trend_frame["_month_date"] = month_dates
    trend_frame = trend_frame.dropna(subset=["_month_date"])

    if trend_frame.empty:
        figure.update_layout(
            title=online_chart_title("Monthly Sales & MoM Change"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No monthly timeline for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 28, "r": 28, "t": 90, "b": 40},
            height=520,
        )
        return figure

    monthly_sales = (
        trend_frame.groupby("_month_date", dropna=True)["_gross_sales"]
        .sum()
        .reset_index()
        .sort_values("_month_date")
    )
    monthly_sales["month_label"] = monthly_sales["_month_date"].dt.strftime("%b %Y")
    monthly_sales["previous_sales"] = monthly_sales["_gross_sales"].shift(1)
    monthly_sales["mom_change_pct"] = (
        (monthly_sales["_gross_sales"] - monthly_sales["previous_sales"])
        / monthly_sales["previous_sales"]
        * 100.0
    )
    monthly_sales.loc[monthly_sales["previous_sales"] == 0, "mom_change_pct"] = pd.NA

    month_labels = monthly_sales["month_label"].tolist()
    sales_values = monthly_sales["_gross_sales"].tolist()
    mom_values = monthly_sales["mom_change_pct"].tolist()
    range_label = (
        f"{monthly_sales['_month_date'].iloc[0].strftime('%b %Y')} – "
        f"{monthly_sales['_month_date'].iloc[-1].strftime('%b %Y')}"
    )

    figure.add_trace(
        go.Bar(
            name="Monthly Gross Sales",
            x=month_labels,
            y=sales_values,
            marker={"color": "#5A8DEE"},
            opacity=0.9,
            text=[f"€{format_compact_value(float(value))}" for value in sales_values],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "#FFFFFF", "size": 13, "family": "Inter, Segoe UI, sans-serif"},
            hovertemplate="%{x}<br>Sales: €%{y:,.2f}<extra></extra>",
            yaxis="y",
        )
    )

    positive_mom = [value if pd.notna(value) and value >= 0 else None for value in mom_values]
    negative_mom = [value if pd.notna(value) and value < 0 else None for value in mom_values]

    figure.add_trace(
        go.Scatter(
            name="MoM Change %",
            x=month_labels,
            y=positive_mom,
            mode="lines+markers",
            line={"color": "#2ECC71", "width": 3, "shape": "spline", "smoothing": 0.9},
            marker={"size": 8, "color": "#FFFFFF", "line": {"color": "#2ECC71", "width": 2}},
            hovertemplate="%{x}<br>MoM: %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="MoM Change % (Negative)",
            x=month_labels,
            y=negative_mom,
            mode="lines+markers",
            line={"color": "#FF5C5C", "width": 3, "shape": "spline", "smoothing": 0.9},
            marker={"size": 8, "color": "#FFFFFF", "line": {"color": "#FF5C5C", "width": 2}},
            hovertemplate="%{x}<br>MoM: %{y:.1f}%<extra></extra>",
            yaxis="y2",
            showlegend=False,
        )
    )

    figure.update_layout(
        title=online_chart_title("Monthly Sales & MoM Change", range_label),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 34, "r": 34, "t": 92, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.935,
            "yref": "container",
            "xanchor": "left",
            "x": 0.34,
            "xref": "container",
            "font": {"color": "#CFC7BD", "size": 14},
            "tracegroupgap": 8,
            "itemwidth": 30,
        },
        height=520,
        barmode="group",
        hovermode="x unified",
        yaxis={
            "title": "",
            "tickprefix": "€",
            "tickformat": "~s",
            "tickfont": {"size": 12, "color": "#FFFFFF"},
            "gridcolor": "#2A2A2A",
            "zerolinecolor": "#2A2A2A",
            "rangemode": "tozero",
        },
        yaxis2={
            "title": "",
            "overlaying": "y",
            "side": "right",
            "ticksuffix": "%",
            "tickformat": ".0f",
            "tickfont": {"size": 12, "color": "#FFFFFF"},
            "showgrid": False,
            "zeroline": False,
        },
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 14, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )

    return figure


def parse_month_start_text_series(series: pd.Series) -> pd.Series:
    month_start = parse_date_series(series).dt.to_period("M").dt.to_timestamp().copy()
    missing = month_start.isna()

    if missing.any():
        month_year_text = (
            series.loc[missing]
            .astype("string")
            .str.extract(r"([A-Za-z]{3,9}\s+\d{4})", expand=False)
        )
        parsed_month_year = pd.to_datetime(
            month_year_text,
            format="%B %Y",
            errors="coerce",
        )
        unresolved = parsed_month_year.isna()
        if unresolved.any():
            parsed_month_year.loc[unresolved] = pd.to_datetime(
                month_year_text.loc[unresolved],
                format="%b %Y",
                errors="coerce",
            )
        month_start.loc[missing] = parsed_month_year.dt.to_period("M").dt.to_timestamp().copy()

    missing = month_start.isna()
    if missing.any():
        date_tokens = (
            series.loc[missing]
            .astype("string")
            .str.extract(
                r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
                expand=False,
            )
        )
        parsed_tokens = parse_date_series(date_tokens)
        month_start.loc[missing] = parsed_tokens.dt.to_period("M").dt.to_timestamp().copy()

    return month_start


def infer_payout_period_year(payouts_frame: pd.DataFrame, selected_year: str) -> str | None:
    if selected_year != "All Years" and re.fullmatch(r"\d{4}", str(selected_year)):
        return str(selected_year)

    year_values: list[str] = []
    for column in ("invoice_period", "period", "payout_period", "fy", "year"):
        if column not in payouts_frame.columns:
            continue
        extracted = (
            payouts_frame[column]
            .dropna()
            .astype("string")
            .str.extractall(r"(\d{4})")[0]
            .dropna()
            .astype(str)
            .tolist()
        )
        year_values.extend(extracted)

    unique_years = sorted(set(year_values))
    if len(unique_years) == 1:
        return unique_years[0]
    return None


def parse_payout_period_month_start(series: pd.Series, fallback_year: str | None) -> pd.Series:
    month_start = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    date_token = (
        series.astype("string")
        .str.extract(
            r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            expand=False,
        )
    )
    has_date_token = date_token.notna()
    if has_date_token.any():
        parsed_tokens = parse_date_series(date_token.loc[has_date_token])
        month_start.loc[has_date_token] = parsed_tokens.dt.to_period("M").dt.to_timestamp()

    missing = month_start.isna()
    month_pattern = (
        r"(?i)\b("
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept(?:ember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?"
        r")\b"
    )
    period_text = series.loc[missing].astype("string").str.strip()
    month_text = period_text.str.extract(month_pattern, expand=False)
    year_text = period_text.str.extract(r"(\d{4})", expand=False)
    if fallback_year:
        year_text = year_text.fillna(fallback_year)

    combined = month_text.fillna("") + " " + year_text.fillna("")
    parsed = pd.to_datetime(combined, format="%B %Y", errors="coerce")
    unresolved = parsed.isna()
    if unresolved.any():
        parsed.loc[unresolved] = pd.to_datetime(
            combined.loc[unresolved],
            format="%b %Y",
            errors="coerce",
        )
    month_start.loc[missing] = parsed.dt.to_period("M").dt.to_timestamp().copy()
    return month_start


def month_number_from_text(value: object) -> int | None:
    parsed = pd.to_datetime(pd.Series([value]), format="%B", errors="coerce").iloc[0]
    if pd.isna(parsed):
        parsed = pd.to_datetime(pd.Series([value]), format="%b", errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return int(parsed.month)


def year_for_month_label(year_value: object, month_number: object) -> str:
    text = "" if pd.isna(year_value) else str(year_value)
    years = re.findall(r"\d{4}", text)
    if not years:
        return ""
    numeric_month = pd.to_numeric(pd.Series([month_number]), errors="coerce").iloc[0]
    if len(years) >= 2 and pd.notna(numeric_month) and int(numeric_month) in (1, 2, 3):
        return years[1]
    return years[0]


def payout_month_start_series(payouts_frame: pd.DataFrame, selected_year: str) -> pd.Series:
    month_start = pd.Series(pd.NaT, index=payouts_frame.index, dtype="datetime64[ns]")
    fallback_year = infer_payout_period_year(payouts_frame, selected_year)

    date_column = usable_column(payouts_frame, FILTER_DATE_COLUMN) or find_first_column(
        payouts_frame,
        ["date", "order_date"],
    )
    if date_column:
        month_start = pd.to_datetime(
            payouts_frame[date_column],
            errors="coerce",
            dayfirst=True,
        ).dt.to_period("M").dt.to_timestamp()

    for period_column in ("invoice_period", "period", "payout_period"):
        if period_column not in payouts_frame.columns or month_start.notna().all():
            continue
        missing = month_start.isna()
        parsed_periods = parse_payout_period_month_start(
            payouts_frame.loc[missing, period_column],
            fallback_year,
        )
        month_start.loc[missing] = parsed_periods

    month_column = usable_column(payouts_frame, FILTER_MONTH_COLUMN) or find_first_column(
        payouts_frame,
        ["month"],
    )
    year_column = usable_column(payouts_frame, FILTER_YEAR_COLUMN) or find_first_column(
        payouts_frame,
        ["fy", "year"],
    )
    if month_column and year_column and month_start.isna().any():
        missing = month_start.isna()
        month_numbers = payouts_frame.loc[missing, FILTER_MONTH_NUMBER_COLUMN].copy()
        unresolved_months = month_numbers.isna()
        if unresolved_months.any():
            month_numbers.loc[unresolved_months] = payouts_frame.loc[
                missing,
                month_column,
            ].map(month_number_from_text)

        years = [
            year_for_month_label(year_value, month_number)
            for year_value, month_number in zip(
                payouts_frame.loc[missing, year_column],
                month_numbers,
            )
        ]
        combined = payouts_frame.loc[missing, month_column].astype("string").str.strip() + " " + pd.Series(
            years,
            index=payouts_frame.index[missing],
            dtype="string",
        )
        parsed_months = pd.to_datetime(combined, format="%B %Y", errors="coerce")
        unresolved = parsed_months.isna()
        if unresolved.any():
            parsed_months.loc[unresolved] = pd.to_datetime(
                combined.loc[unresolved],
                format="%b %Y",
                errors="coerce",
            )
        replacement_month_start = pd.Series(
            parsed_months.dt.to_period("M").dt.to_timestamp().to_numpy(),
            index=frame.index[missing],
            dtype="datetime64[ns]",
        )
        month_start = month_start.where(
            ~missing,
            replacement_month_start.reindex(month_start.index),
        )

    return month_start


def monthly_deduction_empty_chart(subtitle_text: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=online_chart_title(
            "Monthly Deduction Trends",
            subtitle_text,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        annotations=[
            {
                "text": "No deduction trend data available for selected filters.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": "#9A9A9A", "size": 14},
            }
        ],
        margin={"l": 34, "r": 24, "t": 112, "b": 48},
        height=430,
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#9A9A9A"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickfont={"size": 13, "color": "#9A9A9A"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
    )
    return figure


def empty_monthly_deduction_trend_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "month_date",
            "month",
            "partner",
            "deduction_amount",
            "gross_sales",
            "deduction_rate",
        ]
    )


def build_monthly_deduction_trend_frame(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> pd.DataFrame:
    filtered_payouts = apply_payout_filters(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    if filtered_payouts.empty:
        return empty_monthly_deduction_trend_frame()

    partner_column = find_first_column(filtered_payouts, ["partner", "patner"])
    gross_sale_column = find_first_column(
        filtered_payouts,
        [
            "gross_sale",
            "gross_sales",
            "gross_revenue",
            "gross_revenue_before_partner_deductions",
            "gross",
            "sale",
            "sales",
        ],
    )
    net_payout_column = find_first_column(
        filtered_payouts,
        [
            "net_sale_payout",
            "net_sale",
            "net_payout",
            "net_revenue",
            "net_revenue_after_delivery_partner_deductions",
            "payout",
        ],
    )
    deduction_column = find_first_column(
        filtered_payouts,
        [
            "total_deductions",
            "total_deduction",
            "deductions",
            "deduction",
            "partner_deduction",
            "delivery_partner_deduction",
            "commission",
            "commission_amount",
            "fees",
            "fee",
            "charges",
            "charge",
            "platform_fee",
            "service_fee",
        ],
    )

    if not partner_column or not gross_sale_column or (not deduction_column and not net_payout_column):
        return empty_monthly_deduction_trend_frame()

    trend_frame = filtered_payouts.copy()
    trend_frame["_partner_display"] = trend_frame[partner_column].map(valid_partner_display_name)
    trend_frame = trend_frame[trend_frame["_partner_display"].notna()]
    if trend_frame.empty:
        return empty_monthly_deduction_trend_frame()

    trend_frame["_month_date"] = payout_month_start_series(trend_frame, selected_year)
    trend_frame = trend_frame.dropna(subset=["_month_date"])
    if trend_frame.empty:
        return empty_monthly_deduction_trend_frame()

    trend_frame["_gross_sale"] = pd.to_numeric(
        trend_frame[gross_sale_column],
        errors="coerce",
    ).fillna(0)
    if deduction_column:
        trend_frame["_deductions"] = pd.to_numeric(
            trend_frame[deduction_column],
            errors="coerce",
        ).fillna(0)
    else:
        trend_frame["_net_payout"] = pd.to_numeric(
            trend_frame[net_payout_column],
            errors="coerce",
        ).fillna(0)
        trend_frame["_deductions"] = (
            trend_frame["_gross_sale"] - trend_frame["_net_payout"]
        ).clip(lower=0)

    monthly_deductions = (
        trend_frame.groupby(["_month_date", "_partner_display"], dropna=True)[
            ["_deductions", "_gross_sale"]
        ]
        .sum()
        .reset_index()
        .sort_values("_month_date")
    )
    if monthly_deductions.empty:
        return empty_monthly_deduction_trend_frame()

    monthly_deductions["_deduction_rate"] = 0.0
    positive_gross = monthly_deductions["_gross_sale"] > 0
    monthly_deductions.loc[positive_gross, "_deduction_rate"] = (
        monthly_deductions.loc[positive_gross, "_deductions"]
        / monthly_deductions.loc[positive_gross, "_gross_sale"]
        * 100.0
    )
    monthly_deductions["month"] = monthly_deductions["_month_date"].dt.strftime("%b %Y")

    return (
        monthly_deductions.rename(
            columns={
                "_month_date": "month_date",
                "_partner_display": "partner",
                "_deductions": "deduction_amount",
                "_gross_sale": "gross_sales",
                "_deduction_rate": "deduction_rate",
            }
        )[
            [
                "month_date",
                "month",
                "partner",
                "deduction_amount",
                "gross_sales",
                "deduction_rate",
            ]
        ]
        .sort_values(["month_date", "partner"])
        .reset_index(drop=True)
    )


def build_monthly_deduction_trends_chart(
    trend_df: pd.DataFrame,
    metric_mode: str,
) -> go.Figure:
    if trend_df.empty:
        return monthly_deduction_empty_chart()

    month_order = trend_df["month_date"].drop_duplicates().sort_values()
    month_labels = [month.strftime("%b %Y") for month in month_order]
    subtitle = (
        month_order.iloc[0].strftime("%b %Y")
        if len(month_order) == 1
        else f"{month_order.iloc[0].strftime('%b %Y')} – {month_order.iloc[-1].strftime('%b %Y')}"
    )
    value_column = "deduction_amount" if metric_mode == "€ Amount" else "deduction_rate"
    is_amount = value_column == "deduction_amount"

    figure = go.Figure()
    dash_by_partner = {
        "Wolt": "solid",
        "Uber Eats": "dash",
        "Lieferando": "dot",
    }
    partner_totals = (
        trend_df.groupby("partner")[value_column]
        .sum()
        .reindex(VALID_PARTNERS)
        .dropna()
    )
    for partner_name in partner_totals.index:
        partner_rows = trend_df[trend_df["partner"] == partner_name]
        values_by_month = (
            partner_rows.set_index("month")[value_column].reindex(month_labels)
        )
        color = partner_color(str(partner_name))
        hovertemplate = (
            "<b>%{fullData.name}</b><br>"
            + "Month: %{x}<br>"
            + (
                "Deduction amount: €%{y:,.2f}<extra></extra>"
                if is_amount
                else "Deduction rate: %{y:.1f}%<extra></extra>"
            )
        )
        figure.add_trace(
            go.Scatter(
                name=str(partner_name),
                x=month_labels,
                y=values_by_month.tolist(),
                mode="lines+markers",
                line={
                    "color": color,
                    "width": 3,
                    "shape": "spline",
                    "smoothing": 0.9,
                    "dash": dash_by_partner.get(str(partner_name), "solid"),
                },
                marker={"size": 9, "color": color, "line": {"width": 1, "color": color}},
                connectgaps=False,
                hovertemplate=hovertemplate,
            )
        )

    max_value = float(trend_df[value_column].max()) if not trend_df.empty else 0.0
    y_top = max_value * 1.18 if max_value > 0 else 1.0
    y_axis_settings = {
        "title_text": "",
        "range": [0, y_top],
        "tickfont": {"size": 13, "color": "#9A9A9A"},
        "gridcolor": "#2A2A2A",
        "zerolinecolor": "#2A2A2A",
        "rangemode": "tozero",
        "ticklen": 0,
    }
    if is_amount:
        y_axis_settings.update({"tickprefix": "€", "tickformat": "~s"})
    else:
        y_axis_settings.update({"ticksuffix": "%", "tickformat": ".1f"})

    figure.update_layout(
        title=online_chart_title(
            "Monthly Deduction Trends",
            subtitle,
            subtitle_color="#9A9A9A",
        ),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 42, "r": 24, "t": 92, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.935,
            "yref": "container",
            "xanchor": "left",
            "x": 0.31,
            "xref": "container",
            "font": {"color": "#9A9A9A", "size": 14},
            "tracegroupgap": 8,
            "itemwidth": 30,
        },
        height=430,
        hovermode="x unified",
        showlegend=True,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#9A9A9A"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(**y_axis_settings)
    return figure


def render_monthly_deduction_trends_section(
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> None:
    metric_options = ["€ Amount", "% Rate"]
    if hasattr(st, "segmented_control"):
        metric_mode = st.segmented_control(
            "Deduction trend metric",
            metric_options,
            default=metric_options[0],
            key="monthly_deduction_trend_metric",
            label_visibility="collapsed",
        )
    else:
        metric_mode = st.radio(
            "Deduction trend metric",
            metric_options,
            horizontal=True,
            key="monthly_deduction_trend_metric",
            label_visibility="collapsed",
        )

    trend_df = build_monthly_deduction_trend_frame(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    st.plotly_chart(
        build_monthly_deduction_trends_chart(
            trend_df,
            metric_mode or metric_options[0],
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def build_deduction_rate_comparison_chart(
    online_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> go.Figure:
    figure = go.Figure()
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    partner_column = find_first_column(online_df, ["partner", "patner"])
    date_column = find_first_column(online_df, ["order_date", "date"])

    if online_df.empty or not sales_column or not partner_column:
        figure.update_layout(
            title=online_chart_title("Deduction Rate Comparison (%)"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No data for selected filters",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 34, "r": 24, "t": 92, "b": 40},
            height=460,
        )
        return figure

    partner_frame = online_df.copy()
    partner_frame["_partner_display"] = partner_frame[partner_column].map(partner_display_name)
    partner_frame["_gross_sales"] = pd.to_numeric(
        partner_frame[sales_column], errors="coerce"
    ).fillna(0)
    grouped = partner_frame.groupby("_partner_display", dropna=True)

    rows: list[dict] = []
    for partner_name in grouped.groups:
        partner_rows = grouped.get_group(partner_name)
        gross_sales = float(partner_rows["_gross_sales"].sum())
        deductions, _ = calculate_partner_payout_metrics(
            payouts_df,
            str(partner_name),
            selected_year,
            selected_month,
        )
        deduction_rate = (deductions / gross_sales * 100.0) if gross_sales else 0.0
        if date_column:
            date_range = format_partner_date_range(partner_rows, date_column)
            label = f"{partner_name} ({date_range})"
        else:
            label = str(partner_name)
        rows.append(
            {
                "partner": str(partner_name),
                "label": label,
                "deduction_rate": deduction_rate,
                "color": partner_color(str(partner_name)),
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        figure.update_layout(
            title=online_chart_title("Deduction Rate Comparison (%)"),
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            annotations=[
                {
                    "text": "No partner deduction data available",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
            margin={"l": 34, "r": 24, "t": 92, "b": 40},
            height=460,
        )
        return figure

    comparison = comparison.sort_values("deduction_rate", ascending=False)
    max_rate = float(comparison["deduction_rate"].max()) if not comparison.empty else 0.0
    min_rate = float(comparison["deduction_rate"].min()) if not comparison.empty else 0.0
    x_min = max(0.0, min_rate - 2.0) if min_rate > 2.0 else 0.0
    x_max = max_rate * 1.08 if max_rate > 0 else 1.0

    figure.add_trace(
        go.Bar(
            x=comparison["deduction_rate"].tolist(),
            y=comparison["label"].tolist(),
            orientation="h",
            marker={"color": comparison["color"].tolist()},
            text=[f"{value:.1f}%" for value in comparison["deduction_rate"].tolist()],
            textposition="outside",
            textfont={"color": "#EAE5DD", "size": 14},
            hovertemplate="%{y}<br>Deduction rate: %{x:.1f}%<extra></extra>",
        )
    )

    figure.update_layout(
        title=online_chart_title("Deduction Rate Comparison (%)"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 34, "r": 30, "t": 92, "b": 40},
        height=460,
        showlegend=False,
        bargap=0.42,
    )
    figure.update_xaxes(
        title_text="",
        range=[x_min, x_max],
        ticksuffix="%",
        tickformat=".0f",
        tickfont={"size": 13, "color": "#CFC7BD"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
    )
    figure.update_yaxes(
        title_text="",
        tickfont={"size": 13, "color": "#CFC7BD"},
        categoryorder="array",
        categoryarray=comparison["label"].tolist()[::-1],
    )

    return figure


def format_partner_date_range_with_dash(partner_df: pd.DataFrame, date_column: str | None) -> str:
    return format_partner_date_range(partner_df, date_column).replace(" to ", " - ")


def build_order_share_by_partner_frame(online_df: pd.DataFrame) -> pd.DataFrame:
    partner_column = find_first_column(online_df, ["partner", "patner"])
    date_column = usable_column(online_df, FILTER_DATE_COLUMN) or find_first_column(
        online_df,
        ["order_date", "date"],
    )
    if online_df.empty or (not partner_column and "_partner_display" not in online_df.columns):
        return pd.DataFrame(columns=["partner", "orders", "share_pct", "date_range", "color"])

    order_frame = online_df.copy()
    if "_partner_display" in order_frame.columns:
        order_frame["_partner_display"] = order_frame["_partner_display"].map(
            valid_partner_display_name
        )
    else:
        order_frame["_partner_display"] = order_frame[partner_column].map(valid_partner_display_name)
    order_frame = order_frame[order_frame["_partner_display"].notna()]
    order_frame = order_frame[order_frame["_partner_display"].isin(VALID_PARTNER_SET)]
    if order_frame.empty:
        return pd.DataFrame(columns=["partner", "orders", "share_pct", "date_range", "color"])

    order_column = find_first_column(order_frame, ["order_id"])
    grouped = order_frame.groupby("_partner_display", dropna=True)
    rows: list[dict] = []
    for partner_name in VALID_PARTNERS:
        if partner_name not in grouped.groups:
            continue
        partner_rows = grouped.get_group(partner_name)
        orders = (
            int(partner_rows[order_column].dropna().nunique())
            if order_column
            else int(len(partner_rows.index))
        )
        if orders <= 0:
            continue
        rows.append(
            {
                "partner": partner_name,
                "orders": orders,
                "date_range": format_partner_date_range_with_dash(partner_rows, date_column),
                "color": partner_color(partner_name),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["partner", "orders", "share_pct", "date_range", "color"])

    order_share_frame = pd.DataFrame(rows)
    total_orders = int(order_share_frame["orders"].sum())
    if total_orders <= 0:
        return pd.DataFrame(columns=["partner", "orders", "share_pct", "date_range", "color"])

    order_share_frame["share_pct"] = order_share_frame["orders"] / total_orders * 100.0
    order_share_frame = order_share_frame.sort_values("share_pct", ascending=False).reset_index(
        drop=True
    )
    return order_share_frame


def build_order_share_by_partner_chart(order_share_frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if order_share_frame.empty:
        figure.update_layout(
            paper_bgcolor="#1A1A1A",
            plot_bgcolor="#1A1A1A",
            font={"color": "#FFFFFF"},
            margin={"l": 6, "r": 6, "t": 6, "b": 6},
            height=390,
            annotations=[
                {
                    "text": "No valid partner order data available",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#9A9A9A", "size": 14},
                }
            ],
        )
        return figure

    figure.add_trace(
        go.Pie(
            labels=order_share_frame["partner"].tolist(),
            values=order_share_frame["orders"].tolist(),
            hole=0.36,
            sort=False,
            marker={
                "colors": order_share_frame["color"].tolist(),
                "line": {"color": "#1A1A1A", "width": 1},
            },
            texttemplate="<b>%{percent:.1%}</b>",
            textposition="inside",
            textfont={"color": "#F4EEE7", "size": 17},
            customdata=order_share_frame["orders"].tolist(),
            hovertemplate="<b>%{label}</b><br>"
            + "Orders: %{customdata:,.0f}<br>"
            + "Share: %{percent:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        showlegend=False,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        height=390,
    )
    return figure


def render_order_share_by_partner_section(
    online_df: pd.DataFrame,
    selected_year: str,
    selected_month: str,
) -> None:
    date_column = usable_column(online_df, FILTER_DATE_COLUMN) or find_first_column(
        online_df,
        ["order_date", "date"],
    )
    order_share_frame = build_order_share_by_partner_frame(online_df)
    total_orders = int(order_share_frame["orders"].sum()) if not order_share_frame.empty else 0

    if selected_month == "All Months":
        subtitle = (
            f"Total orders across all active months - {format_whole_number(total_orders)} orders"
        )
    else:
        selected_period = format_partner_date_range_with_dash(online_df, date_column)
        if selected_period == "Selected period":
            selected_period = (
                f"{selected_month} {selected_year}"
                if selected_year != "All Years"
                else selected_month
            )
        subtitle = f"Total orders in {selected_period} - {format_whole_number(total_orders)} orders"

    with st.container(key="order_share_partner_card"):
        st.markdown('<div class="online-chart-title">Order Share by Partner</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="order-share-card-subtitle">{escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )

        if order_share_frame.empty:
            st.info("No valid partner order data available for the selected filters.")
            return

        chart_column, breakdown_column = st.columns([1.08, 1], gap="medium")
        with chart_column:
            st.plotly_chart(
                build_order_share_by_partner_chart(order_share_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with breakdown_column:
            cards: list[str] = []
            for row in order_share_frame.itertuples(index=False):
                progress_width = max(0.0, min(100.0, float(row.share_pct)))
                cards.append(
                    '<div class="order-share-breakdown-card">'
                    '<div class="order-share-breakdown-head">'
                    '<div class="order-share-breakdown-partner">'
                    f'<span class="order-share-color-chip" style="background:{escape(str(row.color))};"></span>'
                    f'<span class="order-share-partner-name">{escape(str(row.partner))}</span>'
                    "</div>"
                    f'<div class="order-share-partner-percent" style="color:{escape(str(row.color))};">'
                    f"{escape(format_percent(float(row.share_pct)))}"
                    "</div>"
                    "</div>"
                    '<div class="order-share-partner-meta">'
                    f"{escape(format_whole_number(int(row.orders)))} orders"
                    f" &middot; {escape(str(row.date_range))}"
                    "</div>"
                    '<div class="order-share-progress-track">'
                    '<div class="order-share-progress-fill" '
                    f'style="width:{progress_width:.1f}%; background:{escape(str(row.color))};"></div>'
                    "</div>"
                    "</div>"
                )
            st.markdown("".join(cards), unsafe_allow_html=True)


def render_revenue_breakdown(
    online_df: pd.DataFrame,
    product_analysis_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> None:
    revenue_frame = build_revenue_breakdown_frame(
        online_df,
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    gross_net_chart = build_gross_net_chart(revenue_frame)
    gross_share_chart = build_gross_share_chart(revenue_frame)
    monthly_gross_chart = build_monthly_gross_sales_chart(online_df)
    monthly_sales_mom_chart = build_monthly_sales_mom_chart(online_df)
    deduction_rate_chart = build_deduction_rate_comparison_chart(
        online_df,
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )

    left_column, right_column = st.columns(2, gap="medium")
    with left_column:
        with st.container(key="revenue_chart_left"):
            st.plotly_chart(
                gross_net_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right_column:
        with st.container(key="revenue_chart_right"):
            st.plotly_chart(
                gross_share_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    trend_left_column, trend_right_column = st.columns(2, gap="medium")
    with trend_left_column:
        with st.container(key="revenue_chart_monthly_left"):
            st.plotly_chart(
                monthly_gross_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with trend_right_column:
        with st.container(key="revenue_chart_monthly_right"):
            st.plotly_chart(
                monthly_sales_mom_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown(
        '<div style="height: 36px;"></div><div class="online-section-heading">ORDERS INSIGHTS</div>',
        unsafe_allow_html=True,
    )

    share_left_column, share_right_column = st.columns(2, gap="medium")
    with share_left_column:
        render_order_share_by_partner_section(online_df, selected_year, selected_month)
    with share_right_column:
        render_top_ordered_products_section(
            product_analysis_df,
            selected_partner,
            selected_year,
            selected_month,
        )

    st.markdown(
        '<div style="height: 36px;"></div><div class="online-section-heading">DEDUCTION ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    deduction_left_column, deduction_right_column = st.columns(2, gap="medium")
    with deduction_left_column:
        with st.container(key="revenue_chart_deduction_left"):
            st.plotly_chart(
                deduction_rate_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with deduction_right_column:
        with st.container(key="revenue_chart_deduction_trends"):
            render_monthly_deduction_trends_section(
                payouts_df,
                selected_partner,
                selected_year,
                selected_month,
            )

    st.markdown(
        '<div style="height: 36px;"></div><div class="online-section-heading">GRAIN ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    render_grain_analysis_kpis(online_df)
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    grain_left_column, grain_right_column = st.columns(2, gap="medium")
    with grain_left_column:
        render_weekday_order_revenue_section(online_df)
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        render_ticket_size_distribution_section(online_df)
    with grain_right_column:
        render_hourly_revenue_heatmap_section(online_df)


def render_detailed_partner_cards(
    online_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    selected_year: str,
    selected_month: str,
) -> None:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    partner_column = find_first_column(online_df, ["partner", "patner"])
    date_column = find_first_column(online_df, ["order_date", "date"])

    if online_df.empty or not sales_column or not partner_column:
        st.info("No online partner data available for the selected filters.")
        return

    partner_frame = online_df.copy()
    partner_frame["_partner_display"] = partner_frame[partner_column].map(
        valid_partner_display_name
    )
    partner_frame = partner_frame[partner_frame["_partner_display"].notna()]
    if partner_frame.empty:
        st.info("No online partner data available for the selected filters.")
        return

    partner_frame["_online_sales"] = pd.to_numeric(
        partner_frame[sales_column], errors="coerce"
    ).fillna(0)

    columns = st.columns(3)
    grouped = partner_frame.groupby("_partner_display", dropna=True)
    partner_order = (
        grouped["_online_sales"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    for index, partner_name in enumerate(partner_order):
        partner_rows = grouped.get_group(partner_name)
        gross_sales = float(partner_rows["_online_sales"].sum())
        orders = (
            int(partner_rows[order_column].dropna().nunique())
            if order_column
            else int(len(partner_rows.index))
        )
        average_order_value = gross_sales / orders if orders else 0.0
        total_deductions, net_payout = calculate_partner_payout_metrics(
            payouts_df,
            str(partner_name),
            selected_year,
            selected_month,
        )
        deduction_rate = (total_deductions / gross_sales * 100.0) if gross_sales else 0.0
        date_range = format_partner_date_range(partner_rows, date_column)

        with columns[index % 3]:
            render_detailed_partner_card(
                str(partner_name),
                date_range,
                orders,
                gross_sales,
                total_deductions,
                net_payout,
                deduction_rate,
                average_order_value,
            )


def render_online_kpis(
    online_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    selected_partner: str,
    selected_year: str,
    selected_month: str,
) -> None:
    kpis = calculate_online_kpis(online_df)
    total_gross_sales = calculate_payout_gross_sales(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    net_sale = calculate_net_sale(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    total_deductions = calculate_total_deductions(
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
        total_gross_sales,
        net_sale,
    )
    deduction_percentage = (
        (total_deductions / total_gross_sales * 100.0) if total_gross_sales else 0.0
    )
    top_columns = st.columns(3)
    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
    bottom_columns = st.columns(3)

    with top_columns[0]:
        render_summary_card(
            "Total Gross Sales",
            format_euro(total_gross_sales),
            "All selected partners combined",
        )
    with top_columns[1]:
        render_summary_card(
            "Net Sale",
            format_euro(net_sale),
            "After partner deductions",
        )
    with top_columns[2]:
        render_summary_card(
            "Deduction",
            format_euro(total_deductions),
            f"Deduction percentage is {format_percent(deduction_percentage)}",
        )
    with bottom_columns[0]:
        render_summary_card(
            "Total Orders",
            format_whole_number(kpis["orders"]),
            "Across selected period",
        )
    with bottom_columns[1]:
        render_summary_card(
            "Average Order Value",
            format_euro(kpis["average_order_value"]),
            "Across selected orders",
        )
    with bottom_columns[2]:
        st.empty()


def render_partner_snapshot(online_df: pd.DataFrame) -> None:
    sales_column = find_first_column(online_df, ["gross_after_refund", "subtotal"])
    order_column = find_first_column(online_df, ["order_id"])
    partner_column = find_first_column(online_df, ["partner", "patner"])

    if online_df.empty or not sales_column or not partner_column:
        st.info("No partner data available.")
        return

    partner_frame = online_df.copy()
    partner_frame["_partner_display"] = partner_frame[partner_column].map(
        valid_partner_display_name
    )
    partner_frame = partner_frame[partner_frame["_partner_display"].notna()]
    if partner_frame.empty:
        st.info("No partner data available.")
        return

    partner_frame["_online_sales"] = pd.to_numeric(
        partner_frame[sales_column], errors="coerce"
    ).fillna(0)

    if order_column:
        grouped = (
            partner_frame.groupby("_partner_display", dropna=True)
            .agg(gross_sales=("_online_sales", "sum"), orders=(order_column, "nunique"))
            .sort_values("gross_sales", ascending=False)
        )
    else:
        grouped = (
            partner_frame.groupby("_partner_display", dropna=True)
            .agg(gross_sales=("_online_sales", "sum"), orders=("_online_sales", "size"))
            .sort_values("gross_sales", ascending=False)
        )

    columns = st.columns(3)
    for index, (partner_name, row) in enumerate(grouped.iterrows()):
        gross_sales = float(row["gross_sales"])
        orders = int(row["orders"])
        aov = gross_sales / orders if orders else 0.0
        with columns[index % 3]:
            render_partner_card(str(partner_name), gross_sales, orders, aov)


def reporting_month_start_series(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="datetime64[ns]")

    month_start = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    date_column = usable_column(frame, FILTER_DATE_COLUMN) or find_first_column(
        frame,
        ["order_date", "date"],
    )
    if date_column:
        month_start = parse_date_series(frame[date_column]).dt.to_period("M").dt.to_timestamp()

    month_column = usable_column(frame, FILTER_MONTH_COLUMN) or find_first_column(frame, ["month"])
    year_column = usable_column(frame, FILTER_YEAR_COLUMN) or find_first_column(frame, ["fy", "year"])
    if month_column and year_column and month_start.isna().any():
        missing = month_start.isna()
        month_numbers = frame.loc[missing, FILTER_MONTH_NUMBER_COLUMN].copy()
        unresolved_months = month_numbers.isna()
        if unresolved_months.any():
            month_numbers.loc[unresolved_months] = frame.loc[missing, month_column].map(
                month_number_from_text
            )
        years = [
            year_for_month_label(year_value, month_number)
            for year_value, month_number in zip(frame.loc[missing, year_column], month_numbers)
        ]
        combined = frame.loc[missing, month_column].astype("string").str.strip() + " " + pd.Series(
            years,
            index=frame.index[missing],
            dtype="string",
        )
        parsed_months = pd.to_datetime(combined, format="%B %Y", errors="coerce")
        unresolved = parsed_months.isna()
        if unresolved.any():
            parsed_months.loc[unresolved] = pd.to_datetime(
                combined.loc[unresolved],
                format="%b %Y",
                errors="coerce",
            )
        replacement_month_start = pd.Series(
            parsed_months.dt.to_period("M").dt.to_timestamp().to_numpy(),
            index=frame.index[missing],
            dtype="datetime64[ns]",
        )
        month_start = month_start.where(
            ~missing,
            replacement_month_start.reindex(month_start.index),
        )

    return month_start


def monthly_revenue_frame(frame: pd.DataFrame, revenue_column: str | None) -> pd.DataFrame:
    if frame.empty or not revenue_column or revenue_column not in frame.columns:
        return pd.DataFrame(columns=["month_date", "revenue"])

    revenue_frame = frame.copy()
    revenue_frame["_business_month"] = reporting_month_start_series(revenue_frame)
    revenue_frame["_business_revenue"] = pd.to_numeric(
        revenue_frame[revenue_column],
        errors="coerce",
    ).fillna(0)
    revenue_frame = revenue_frame.dropna(subset=["_business_month"])
    if revenue_frame.empty:
        return pd.DataFrame(columns=["month_date", "revenue"])

    return (
        revenue_frame.groupby("_business_month", dropna=True)["_business_revenue"]
        .sum()
        .reset_index()
        .rename(columns={"_business_month": "month_date", "_business_revenue": "revenue"})
        .sort_values("month_date")
        .reset_index(drop=True)
    )


def count_overall_orders(online_df: pd.DataFrame) -> int:
    if online_df.empty:
        return 0

    order_column = find_first_column(online_df, ["order_id"])
    if order_column:
        return int(online_df[order_column].dropna().nunique())
    return int(len(online_df.index))


def best_overall_gross_month(monthly_total_gross: pd.DataFrame) -> tuple[str, float]:
    if monthly_total_gross.empty:
        return "N/A", 0.0

    ranked = monthly_total_gross.copy()
    ranked["monthly_total_gross"] = pd.to_numeric(
        ranked["monthly_total_gross"],
        errors="coerce",
    ).fillna(0)
    ranked = ranked.sort_values(
        ["monthly_total_gross", "month_date"],
        ascending=[False, False],
    )
    if ranked.empty:
        return "N/A", 0.0

    best_row = ranked.iloc[0]
    month_date = pd.to_datetime(best_row["month_date"], errors="coerce")
    month_label = month_date.strftime("%b %Y") if pd.notna(month_date) else "N/A"
    return month_label, float(best_row["monthly_total_gross"])


def count_optional_offline_orders(offline_df: pd.DataFrame) -> int | None:
    if offline_df.empty:
        return None

    order_id_column = find_first_column(offline_df, ["order_id", "bill_nr", "billnr"])
    if order_id_column:
        return int(offline_df[order_id_column].dropna().nunique())

    order_count_column = find_first_column(
        offline_df,
        ["orders", "order_count", "total_orders"],
    )
    if not order_count_column:
        return None

    order_counts = pd.to_numeric(offline_df[order_count_column], errors="coerce").dropna()
    if order_counts.empty:
        return None
    return int(order_counts.sum())


def split_percentage(value: float | None, total: float) -> float | None:
    if value is None or total <= 0:
        return None
    return float(value) / total * 100.0


def format_split_percent(value: float | None) -> str:
    if value is None:
        return "Not available"
    return f"{value:.0f}%"


def split_progress_width(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(100.0, float(value)))


def format_split_value(value: float | int | None, value_type: str) -> str:
    if value is None:
        return "Not available"
    if value_type == "currency":
        return format_euro(float(value))
    return format_whole_number(int(value))


def build_overall_split_metrics(
    metrics: dict,
    offline_df: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    online_gross = float(
        pd.to_numeric(
            metrics["online_gross_by_month"].get("online_gross_sales", pd.Series(dtype="float64")),
            errors="coerce",
        ).fillna(0).sum()
    )
    offline_gross = float(
        pd.to_numeric(
            metrics["offline_total_by_month"].get("offline_total", pd.Series(dtype="float64")),
            errors="coerce",
        ).fillna(0).sum()
    )
    gross_total = online_gross + offline_gross

    online_orders = int(metrics["total_orders"])
    offline_orders = count_optional_offline_orders(offline_df)
    available_order_total = online_orders + (offline_orders if offline_orders is not None else 0)

    online_aov = online_gross / online_orders if online_orders else None
    offline_aov = (
        offline_gross / offline_orders
        if offline_orders is not None and offline_orders > 0
        else None
    )
    aov_total = (online_aov or 0.0) + (offline_aov or 0.0)

    return {
        "gross_sale": {
            "offline_value": offline_gross,
            "online_value": online_gross,
            "offline_percent": split_percentage(offline_gross, gross_total),
            "online_percent": split_percentage(online_gross, gross_total),
            "value_type": "currency",
        },
        "orders": {
            "offline_value": offline_orders,
            "online_value": online_orders,
            "offline_percent": split_percentage(offline_orders, available_order_total),
            "online_percent": split_percentage(online_orders, available_order_total),
            "value_type": "number",
        },
        "aov": {
            "offline_value": offline_aov,
            "online_value": online_aov,
            "offline_percent": split_percentage(offline_aov, aov_total),
            "online_percent": split_percentage(online_aov, aov_total),
            "value_type": "currency",
        },
    }


def build_overall_monthly_channel_frame(metrics: dict) -> pd.DataFrame:
    online_monthly = metrics["online_gross_by_month"].rename(
        columns={"online_gross_sales": "online"}
    )
    offline_monthly = metrics["offline_total_by_month"].rename(
        columns={"offline_total": "offline"}
    )
    month_parts = [
        frame["month_date"]
        for frame in (online_monthly, offline_monthly)
        if not frame.empty and "month_date" in frame.columns
    ]
    columns = ["month_date", "month_label", "offline", "online", "total"]
    if not month_parts:
        return pd.DataFrame(columns=columns)

    monthly = pd.DataFrame(
        {
            "month_date": pd.concat(month_parts, ignore_index=True)
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        }
    )
    monthly = monthly.merge(offline_monthly, on="month_date", how="left")
    monthly = monthly.merge(online_monthly, on="month_date", how="left")
    monthly["offline"] = pd.to_numeric(monthly["offline"], errors="coerce").fillna(0)
    monthly["online"] = pd.to_numeric(monthly["online"], errors="coerce").fillna(0)
    monthly["total"] = monthly["offline"] + monthly["online"]
    monthly["month_label"] = monthly["month_date"].dt.strftime("%b %Y")
    return monthly.sort_values("month_date").reset_index(drop=True)[columns]


def overall_sales_breakdown_empty_chart(title: str, subtitle: str, message: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"color": "#9A9A9A", "size": 14},
            }
        ],
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        height=520,
        showlegend=False,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    return figure


def build_monthly_total_gross_sale_chart(monthly_frame: pd.DataFrame) -> go.Figure:
    title = "Monthly Total Gross Sale"
    subtitle = "Offline + Online combined per month"
    if monthly_frame.empty or float(monthly_frame["total"].sum()) <= 0:
        return overall_sales_breakdown_empty_chart(
            title,
            subtitle,
            "No gross sales data available for selected filters.",
        )

    max_total = float(monthly_frame["total"].max())
    y_top = max_total * 1.22 if max_total > 0 else 1.0
    labels = [
        compact_euro_one_decimal(float(value)) if float(value) > 0 else ""
        for value in monthly_frame["total"].tolist()
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Total",
            x=monthly_frame["month_label"].tolist(),
            y=monthly_frame["total"].tolist(),
            marker={"color": "#22B8A9", "line": {"color": "#22B8A9", "width": 0}},
            text=labels,
            texttemplate="<b>%{text}</b>",
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "#FFFFFF", "size": 13},
            cliponaxis=False,
            hovertemplate="%{x}<br>Total gross sale: €%{y:,.2f}<extra></extra>",
            width=0.62,
        )
    )
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        height=520,
        showlegend=False,
        bargap=0.34,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        range=[0, y_top],
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    return figure


def build_online_offline_trend_chart(monthly_frame: pd.DataFrame) -> go.Figure:
    title = "Online vs Offline Trend"
    subtitle = "Month-by-month channel comparison"
    trend_columns = ["offline", "online"]
    if monthly_frame.empty or float(monthly_frame[trend_columns].sum().sum()) <= 0:
        return overall_sales_breakdown_empty_chart(
            title,
            subtitle,
            "No channel trend data available for selected filters.",
        )

    series_config = [
        ("Offline", "offline", "#2ECC71"),
        ("Online", "online", "#4A9EFF"),
    ]
    figure = go.Figure()
    for series_name, column, color in series_config:
        values = pd.to_numeric(monthly_frame[column], errors="coerce").fillna(0).tolist()
        labels = [
            compact_euro_one_decimal(float(value)) if float(value) > 0 else ""
            for value in values
        ]
        figure.add_trace(
            go.Scatter(
                name=series_name,
                x=monthly_frame["month_label"].tolist(),
                y=values,
                mode="lines+markers+text",
                line={"color": color, "width": 3, "shape": "spline", "smoothing": 0.8},
                marker={"size": 9, "color": color, "line": {"color": color, "width": 1}},
                text=labels,
                texttemplate="<b>%{text}</b>",
                textposition="top center",
                textfont={"color": color, "size": 11},
                cliponaxis=False,
                hovertemplate="%{x}<br>"
                + escape(series_name)
                + ": €%{y:,.2f}<extra></extra>",
            )
        )

    max_value = float(monthly_frame[trend_columns].max().max())
    y_top = max_value * 1.24 if max_value > 0 else 1.0
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
            "font": {"color": "#CFC7BD", "size": 13},
            "tracegroupgap": 8,
        },
        height=520,
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        range=[0, y_top],
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    return figure


def build_month_on_month_change_frame(
    monthly_frame: pd.DataFrame,
    channel: str,
) -> pd.DataFrame:
    columns = ["month_date", "month_label", "sales", "previous_sales", "mom_change_pct"]
    channel_column = {
        "Combined": "total",
        "Online": "online",
        "Offline": "offline",
    }.get(channel, "total")
    if monthly_frame.empty or channel_column not in monthly_frame.columns:
        return pd.DataFrame(columns=columns)

    frame = monthly_frame[["month_date", "month_label", channel_column]].copy()
    frame = frame.rename(columns={channel_column: "sales"})
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce").fillna(0)
    frame = frame[frame["sales"] > 0].sort_values("month_date").reset_index(drop=True)
    if len(frame.index) < 2:
        return pd.DataFrame(columns=columns)

    frame["previous_sales"] = frame["sales"].shift(1)
    valid_previous = pd.to_numeric(frame["previous_sales"], errors="coerce").fillna(0) > 0
    frame["mom_change_pct"] = pd.NA
    frame.loc[valid_previous, "mom_change_pct"] = (
        (frame.loc[valid_previous, "sales"] - frame.loc[valid_previous, "previous_sales"])
        / frame.loc[valid_previous, "previous_sales"]
        * 100.0
    )
    frame = frame.dropna(subset=["mom_change_pct"])
    return frame[columns].reset_index(drop=True)


def build_month_on_month_change_chart(mom_frame: pd.DataFrame) -> go.Figure:
    title = "Month-on-Month Change Percentage"
    subtitle = "Sales change vs previous month · combined gross sales"
    if mom_frame.empty:
        return overall_sales_breakdown_empty_chart(
            title,
            subtitle,
            "At least two months are required to calculate month-on-month change.",
        )

    changes = pd.to_numeric(mom_frame["mom_change_pct"], errors="coerce").fillna(0).tolist()
    marker_colors = ["#2ECC71" if value >= 0 else "#FF5C5C" for value in changes]
    labels = [format_signed_percent(float(value)) for value in changes]
    text_positions = ["top center" if value >= 0 else "bottom center" for value in changes]
    is_single_point = len(changes) == 1
    marker_size = 17 if is_single_point else 12
    if is_single_point:
        center_value = changes[0]
        span = max(abs(center_value), 8)
        y_bottom = min(0, center_value - span)
        y_top = max(0, center_value + span)
    else:
        min_change = min(changes + [0])
        max_change = max(changes + [0])
        span = max(max_change - min_change, 8)
        y_bottom = min_change - span * 0.28
        y_top = max_change + span * 0.34

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="MoM Change %",
            x=mom_frame["month_label"].tolist(),
            y=changes,
            mode="lines+markers+text",
            line={"color": "#9B7CF2", "width": 3, "shape": "spline", "smoothing": 0.8},
            marker={
                "size": marker_size,
                "color": marker_colors,
                "line": {"color": "#1A1A1A", "width": 2},
            },
            fill="tozeroy",
            fillcolor="rgba(155, 124, 242, 0.16)",
            text=labels,
            texttemplate="<b>%{text}</b>",
            textposition=text_positions,
            textfont={"color": "#E7E1D8", "size": 13 if is_single_point else 12},
            cliponaxis=False,
            customdata=list(
                zip(
                    mom_frame["sales"].astype(float).tolist(),
                    mom_frame["previous_sales"].astype(float).tolist(),
                )
            ),
            hovertemplate="%{x}<br>"
            "Current gross sales: €%{customdata[0]:,.2f}<br>"
            "Previous gross sales: €%{customdata[1]:,.2f}<br>"
            "MoM change: %{y:+.1f}%<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="#2A2A2A", line_width=1)
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        height=470,
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        ticksuffix="%",
        tickformat=".1f",
        range=[y_bottom, y_top],
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
    )
    return figure


def fiscal_quarter_label_and_start(month_date: object) -> tuple[str | None, object]:
    timestamp = pd.to_datetime(month_date, errors="coerce")
    if pd.isna(timestamp):
        return None, pd.NaT

    month = int(timestamp.month)
    year = int(timestamp.year)
    quarter = fiscal_quarter_for_month(month)
    if not quarter:
        return None, pd.NaT

    quarter_start_month = {
        "Q1": 4,
        "Q2": 7,
        "Q3": 10,
        "Q4": 1,
    }[quarter]
    return f"{quarter} {year}", pd.Timestamp(year=year, month=quarter_start_month, day=1)


def build_quarterly_growth_frame(monthly_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["quarter_label", "quarter_start", "revenue", "previous_revenue", "growth_pct"]
    if monthly_frame.empty or "month_date" not in monthly_frame.columns or "total" not in monthly_frame.columns:
        return pd.DataFrame(columns=columns)

    frame = monthly_frame[["month_date", "total"]].copy()
    frame["total"] = pd.to_numeric(frame["total"], errors="coerce").fillna(0)
    frame = frame[frame["total"] > 0].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    quarter_parts = frame["month_date"].map(fiscal_quarter_label_and_start)
    frame["quarter_label"] = quarter_parts.map(lambda value: value[0])
    frame["quarter_start"] = quarter_parts.map(lambda value: value[1])
    frame = frame.dropna(subset=["quarter_label", "quarter_start"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    quarterly = (
        frame.groupby(["quarter_label", "quarter_start"], dropna=True)["total"]
        .sum()
        .reset_index()
        .rename(columns={"total": "revenue"})
        .sort_values("quarter_start")
        .reset_index(drop=True)
    )
    quarterly["previous_revenue"] = quarterly["revenue"].shift(1)
    quarterly["growth_pct"] = pd.NA
    valid_previous = pd.to_numeric(
        quarterly["previous_revenue"],
        errors="coerce",
    ).fillna(0) > 0
    quarterly.loc[valid_previous, "growth_pct"] = (
        (quarterly.loc[valid_previous, "revenue"] - quarterly.loc[valid_previous, "previous_revenue"])
        / quarterly.loc[valid_previous, "previous_revenue"]
        * 100.0
    )
    return quarterly[columns].reset_index(drop=True)


def build_quarterly_revenue_chart(quarterly_frame: pd.DataFrame) -> go.Figure:
    title = "Quarter-by-Quarter Revenue"
    subtitle = "Revenue grouped by financial-year quarter"
    if quarterly_frame.empty or float(quarterly_frame["revenue"].sum()) <= 0:
        return overall_sales_breakdown_empty_chart(
            title,
            subtitle,
            "No quarterly gross revenue data available for selected filters.",
        )

    max_revenue = float(quarterly_frame["revenue"].max())
    y_top = max_revenue * 1.22 if max_revenue > 0 else 1.0
    labels = [
        compact_euro_one_decimal(float(value)) if float(value) > 0 else ""
        for value in quarterly_frame["revenue"].tolist()
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Total",
            x=quarterly_frame["quarter_label"].tolist(),
            y=quarterly_frame["revenue"].tolist(),
            marker={"color": "#22B8A9", "line": {"color": "#22B8A9", "width": 0}},
            text=labels,
            texttemplate="<b>%{text}</b>",
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": "#FFFFFF", "size": 13},
            cliponaxis=False,
            hovertemplate="%{x}<br>Total gross revenue: €%{y:,.2f}<extra></extra>",
            width=0.62,
        )
    )
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        height=520,
        showlegend=False,
        bargap=0.34,
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        tickprefix="€",
        tickformat="~s",
        range=[0, y_top],
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
        rangemode="tozero",
    )
    return figure


def build_quarterly_growth_chart(quarterly_frame: pd.DataFrame) -> go.Figure:
    title = "Quarterly Growth %"
    subtitle = "Financial-year quarter growth vs previous quarter"
    message = "At least two quarters are required to calculate quarterly growth."
    growth_frame = quarterly_frame.dropna(subset=["growth_pct"]).copy() if not quarterly_frame.empty else quarterly_frame
    if quarterly_frame.empty or len(quarterly_frame.index) < 2 or growth_frame.empty:
        return overall_sales_breakdown_empty_chart(title, subtitle, message)

    growth_values = pd.to_numeric(growth_frame["growth_pct"], errors="coerce").fillna(0).tolist()
    marker_colors = ["#2ECC71" if value >= 0 else "#FF5C7A" for value in growth_values]
    labels = [format_signed_percent(float(value)) for value in growth_values]
    text_positions = ["top center" if value >= 0 else "bottom center" for value in growth_values]
    is_single_point = len(growth_values) == 1
    if is_single_point:
        center_value = growth_values[0]
        span = max(abs(center_value), 8)
        y_bottom = min(0, center_value - span)
        y_top = max(0, center_value + span)
    else:
        min_growth = min(growth_values + [0])
        max_growth = max(growth_values + [0])
        span = max(max_growth - min_growth, 8)
        y_bottom = min_growth - span * 0.32
        y_top = max_growth + span * 0.38

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Quarterly Growth %",
            x=growth_frame["quarter_label"].tolist(),
            y=growth_values,
            mode="lines+markers+text",
            line={"color": "#9B7CF2", "width": 3, "shape": "spline", "smoothing": 0.8},
            marker={
                "size": 15 if is_single_point else 12,
                "color": marker_colors,
                "line": {"color": "#1A1A1A", "width": 2},
            },
            fill="tozeroy",
            fillcolor="rgba(155, 124, 242, 0.16)",
            text=labels,
            texttemplate="<b>%{text}</b>",
            textposition=text_positions,
            textfont={"color": "#E7E1D8", "size": 12},
            cliponaxis=False,
            customdata=list(
                zip(
                    growth_frame["revenue"].astype(float).tolist(),
                    growth_frame["previous_revenue"].astype(float).tolist(),
                )
            ),
            hovertemplate="%{x}<br>"
            "Current quarter revenue: €%{customdata[0]:,.2f}<br>"
            "Previous quarter revenue: €%{customdata[1]:,.2f}<br>"
            "Quarterly growth: %{y:+.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Positive",
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 11, "color": "#2ECC71"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Negative",
            x=[None],
            y=[None],
            mode="markers",
            marker={"size": 11, "color": "#FF5C7A"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Zero line",
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#2A2A2A", "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_hline(y=0, line_color="#2A2A2A", line_width=1)
    figure.update_layout(
        title=online_chart_title(title, subtitle, subtitle_color="#9A9A9A"),
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        font={"color": "#FFFFFF"},
        margin={"l": 52, "r": 34, "t": 92, "b": 56},
        height=520,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#CFC7BD", "size": 13},
            "tracegroupgap": 8,
        },
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="",
        tickfont={"size": 13, "color": "#B8B0A7"},
        showgrid=False,
        linecolor="#2A2A2A",
        ticklen=0,
    )
    figure.update_yaxes(
        title_text="",
        ticksuffix="%",
        tickformat=".1f",
        range=[y_bottom, y_top],
        tickfont={"size": 13, "color": "#B8B0A7"},
        gridcolor="#2A2A2A",
        zerolinecolor="#2A2A2A",
    )
    return figure


def calculate_overall_business_metrics(
    payouts_df: pd.DataFrame,
    offline_df: pd.DataFrame,
    online_orders_df: pd.DataFrame | None = None,
) -> dict:
    payout_gross_column = find_first_column(
        payouts_df,
        ["gross_sale", "gross_sales", "gross_revenue", "gross"],
    )
    offline_total_column = find_first_column(offline_df, ["total"])

    online_monthly = monthly_revenue_frame(payouts_df, payout_gross_column)
    offline_monthly = monthly_revenue_frame(offline_df, offline_total_column)
    online_monthly = online_monthly.rename(columns={"revenue": "online_gross_sales"})
    offline_monthly = offline_monthly.rename(columns={"revenue": "offline_total"})

    monthly_total = pd.DataFrame(columns=["month_date", "revenue"])
    if not online_monthly.empty or not offline_monthly.empty:
        month_date_parts = [
            frame["month_date"]
            for frame in (online_monthly, offline_monthly)
            if not frame.empty and "month_date" in frame.columns
        ]
        monthly_total = pd.DataFrame(
            {
                "month_date": pd.concat(month_date_parts, ignore_index=True)
                .dropna()
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True),
            }
        )
        if not monthly_total.empty:
            monthly_total = monthly_total.merge(online_monthly, on="month_date", how="left")
            monthly_total = monthly_total.merge(offline_monthly, on="month_date", how="left")
            monthly_total["online_gross_sales"] = pd.to_numeric(
                monthly_total["online_gross_sales"],
                errors="coerce",
            ).fillna(0)
            monthly_total["offline_total"] = pd.to_numeric(
                monthly_total["offline_total"],
                errors="coerce",
            ).fillna(0)
            monthly_total["revenue"] = (
                monthly_total["online_gross_sales"] + monthly_total["offline_total"]
            )
            monthly_total = monthly_total.sort_values("month_date").reset_index(drop=True)

    online_gross_by_month = (
        online_monthly.sort_values("month_date").reset_index(drop=True)
        if not online_monthly.empty
        else pd.DataFrame(columns=["month_date", "online_gross_sales"])
    )
    offline_total_by_month = (
        offline_monthly.sort_values("month_date").reset_index(drop=True)
        if not offline_monthly.empty
        else pd.DataFrame(columns=["month_date", "offline_total"])
    )
    monthly_total_gross = (
        monthly_total[["month_date", "revenue"]]
        .rename(columns={"revenue": "monthly_total_gross"})
        .sort_values("month_date")
        .reset_index(drop=True)
        if not monthly_total.empty
        else pd.DataFrame(columns=["month_date", "monthly_total_gross"])
    )

    overall_gross_sales = (
        float(monthly_total["revenue"].sum()) if not monthly_total.empty else 0.0
    )
    month_count = int(monthly_total["month_date"].nunique()) if not monthly_total.empty else 0
    average_monthly_sales = overall_gross_sales / month_count if month_count else 0.0
    total_orders = count_overall_orders(
        online_orders_df if online_orders_df is not None else pd.DataFrame()
    )
    average_order_value = overall_gross_sales / total_orders if total_orders else 0.0
    best_month_label, best_month_gross_sales = best_overall_gross_month(monthly_total_gross)

    average_monthly_growth_pct: float | None = None
    if len(monthly_total.index) >= 2:
        previous_revenue = float(monthly_total["revenue"].iloc[-2])
        latest_revenue = float(monthly_total["revenue"].iloc[-1])
        if previous_revenue > 0:
            average_monthly_growth_pct = (
                (latest_revenue - previous_revenue) / previous_revenue * 100.0
            )

    return {
        "online_gross_by_month": online_gross_by_month,
        "offline_total_by_month": offline_total_by_month,
        "monthly_total_gross": monthly_total_gross,
        "overall_gross_sales": overall_gross_sales,
        "average_monthly_sales": average_monthly_sales,
        "average_monthly_growth_pct": average_monthly_growth_pct,
        "average_order_value": average_order_value,
        "total_orders": total_orders,
        "best_month_label": best_month_label,
        "best_month_gross_sales": best_month_gross_sales,
    }


def format_signed_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def render_overall_business_analysis(
    payouts_df: pd.DataFrame,
    offline_df: pd.DataFrame,
    online_orders_df: pd.DataFrame,
) -> None:
    metrics = calculate_overall_business_metrics(payouts_df, offline_df, online_orders_df)
    st.markdown(
        '<div class="online-section-heading">OVERALL BUSINESS ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="medium")
    with columns[0]:
        render_summary_card(
            "Overall Gross Sales",
            format_euro(metrics["overall_gross_sales"]),
            "Selected Period",
        )
    with columns[1]:
        render_summary_card(
            "Average Monthly Sales",
            format_euro(metrics["average_monthly_sales"]),
            "Per Month",
        )
    with columns[2]:
        render_summary_card(
            "Average Monthly Growth %",
            format_signed_percent(metrics["average_monthly_growth_pct"]),
            "Month-over-Month",
        )
    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
    second_row_columns = st.columns(3, gap="medium")
    with second_row_columns[0]:
        render_summary_card(
            "Average Order Value",
            "Insufficient Data",
            "Offline order-level data required",
        )
    with second_row_columns[1]:
        render_summary_card(
            "Total Orders",
            "Insufficient Data",
            "Offline order-level data required",
        )
    with second_row_columns[2]:
        render_summary_card_with_secondary(
            "Best Performing Month",
            metrics["best_month_label"],
            format_euro(metrics["best_month_gross_sales"]),
            "Highest Gross Sales Month",
        )
    st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
    split_metrics = build_overall_split_metrics(metrics, offline_df)
    split_columns = st.columns(3, gap="medium")
    with split_columns[0]:
        render_overall_split_card("Gross Sale Split", split_metrics["gross_sale"])
    with split_columns[1]:
        render_overall_split_card("Total Orders Split", split_metrics["orders"])
    with split_columns[2]:
        render_overall_split_card("Average Order Value Split", split_metrics["aov"])
    st.markdown('<div style="height: 34px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="overall-sales-breakdown-heading">Sales Breakdown</div>', unsafe_allow_html=True)
    monthly_channel_frame = build_overall_monthly_channel_frame(metrics)
    breakdown_columns = st.columns(2, gap="medium")
    with breakdown_columns[0]:
        with st.container(key="overall_sales_breakdown_total_card"):
            st.plotly_chart(
                build_monthly_total_gross_sale_chart(monthly_channel_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with breakdown_columns[1]:
        with st.container(key="overall_sales_breakdown_trend_card"):
            st.plotly_chart(
                build_online_offline_trend_chart(monthly_channel_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
    mom_columns = st.columns(2, gap="medium")
    with mom_columns[0]:
        with st.container(key="overall_sales_mom_change_card"):
            control_columns = st.columns([0.68, 0.32], gap="small", vertical_alignment="top")
            with control_columns[1]:
                mom_channel = st.selectbox(
                    "Channel",
                    ["Combined", "Online", "Offline"],
                    key="overall_mom_change_channel",
                )
            mom_frame = build_month_on_month_change_frame(monthly_channel_frame, mom_channel)
            st.plotly_chart(
                build_month_on_month_change_chart(mom_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    st.markdown('<div style="height: 34px;"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="overall-quarterly-growth-header">
            <div class="overall-quarterly-growth-title">Quarterly Business Growth</div>
            <div class="overall-quarterly-growth-subtitle">% growth vs previous quarter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quarterly_frame = build_quarterly_growth_frame(monthly_channel_frame)
    quarterly_columns = st.columns(2, gap="medium")
    with quarterly_columns[0]:
        with st.container(key="overall_quarterly_revenue_card"):
            st.plotly_chart(
                build_quarterly_revenue_chart(quarterly_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with quarterly_columns[1]:
        with st.container(key="overall_quarterly_growth_card"):
            st.plotly_chart(
                build_quarterly_growth_chart(quarterly_frame),
                use_container_width=True,
                config={"displayModeBar": False},
            )


def render_overall_analysis(data: dict) -> None:
    online_df = prepare_dynamic_reporting_frame(data.get("online", pd.DataFrame()))
    payouts_df = prepare_overall_payout_frame(data.get("payouts", pd.DataFrame()))
    offline_df = prepare_overall_offline_frame(
        data.get("offline", pd.DataFrame()),
        infer_single_reporting_year(payouts_df),
    )

    frames_for_years = [frame for frame in (payouts_df, offline_df) if not frame.empty]
    year_source = (
        pd.concat(frames_for_years, ignore_index=True, sort=False)
        if frames_for_years
        else pd.DataFrame()
    )
    year_options = year_filter_options(year_source)
    quarter_options = ["All Quarters", "Q1", "Q2", "Q3", "Q4"]
    channel_options = ["Both", "Online", "Offline"]

    ensure_selectbox_state("overall_year_filter", year_options)
    ensure_selectbox_state("overall_quarter_filter", quarter_options)
    ensure_selectbox_state("overall_channel_filter", channel_options)
    if st.session_state.get("_overall_business_source_version") != OVERALL_BUSINESS_SOURCE_VERSION:
        if "All Years" in year_options:
            st.session_state["overall_year_filter"] = "All Years"
        st.session_state["overall_quarter_filter"] = "All Quarters"
        st.session_state["overall_channel_filter"] = "Both"
    active_channel = str(st.session_state.get("overall_channel_filter", "Both"))
    month_options = build_overall_month_options(payouts_df, offline_df, active_channel)
    ensure_selectbox_state("overall_month_filter", month_options)
    if st.session_state.get("_overall_business_source_version") != OVERALL_BUSINESS_SOURCE_VERSION:
        if "All Months" in month_options:
            st.session_state["overall_month_filter"] = "All Months"
        st.session_state["_overall_business_source_version"] = OVERALL_BUSINESS_SOURCE_VERSION

    st.markdown(
        """
        <style>
            .st-key-overall_title_filters div[data-testid="stSelectbox"] {
                min-width: 140px;
            }

            .st-key-overall_title_filters div[data-testid="stSelectbox"] label {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .st-key-overall_title_filters div[data-baseweb="select"] > div {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 12px;
                min-height: 48px;
                padding-left: 0.15rem;
                padding-right: 0.15rem;
            }

            .st-key-overall_title_filters div[data-baseweb="select"] span {
                color: #FFFFFF !important;
                font-size: 0.96rem;
                font-weight: 500;
            }

            .online-section-heading {
                color: #FFFFFF;
                font-size: 0.95rem;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin: 0 0 0.95rem;
                text-transform: uppercase;
            }

            .online-summary-card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                box-shadow: none !important;
                height: 150px;
                padding: 22px 28px 20px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .online-card-label {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin-bottom: 14px;
                text-transform: uppercase;
            }

            .online-card-value {
                color: #FFFFFF;
                font-size: clamp(2.25rem, 2.4vw, 2.5rem);
                font-weight: 700;
                line-height: 1.04;
                margin-bottom: 12px;
                overflow-wrap: anywhere;
            }

            .online-card-secondary-value {
                color: #9A9A9A;
                font-size: 14px;
                font-weight: 600;
                line-height: 1.1;
                margin: -4px 0 8px;
            }

            .online-card-subtext {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
            }

            .overall-info-card {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                box-shadow: none;
                height: 150px;
                padding: 22px 28px 20px;
            }

            .overall-info-card-tall {
                min-height: 238px;
                height: auto;
                padding: 24px 28px;
            }

            .overall-info-title {
                color: #FFFFFF;
                font-size: 0.95rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                line-height: 1.25;
                margin-bottom: 14px;
            }

            .overall-info-message {
                color: #9A9A9A;
                font-size: 0.92rem;
                font-weight: 400;
                line-height: 1.45;
                max-width: 760px;
            }

            .overall-split-card {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                box-shadow: none;
                min-height: 238px;
                padding: 22px 26px;
            }

            .overall-split-title {
                color: #A8B0BE;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.2;
                margin-bottom: 22px;
            }

            .overall-split-content {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            .overall-split-row {
                align-items: center;
                display: grid;
                gap: 18px;
                grid-template-columns: minmax(0, 1fr) minmax(118px, 0.74fr);
            }

            .overall-split-row-header {
                align-items: baseline;
                display: flex;
                gap: 12px;
                justify-content: space-between;
                margin-bottom: 12px;
            }

            .overall-split-row-header span {
                color: #A8B0BE;
                font-size: 1.15rem;
                font-weight: 700;
            }

            .overall-split-row-header strong {
                font-size: 1.05rem;
                font-weight: 800;
                white-space: nowrap;
            }

            .overall-split-track {
                background: #2A2A2A;
                border-radius: 999px;
                height: 11px;
                overflow: hidden;
                width: 100%;
            }

            .overall-split-fill {
                border-radius: 999px;
                height: 100%;
            }

            .overall-split-fill.offline {
                background: #2ECC71;
            }

            .overall-split-fill.online {
                background: #4A9EFF;
            }

            .overall-split-row-header strong.offline,
            .overall-split-value-box.offline strong {
                color: #2ECC71;
            }

            .overall-split-row-header strong.online,
            .overall-split-value-box.online strong {
                color: #4A9EFF;
            }

            .overall-split-value-box {
                background: #161F2C;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                min-height: 76px;
                padding: 16px 18px;
            }

            .overall-split-value-box.offline {
                background: #162821;
                border-color: rgba(46, 204, 113, 0.22);
            }

            .overall-split-value-box.online {
                background: #152238;
                border-color: rgba(74, 158, 255, 0.24);
            }

            .overall-split-value-box span {
                color: #9A9A9A;
                display: block;
                font-size: 0.82rem;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .overall-split-value-box strong {
                display: block;
                font-size: 1rem;
                font-weight: 800;
                line-height: 1.18;
                overflow-wrap: anywhere;
            }

            .overall-sales-breakdown-heading {
                color: #FFFFFF;
                font-size: 1.55rem;
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.15;
                margin: 0 0 1.1rem;
            }

            .overall-quarterly-growth-header {
                margin: 0 0 1.1rem;
            }

            .overall-quarterly-growth-title {
                color: #FFFFFF;
                font-size: 1.55rem;
                font-weight: 800;
                letter-spacing: 0;
                line-height: 1.15;
                margin: 0 0 0.55rem;
            }

            .overall-quarterly-growth-subtitle {
                color: #9A9A9A;
                font-size: 1rem;
                font-weight: 700;
                line-height: 1.25;
            }

            .st-key-overall_sales_breakdown_total_card,
            .st-key-overall_sales_breakdown_trend_card,
            .st-key-overall_sales_mom_change_card,
            .st-key-overall_quarterly_revenue_card,
            .st-key-overall_quarterly_growth_card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                box-sizing: border-box;
                height: 560px;
                min-height: 560px;
                padding: 16px 16px 10px;
            }

            .st-key-overall_sales_mom_change_card [data-testid="stHorizontalBlock"] {
                margin-bottom: -0.35rem;
                position: relative;
                z-index: 2;
            }

            .st-key-overall_sales_mom_change_card [data-testid="stSelectbox"] label {
                color: #9A9A9A;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.12rem;
            }

            .st-key-overall_sales_mom_change_card [data-baseweb="select"] > div {
                background: #111111;
                border: 1px solid #2A2A2A;
                border-radius: 8px;
                min-height: 38px;
            }

            @media (max-width: 1100px) {
                .overall-split-row {
                    grid-template-columns: 1fr;
                }

                .st-key-overall_sales_breakdown_total_card,
                .st-key-overall_sales_breakdown_trend_card,
                .st-key-overall_sales_mom_change_card,
                .st-key-overall_quarterly_revenue_card,
                .st-key-overall_quarterly_growth_card {
                    min-height: 520px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="overall_title_filters"):
        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        title_column, filters_column = st.columns([1.45, 1.75], gap="medium", vertical_alignment="bottom")
        with title_column:
            st.header("IBC FY 2026-2027 Overall Performance")
        with filters_column:
            year_column, quarter_column, channel_column, month_column = st.columns(
                [1, 1, 1, 1],
                gap="small",
                vertical_alignment="bottom",
            )
            with year_column:
                selected_year = st.selectbox(
                    "Year",
                    year_options,
                    key="overall_year_filter",
                )
            with quarter_column:
                selected_quarter = st.selectbox(
                    "Quarter",
                    quarter_options,
                    key="overall_quarter_filter",
                )
            with channel_column:
                selected_channel = st.selectbox(
                    "Channel",
                    channel_options,
                    key="overall_channel_filter",
                )
            with month_column:
                selected_month = st.selectbox(
                    "Month",
                    month_options,
                    key="overall_month_filter",
                )

    filtered_payouts = apply_overall_time_filters(
        payouts_df,
        selected_year,
        selected_quarter,
        selected_month,
    )
    filtered_offline = apply_overall_time_filters(
        offline_df,
        selected_year,
        selected_quarter,
        selected_month,
    )
    filtered_online = apply_overall_time_filters(
        online_df,
        selected_year,
        selected_quarter,
        selected_month,
    )

    if selected_channel == "Online":
        filtered_offline = filtered_offline.iloc[0:0].copy()
    elif selected_channel == "Offline":
        filtered_payouts = filtered_payouts.iloc[0:0].copy()
        filtered_online = filtered_online.iloc[0:0].copy()

    render_overall_business_analysis(filtered_payouts, filtered_offline, filtered_online)

    combined_frames = [frame for frame in (filtered_online, filtered_offline) if not frame.empty]
    combined_filtered = (
        pd.concat(combined_frames, ignore_index=True, sort=False) if combined_frames else pd.DataFrame()
    )

    st.session_state["overall_filters"] = {
        "year": selected_year,
        "quarter": selected_quarter,
        "channel": selected_channel,
        "month": selected_month,
    }
    st.session_state["overall_filtered_data"] = {
        "online": filtered_online,
        "payouts": filtered_payouts,
        "offline": filtered_offline,
        "combined": combined_filtered,
    }


def render_online_analysis(data: dict) -> None:
    online_df = prepare_dynamic_online_data(data.get("online", pd.DataFrame()))
    product_analysis_df = prepare_dynamic_product_data(
        data.get("product_analysis", pd.DataFrame())
    )
    payouts_df = data.get("payouts", pd.DataFrame())
    partner_column = find_first_column(online_df, ["partner", "patner"])
    year_column = usable_column(online_df, FILTER_YEAR_COLUMN) or find_first_column(
        online_df,
        ["fy", "year"],
    )
    month_column = usable_column(online_df, FILTER_MONTH_COLUMN) or find_first_column(
        online_df,
        ["month"],
    )
    partner_options = ["All Partners", *VALID_PARTNERS]
    year_options = year_filter_options(online_df)
    month_options = month_filter_options(online_df)
    ensure_selectbox_state("online_partner_filter", partner_options)
    ensure_selectbox_state("online_year_filter", year_options)
    ensure_selectbox_state("online_month_filter", month_options)

    st.markdown(
        """
        <style>
            html,
            body,
            .stApp {
                min-height: 100%;
                height: auto !important;
                overflow-y: auto !important;
                font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
            }

            html,
            body,
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            .main,
            .block-container {
                background: #0E0E0E !important;
                color: #FFFFFF !important;
            }

            [data-testid="stAppViewContainer"] {
                min-height: 100vh;
                height: auto !important;
                overflow-y: auto !important;
            }

            [data-testid="stHeader"] {
                background: transparent !important;
            }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                background: transparent !important;
            }

            .block-container {
                padding-top: 0.55rem;
                padding-left: 4.7rem;
                padding-right: 4.7rem;
                padding-bottom: 80px !important;
                max-width: 1920px;
                height: auto !important;
                min-height: 100vh;
                overflow: visible !important;
            }

            .main {
                height: auto !important;
                min-height: 100vh;
                overflow: visible !important;
            }

            .online-title {
                color: #FFFFFF;
                font-size: clamp(2.55rem, 3.05vw, 3.9rem);
                font-weight: 700;
                line-height: 1;
                letter-spacing: 0;
                margin: 1.75rem 0 3.8rem;
                white-space: nowrap;
            }

            .online-section-heading {
                color: #FFFFFF;
                font-size: 0.95rem;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin: 0 0 0.95rem;
                text-transform: uppercase;
            }

            .online-summary-card,
            .partner-card,
            .detailed-partner-card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                box-shadow: none !important;
            }

            .st-key-revenue_chart_left,
            .st-key-revenue_chart_right,
            .st-key-revenue_chart_monthly_left,
            .st-key-revenue_chart_monthly_right,
            .st-key-revenue_chart_deduction_left,
            .st-key-revenue_chart_deduction_right,
            .st-key-revenue_chart_deduction_trends,
            .st-key-order_share_partner_card,
            .st-key-top_ordered_products_card,
            .st-key-grain_weekday_order_revenue_card,
            .st-key-grain_hourly_revenue_heatmap_card,
            .st-key-grain_ticket_size_distribution_card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                min-height: 500px;
                padding: 16px 16px 10px;
            }

            .st-key-revenue_chart_deduction_left,
            .st-key-revenue_chart_deduction_trends {
                box-sizing: border-box;
                height: 540px;
                min-height: 540px;
                padding: 16px 16px 10px;
            }

            .st-key-order_share_partner_card,
            .st-key-top_ordered_products_card {
                box-sizing: border-box;
                height: 560px;
                min-height: 560px;
                padding: 16px 16px 10px;
            }

            .online-chart-title {
                color: #FFFFFF;
                font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 1.375rem;
                font-weight: 700;
                line-height: 1.2;
                margin: 0.34rem 0 0.52rem;
            }

            .order-share-card-subtitle {
                color: #CFC7BD;
                font-size: 1.06rem;
                font-weight: 500;
                margin: 0 0 0.82rem;
            }

            .ticket-size-card-subtitle {
                color: #9A9A9A;
                font-size: 0.98rem;
                font-weight: 500;
                line-height: 1.35;
                margin: 0 0 1rem;
            }

            .ticket-size-bucket-grid {
                display: grid;
                gap: 0.78rem;
                grid-template-columns: repeat(6, minmax(0, 1fr));
                width: 100%;
            }

            .ticket-size-bucket-card {
                background: #151515;
                border: 1px solid #242424;
                border-left: 5px solid #5B9DFF;
                border-radius: 10px;
                box-sizing: border-box;
                min-height: 128px;
                padding: 13px 13px 12px;
            }

            .ticket-size-bucket-card:nth-child(1) {
                grid-column: 1 / span 2;
            }

            .ticket-size-bucket-card:nth-child(2) {
                grid-column: 3 / span 2;
            }

            .ticket-size-bucket-card:nth-child(3) {
                grid-column: 5 / span 2;
            }

            .ticket-size-bucket-card:nth-child(4) {
                grid-column: 2 / span 2;
            }

            .ticket-size-bucket-card:nth-child(5) {
                grid-column: 4 / span 2;
            }

            .ticket-size-bucket {
                color: #CFC7BD;
                font-size: 0.96rem;
                font-weight: 700;
                margin-bottom: 0.55rem;
            }

            .ticket-size-orders {
                color: #FFFFFF;
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.72rem;
                white-space: nowrap;
            }

            .ticket-size-progress-track {
                background: #2A2A2A;
                border-radius: 999px;
                height: 8px;
                overflow: hidden;
                margin-bottom: 7px;
            }

            .ticket-size-progress-fill {
                border-radius: 999px;
                height: 100%;
            }

            .ticket-size-percent {
                color: #9A9A9A;
                font-size: 0.82rem;
                font-weight: 500;
                line-height: 1.25;
            }

            .ticket-size-empty {
                align-items: center;
                color: #9A9A9A;
                display: flex;
                font-size: 14px;
                justify-content: center;
                min-height: 320px;
                text-align: center;
            }

            .st-key-order_share_partner_card div[data-testid="stHorizontalBlock"] {
                align-items: stretch;
            }

            .order-share-breakdown-card {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                margin-bottom: 0.9rem;
                padding: 14px 16px 13px;
            }

            .order-share-breakdown-head {
                align-items: center;
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }

            .order-share-breakdown-partner {
                align-items: center;
                display: flex;
                gap: 10px;
            }

            .order-share-color-chip {
                border-radius: 6px;
                display: inline-block;
                height: 16px;
                width: 16px;
            }

            .order-share-partner-name {
                color: #F4EEE7;
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.2;
            }

            .order-share-partner-percent {
                font-size: 1.95rem;
                font-weight: 700;
                line-height: 1;
            }

            .order-share-partner-meta {
                color: #CFC7BD;
                font-size: 0.95rem;
                font-weight: 500;
                margin-bottom: 11px;
            }

            .order-share-progress-track {
                background: #2A2A2A;
                border-radius: 999px;
                height: 10px;
                overflow: hidden;
            }

            .order-share-progress-fill {
                border-radius: 999px;
                height: 100%;
            }

            .top-ordered-filter-label {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                line-height: 1;
                margin: 0.2rem 0 0.28rem;
                text-align: right;
                text-transform: uppercase;
            }

            .st-key-top_ordered_products_card div[data-testid="stHorizontalBlock"] {
                align-items: flex-start;
                margin-bottom: 0.08rem;
            }

            .st-key-top_ordered_products_card div[data-testid="stSelectbox"] {
                margin-left: auto;
                max-width: 96px;
            }

            .st-key-top_ordered_products_card div[data-testid="stSelectbox"] > div {
                min-width: 96px;
            }

            .st-key-top_ordered_products_card div[data-testid="stSelectbox"] label {
                margin-bottom: 0.2rem;
            }

            .st-key-top_ordered_products_card div[data-baseweb="select"] > div {
                border-radius: 10px;
                min-height: 38px;
                padding-top: 0;
                padding-bottom: 0;
            }

            .st-key-top_ordered_products_card div[data-baseweb="select"] span {
                font-size: 0.9rem;
                font-weight: 600;
            }

            .st-key-top_ordered_products_card div[data-testid="stPlotlyChart"] {
                margin-top: -0.45rem;
            }

            .st-key-revenue_chart_deduction_trends div[data-testid="stRadio"] {
                margin-left: auto;
                max-width: 320px;
            }

            .st-key-revenue_chart_deduction_trends div[data-testid="stRadio"] label {
                color: #9A9A9A;
            }

            .st-key-revenue_chart_deduction_trends div[data-testid="stSegmentedControl"] {
                display: flex;
                justify-content: flex-end;
                margin-bottom: -0.35rem;
            }

            .st-key-revenue_breakdown_section div[data-testid="stHorizontalBlock"] {
                align-items: stretch;
            }

            .online-summary-card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                height: 150px;
                padding: 22px 28px 20px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .grain-kpi-card {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A;
                border-radius: 12px;
                box-shadow: none !important;
                box-sizing: border-box;
                height: 176px;
                padding: 22px 28px 20px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .online-card-label {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin-bottom: 14px;
                text-transform: uppercase;
            }

            .online-card-value {
                color: #FFFFFF;
                font-size: clamp(2.25rem, 2.4vw, 2.5rem);
                font-weight: 700;
                line-height: 1.04;
                margin-bottom: 12px;
                overflow-wrap: anywhere;
            }

            .online-card-subtext {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
            }

            .partner-subtext {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
            }

            .grain-kpi-value {
                color: #FFFFFF;
                font-size: clamp(2.05rem, 2.15vw, 2.36rem);
                font-weight: 700;
                line-height: 1.04;
                margin-bottom: 12px;
                overflow-wrap: anywhere;
            }

            .grain-kpi-detail {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
                line-height: 1.45;
            }

            .partner-card {
                border-radius: 12px;
                border-left: 6px solid #7A7A7A;
                min-height: 176px;
                margin-bottom: 1.05rem;
                padding: 25px 28px 24px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }

            .detailed-partner-card {
                border-radius: 12px;
                box-sizing: border-box;
                height: 500px;
                margin-bottom: 1.2rem;
                padding: 30px 32px 28px;
            }

            .detailed-card-header {
                align-items: center;
                border-bottom: 1px solid #2A2A2A;
                display: flex;
                gap: 20px;
                padding-bottom: 22px;
                margin-bottom: 0;
            }

            .partner-avatar {
                align-items: center;
                border-radius: 999px;
                display: flex;
                flex: 0 0 58px;
                font-size: 0.95rem;
                font-weight: 850;
                height: 58px;
                justify-content: center;
                letter-spacing: 0.02em;
                width: 58px;
            }

            .detailed-partner-name {
                color: #FFFFFF;
                font-size: 1.32rem;
                font-weight: 700;
                line-height: 1.12;
                margin-bottom: 6px;
            }

            .detailed-date-range {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
            }

            .detailed-metrics {
                padding-top: 10px;
                padding-bottom: 12px;
            }

            .detailed-metric-row {
                align-items: center;
                border-bottom: 1px solid #2A2A2A;
                display: flex;
                justify-content: space-between;
                min-height: 62px;
                gap: 18px;
                padding: 2px 0;
            }

            .detailed-metric-row:last-child {
                border-bottom: 0;
                padding-bottom: 10px;
            }

            .detailed-metric-row span {
                color: #9A9A9A;
                font-size: 13px;
                font-weight: 400;
            }

            .detailed-metric-row strong {
                color: #FFFFFF;
                font-size: 1.08rem;
                font-weight: 600;
                text-align: right;
                white-space: nowrap;
            }

            .partner-name {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 18px;
            }

            .partner-value {
                color: #FFFFFF;
                font-size: clamp(2.25rem, 2.4vw, 2.5rem);
                font-weight: 700;
                line-height: 1.05;
                margin-bottom: 16px;
                overflow-wrap: anywhere;
            }

            div[data-testid="stSelectbox"] {
                min-width: 0;
            }

            div[data-testid="stSelectbox"] label {
                color: #7A7A7A;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.05em;
                margin-bottom: 0.45rem;
                text-transform: uppercase;
            }

            div[data-baseweb="select"] > div {
                background: #1A1A1A !important;
                border: 1px solid #2A2A2A !important;
                border-radius: 12px;
                min-height: 58px;
            }

            div[data-baseweb="popover"],
            div[data-baseweb="menu"],
            ul[role="listbox"] {
                background: #1A1A1A !important;
                border-color: #2A2A2A !important;
            }

            div[data-baseweb="select"] span {
                color: #FFFFFF !important;
                font-size: 1rem;
                font-weight: 500;
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-left: 1.1rem;
                    padding-right: 1.1rem;
                }

                .online-title {
                    font-size: 2.35rem;
                    margin: 1rem 0 1.5rem;
                    white-space: normal;
                }

                .online-chart-title {
                    font-size: 1.52rem;
                    margin-top: 0.2rem;
                }

                .order-share-card-subtitle {
                    font-size: 0.98rem;
                }

                .online-summary-card,
                .partner-card,
                .detailed-partner-card {
                    border-radius: 12px;
                }

                .detailed-partner-card {
                    height: auto;
                }

                .st-key-revenue_breakdown_section div[data-testid="stHorizontalBlock"] {
                    flex-direction: column !important;
                    gap: 1rem !important;
                }

                .st-key-revenue_breakdown_section div[data-testid="column"],
                .st-key-revenue_breakdown_section div[data-testid="stColumn"] {
                    width: 100% !important;
                    min-width: 100% !important;
                    flex: 1 1 100% !important;
                }

                .st-key-revenue_chart_left,
                .st-key-revenue_chart_right,
                .st-key-revenue_chart_monthly_left,
                .st-key-revenue_chart_monthly_right,
                .st-key-revenue_chart_deduction_left,
                .st-key-revenue_chart_deduction_right,
                .st-key-revenue_chart_deduction_trends,
                .st-key-order_share_partner_card,
                .st-key-top_ordered_products_card,
                .st-key-grain_weekday_order_revenue_card,
                .st-key-grain_hourly_revenue_heatmap_card,
                .st-key-grain_ticket_size_distribution_card {
                    min-height: 460px;
                }

                .st-key-order_share_partner_card div[data-testid="stHorizontalBlock"] {
                    flex-direction: column !important;
                    gap: 0.8rem !important;
                }

            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title_column, partner_filter, year_filter, month_filter = st.columns(
        [2.08, 0.72, 0.58, 0.68],
        gap="medium",
        vertical_alignment="bottom",
    )

    with title_column:
        st.markdown(
            f'<div class="online-title">{escape(DASHBOARD_TITLE)}</div>',
            unsafe_allow_html=True,
        )

    with partner_filter:
        selected_partner = st.selectbox(
            "Partner",
            partner_options,
            key="online_partner_filter",
            format_func=lambda value: value
            if value == "All Partners"
            else partner_display_name(value),
        )

    with year_filter:
        selected_year = st.selectbox(
            "Year",
            year_options,
            key="online_year_filter",
        )

    with month_filter:
        selected_month = st.selectbox(
            "Month",
            month_options,
            key="online_month_filter",
        )

    filtered_online = apply_online_filters(
        online_df,
        partner_column,
        year_column,
        month_column,
        selected_partner,
        selected_year,
        selected_month,
    )

    st.markdown('<div class="online-section-heading">TOP KPI SUMMARY</div>', unsafe_allow_html=True)
    render_online_kpis(
        filtered_online,
        payouts_df,
        selected_partner,
        selected_year,
        selected_month,
    )
    st.markdown(
        '<div style="height: 1.45rem;"></div><div class="online-section-heading">PARTNER GROSS SNAPSHOT</div>',
        unsafe_allow_html=True,
    )
    render_partner_snapshot(filtered_online)
    st.markdown(
        '<div style="height: 2.15rem;"></div><div class="online-section-heading">DETAILED PARTNER CARDS</div>',
        unsafe_allow_html=True,
    )
    render_detailed_partner_cards(
        filtered_online,
        payouts_df,
        selected_year,
        selected_month,
    )
    with st.container(key="revenue_breakdown_section"):
        st.markdown(
            '<div style="height: 36px;"></div><div class="online-section-heading">REVENUE BREAKDOWN</div>',
            unsafe_allow_html=True,
        )
        render_revenue_breakdown(
            filtered_online,
            product_analysis_df,
            payouts_df,
            selected_partner,
            selected_year,
            selected_month,
        )



def main() -> None:
    st.set_page_config(
        page_title="Business Analysis Dashboard",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page_slug_to_name = {
        "overall-analysis": "Overall Analysis",
        "online-analysis": "Online Analysis",
    }
    page_name_to_slug = {value: key for key, value in page_slug_to_name.items()}

    if "active_report_page" not in st.session_state:
        page_slug = st.query_params.get("page")
        if isinstance(page_slug, list):
            page_slug = page_slug[0] if page_slug else None
        st.session_state["active_report_page"] = page_slug_to_name.get(
            str(page_slug),
            "Overall Analysis",
        )
    elif st.session_state["active_report_page"] not in PAGES:
        st.session_state["active_report_page"] = "Overall Analysis"

    st.sidebar.radio(
        "Navigation",
        PAGES,
        key="active_report_page",
    )
    selected_page = st.session_state["active_report_page"]
    selected_slug = page_name_to_slug.get(
        st.session_state["active_report_page"],
        "overall-analysis",
    )

    current_page_slug = st.query_params.get("page")
    if isinstance(current_page_slug, list):
        current_page_slug = current_page_slug[0] if current_page_slug else None
    if current_page_slug != selected_slug:
        st.query_params["page"] = selected_slug

    if st.sidebar.button("Refresh data"):
        clear_reporting_data_cache()
        st.rerun()

    try:
        data = load_reporting_data(REPORTING_DATA_SOURCE_VERSION)
    except Exception as exc:
        st.warning(f"Data loading failed: {exc}")
        data = empty_data_status()

    if selected_page == "Overall Analysis":
        render_overall_analysis(data)
    else:
        render_online_analysis(data)


if __name__ == "__main__":
    main()

import argparse
import csv
import re
import ssl
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
from pandas.errors import ParserError

QUARTERLY_PAGE_URL = "https://www.ncua.gov/analysis/Pages/call-report-data/quarterly-data.aspx"
FALLBACK_URL_TEMPLATE = "https://www.ncua.gov/files/publications/analysis/call-report-data-{quarter}.zip"
ZIP_LINK_PATTERN = re.compile(r'href=["\']([^"\']+\.zip)["\']', flags=re.IGNORECASE)
FS220_TABLE_PATTERN = re.compile(r"^fs220.*\.txt$", flags=re.IGNORECASE)
YEAR_MONTH_PATTERNS = (
    re.compile(r"call-report-data-(\d{4})-(\d{2})\.zip$", flags=re.IGNORECASE),
    re.compile(r"qcr(\d{4})(\d{2})\.zip$", flags=re.IGNORECASE),
    re.compile(r"5300data(\d{2})(\d{2})final\.zip$", flags=re.IGNORECASE),
    re.compile(r"(\d{4})-(\d{2})\.zip$", flags=re.IGNORECASE),
)
VALID_QUARTER_MONTHS = {"03", "06", "09", "12"}
KEY_COLUMNS = ("CU_NUMBER", "CYCLE_DATE", "JOIN_NUMBER")
FILE_CANDIDATES_ACCT_DESC = ("AcctDesc.txt", "Acct_Desc.txt", "Acct_Des.txt")
USER_AGENT = "Mozilla/5.0 (compatible; NCUA-Ingestion/2.0)"


def build_requested_quarters(start_year: int, end_year: int) -> list[str]:
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    return [
        f"{year}-{month}"
        for year in range(start_year, end_year + 1)
        for month in ("03", "06", "09", "12")
    ]


def parse_quarter_from_zip_href(href: str) -> str | None:
    file_name = Path(href.split("?")[0]).name
    for pattern in YEAR_MONTH_PATTERNS:
        match = pattern.search(file_name)
        if not match:
            continue
        if pattern.pattern.startswith("5300data"):
            month, year_two_digits = match.groups()
            year_short = int(year_two_digits)
            year = 2000 + year_short if year_short <= 30 else 1900 + year_short
        else:
            year, month = match.groups()
            year = int(year)
        if month in VALID_QUARTER_MONTHS:
            return f"{year:04d}-{month}"
    return None


def discover_quarter_urls(ssl_ctx: ssl.SSLContext) -> dict[str, str]:
    request = Request(QUARTERLY_PAGE_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(request, context=ssl_ctx, timeout=90) as response:
        html = response.read().decode("utf-8", errors="ignore")

    quarter_to_url: dict[str, str] = {}
    for href in ZIP_LINK_PATTERN.findall(html):
        quarter = parse_quarter_from_zip_href(href)
        if not quarter:
            continue
        full_url = urljoin("https://www.ncua.gov", href)
        quarter_to_url[quarter] = full_url
    return quarter_to_url


def open_file_from_zip_case_insensitive(zf: ZipFile, name_options: tuple[str, ...]):
    names_lookup = {name.lower(): name for name in zf.namelist()}
    basename_lookup = {Path(name).name.lower(): name for name in zf.namelist()}
    for option in name_options:
        actual_name = names_lookup.get(option.lower())
        if not actual_name:
            actual_name = basename_lookup.get(option.lower())
        if actual_name:
            return zf.open(actual_name)
    return None


def load_account_name_map(zf: ZipFile) -> dict[str, str]:
    acct_desc_handle = open_file_from_zip_case_insensitive(zf, FILE_CANDIDATES_ACCT_DESC)
    if acct_desc_handle is None:
        return {}

    acct_desc_df = pd.read_csv(acct_desc_handle, encoding="ISO-8859-1", dtype="string")
    acct_desc_df.columns = [str(c).strip() for c in acct_desc_df.columns]
    column_lookup = {column.upper(): column for column in acct_desc_df.columns}
    account_col = column_lookup.get("ACCOUNT")
    name_col = column_lookup.get("ACCTNAME")
    if not account_col or not name_col:
        return {}

    acct_desc_df = acct_desc_df[[account_col, name_col]].dropna(subset=[account_col])
    acct_desc_df[account_col] = acct_desc_df[account_col].str.strip().str.upper()
    acct_desc_df[name_col] = acct_desc_df[name_col].fillna("").str.strip()
    return dict(zip(acct_desc_df[account_col], acct_desc_df[name_col]))


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip().strip('"').upper() for column in df.columns}
    return df.rename(columns=renamed)


def read_ncua_csv(handle) -> pd.DataFrame:
    try:
        return pd.read_csv(handle, encoding="ISO-8859-1", low_memory=False)
    except ParserError:
        handle.seek(0)
        try:
            return pd.read_csv(
                handle,
                encoding="ISO-8859-1",
                low_memory=False,
                engine="python",
                on_bad_lines="skip",
            )
        except ParserError:
            handle.seek(0)
            return pd.read_csv(
                handle,
                encoding="ISO-8859-1",
                low_memory=False,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
            )


def mostly_numeric(series: pd.Series, threshold: float = 0.95) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    numeric_values = pd.to_numeric(non_null, errors="coerce")
    return bool(numeric_values.notna().mean() >= threshold)


def numeric_sum(series: pd.Series):
    numeric_values = pd.to_numeric(series, errors="coerce")
    if not numeric_values.notna().any():
        return pd.NA
    return numeric_values.sum(min_count=1)


def join_unique_values(series: pd.Series):
    seen: set[str] = set()
    values: list[str] = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return " | ".join(values) if values else pd.NA


def collapse_multi_record_table(df: pd.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
    value_columns = [column for column in df.columns if column not in key_columns]
    if not value_columns:
        return df.drop_duplicates(subset=list(key_columns))

    aggregation_functions = {}
    for column in value_columns:
        aggregation_functions[column] = numeric_sum if mostly_numeric(df[column]) else join_unique_values

    return (
        df.groupby(list(key_columns), as_index=False, dropna=False)
        .agg(aggregation_functions)
        .reset_index(drop=True)
    )


def merge_quarter_tables(
    quarter_tables: list[tuple[str, pd.DataFrame]],
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    merged_df: pd.DataFrame | None = None
    for table_name, table_df in quarter_tables:
        table_id = Path(table_name).stem.upper()
        if merged_df is None:
            merged_df = table_df
            continue

        overlapping_non_key = [
            column
            for column in table_df.columns
            if column in merged_df.columns and column not in key_columns
        ]
        if overlapping_non_key:
            rename_map = {column: f"{column}__{table_id}" for column in overlapping_non_key}
            table_df = table_df.rename(columns=rename_map)

        merged_df = merged_df.merge(
            table_df,
            on=list(key_columns),
            how="outer",
            sort=False,
            validate="one_to_one",
        )

    if merged_df is None:
        return pd.DataFrame(columns=list(key_columns))
    return merged_df


def format_column_name(column: str, account_name_map: dict[str, str]) -> str:
    description = account_name_map.get(column, "").strip()
    if not description:
        description = column.lower()
    return f"{column} - {description}"


def run_ingestion(
    start_year: int,
    end_year: int,
    output_file: str,
    multirecord_output_file: str,
    cleanup_temp_files: bool = True,
    resume_from_temp: bool = False,
) -> None:
    requested_quarters = build_requested_quarters(start_year=start_year, end_year=end_year)
    ssl_ctx = ssl.create_default_context()

    quarter_urls = {}
    try:
        quarter_urls = discover_quarter_urls(ssl_ctx=ssl_ctx)
        print(f"Discovered {len(quarter_urls)} quarter links from NCUA page.")
    except Exception as error:  # noqa: BLE001
        print(f"WARN: Could not discover quarter links from NCUA page ({error}).")
        print("      Falling back to default modern URL format for requested quarters.")

    quarter_temp_dir = Path(".ncua_quarter_temp")
    quarter_temp_dir.mkdir(exist_ok=True)
    if not resume_from_temp:
        for temp_csv in quarter_temp_dir.glob("quarter-*.csv"):
            temp_csv.unlink()

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    quarter_temp_files: list[tuple[str, Path]] = []
    multi_record_frames: list[pd.DataFrame] = []
    all_columns_seen: set[str] = set()
    ordered_columns: list[str] = []
    global_account_name_map: dict[str, str] = {}

    for quarter in requested_quarters:
        quarter_temp_path = quarter_temp_dir / f"quarter-{quarter}.csv"
        if resume_from_temp and quarter_temp_path.exists():
            header_columns = pd.read_csv(quarter_temp_path, nrows=0).columns.tolist()
            for column in header_columns:
                if column not in all_columns_seen:
                    all_columns_seen.add(column)
                    ordered_columns.append(column)
            quarter_temp_files.append((quarter, quarter_temp_path))
            successful.append(quarter)
            print(f"RESUME: {quarter} from existing temp file.")
            continue

        url = quarter_urls.get(quarter, FALLBACK_URL_TEMPLATE.format(quarter=quarter))
        request = Request(url, headers={"User-Agent": USER_AGENT})

        try:
            with urlopen(request, context=ssl_ctx, timeout=120) as response:
                payload = response.read()

            with ZipFile(BytesIO(payload)) as zf:
                account_map = load_account_name_map(zf)
                for account_code, account_name in account_map.items():
                    if account_name:
                        global_account_name_map[account_code] = account_name

                schedule_files = sorted(
                    [
                        name
                        for name in zf.namelist()
                        if FS220_TABLE_PATTERN.match(Path(name).name)
                    ],
                    key=str.lower,
                )
                if not schedule_files:
                    raise FileNotFoundError("No FS220 schedule files found in quarter zip.")

                quarter_tables: list[tuple[str, pd.DataFrame]] = []
                for schedule_file in schedule_files:
                    with zf.open(schedule_file) as handle:
                        table_df = read_ncua_csv(handle)
                    table_df = normalize_dataframe_columns(table_df)

                    missing_keys = [key for key in KEY_COLUMNS if key not in table_df.columns]
                    if missing_keys:
                        print(f"SKIP TABLE: {quarter} {schedule_file} missing keys {missing_keys}")
                        continue

                    duplicate_rows = table_df.duplicated(subset=list(KEY_COLUMNS)).sum()
                    if duplicate_rows:
                        raw_multi_df = table_df.copy()
                        raw_multi_df["SOURCE_TABLE"] = Path(schedule_file).stem.upper()
                        raw_multi_df["SOURCE_QUARTER"] = quarter
                        multi_record_frames.append(raw_multi_df)

                        table_df = collapse_multi_record_table(table_df, key_columns=KEY_COLUMNS)
                        print(
                            f"COLLAPSED TABLE: {quarter} {schedule_file} "
                            f"{duplicate_rows} duplicate-key rows aggregated."
                        )

                    quarter_tables.append((schedule_file, table_df))

                if not quarter_tables:
                    raise ValueError("No mergeable FS220 tables were found in this quarter.")

                merged_quarter_df = merge_quarter_tables(quarter_tables, key_columns=KEY_COLUMNS)

                for column in merged_quarter_df.columns:
                    if column not in all_columns_seen:
                        all_columns_seen.add(column)
                        ordered_columns.append(column)

                merged_quarter_df.to_csv(quarter_temp_path, index=False)
                quarter_temp_files.append((quarter, quarter_temp_path))
                successful.append(quarter)
                print(
                    f"OK: {quarter} from {url} -> "
                    f"{merged_quarter_df.shape[0]:,} rows x {merged_quarter_df.shape[1]:,} cols"
                )

        except (HTTPError, URLError) as error:
            failed.append((quarter, f"not available: {error}"))
            print(f"SKIP: {quarter} -> not available ({error})")
        except Exception as error:  # noqa: BLE001
            failed.append((quarter, f"{type(error).__name__}: {error}"))
            print(f"ERROR: {quarter} -> {type(error).__name__}: {error}")

    if not quarter_temp_files:
        print("No data was retrieved for the requested quarter range.")
        return

    final_columns = [column for column in KEY_COLUMNS if column in all_columns_seen] + [
        column for column in ordered_columns if column not in KEY_COLUMNS
    ]
    final_headers = [format_column_name(column, global_account_name_map) for column in final_columns]

    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    for index, (_, quarter_temp_path) in enumerate(quarter_temp_files):
        quarter_df = pd.read_csv(quarter_temp_path, low_memory=False)
        quarter_df = quarter_df.reindex(columns=final_columns)
        quarter_df.columns = final_headers
        total_rows += len(quarter_df)
        quarter_df.to_csv(
            output_path,
            mode="w" if index == 0 else "a",
            header=(index == 0),
            index=False,
        )

    if multi_record_frames:
        multi_df = pd.concat(multi_record_frames, axis=0, sort=False)
        ordered_multi = [
            column
            for column in (*KEY_COLUMNS, "SOURCE_TABLE", "SOURCE_QUARTER")
            if column in multi_df.columns
        ] + [
            column
            for column in multi_df.columns
            if column not in (*KEY_COLUMNS, "SOURCE_TABLE", "SOURCE_QUARTER")
        ]
        multi_df = multi_df.reindex(columns=ordered_multi)
        multi_headers = [
            format_column_name(column, global_account_name_map)
            if column not in ("SOURCE_TABLE", "SOURCE_QUARTER")
            else column
            for column in multi_df.columns
        ]
        multi_df.columns = multi_headers
        multi_df.to_csv(multirecord_output_file, index=False)
        print(
            f"Saved multi-record raw rows ({len(multi_df):,}) to {multirecord_output_file}."
        )

    if cleanup_temp_files:
        for _, quarter_temp_path in quarter_temp_files:
            if quarter_temp_path.exists():
                quarter_temp_path.unlink()
        if quarter_temp_dir.exists() and not any(quarter_temp_dir.iterdir()):
            quarter_temp_dir.rmdir()

    print()
    print(f"Done. Retrieved {len(successful)}/{len(requested_quarters)} requested quarters.")
    print(f"Saved {total_rows:,} rows x {len(final_headers):,} columns to {output_file}.")
    if failed:
        print("Missing/failed quarters:")
        for quarter, reason in failed:
            print(f"  - {quarter}: {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NCUA quarterly call report data and build a maximally wide "
            "FS220 dataset by merging all available FS220* schedules."
        )
    )
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-file", default="NCUA_Call_Report.csv")
    parser.add_argument(
        "--multirecord-output-file",
        default="NCUA_Call_Report_Multirecord.csv",
        help=(
            "Output for raw rows from FS220 schedules that contain duplicate keys "
            "(for example FS220CUSO in older years)."
        ),
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary per-quarter CSV files used to assemble the final output.",
    )
    parser.add_argument(
        "--resume-from-temp",
        action="store_true",
        help=(
            "Reuse existing .ncua_quarter_temp/quarter-YYYY-MM.csv files and only "
            "download/process missing quarters before producing final outputs."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_ingestion(
        start_year=cli_args.start_year,
        end_year=cli_args.end_year,
        output_file=cli_args.output_file,
        multirecord_output_file=cli_args.multirecord_output_file,
        cleanup_temp_files=not cli_args.keep_temp_files,
        resume_from_temp=cli_args.resume_from_temp,
    )

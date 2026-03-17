"""
Download NCUA quarterly call‑report data and build a maximally wide
FS220 dataset by merging all available FS220* schedules.

Changes from original
---------------------
- HTTP retry with exponential backoff (fetch_with_retry)
- Parallel ZIP downloads via ThreadPoolExecutor
- Streaming multirecord CSV writes (no in‑memory accumulation)
- raw_column_names flag to skip description formatting (for Delta tables)
- Cached mostly_numeric results across quarters
- return_account_map option to expose the account‑name mapping to callers
"""

import argparse
import csv
import gc
import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd
from pandas.errors import ParserError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUARTERLY_PAGE_URL = (
    "https://www.ncua.gov/analysis/Pages/call-report-data/quarterly-data.aspx"
)
FALLBACK_URL_TEMPLATE = (
    "https://www.ncua.gov/files/publications/analysis/call-report-data-{quarter}.zip"
)
ZIP_LINK_PATTERN = re.compile(
    r'href=["\']([^"\']+\.zip)["\']', flags=re.IGNORECASE
)
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
USER_AGENT = "Mozilla/5.0 (compatible; NCUA-Ingestion/2.1)"

# Retry / concurrency defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_DOWNLOAD_WORKERS = 4


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def fetch_with_retry(
    url: str,
    ssl_ctx: ssl.SSLContext,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """Download *url* with exponential back‑off on transient errors."""
    for attempt in range(max_retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, context=ssl_ctx, timeout=120) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt * 5  # 5 s, 10 s, 20 s
            print(
                f"  Retry {attempt + 1}/{max_retries} for {url} "
                f"after {wait}s ({exc})"
            )
            time.sleep(wait)
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("fetch_with_retry exhausted retries")


def download_quarter(
    quarter: str, url: str, ssl_ctx: ssl.SSLContext
) -> tuple[str, bytes | None, str]:
    """Download a single quarter ZIP. Returns (quarter, payload | None, error)."""
    try:
        payload = fetch_with_retry(url, ssl_ctx)
        return (quarter, payload, "")
    except Exception as exc:  # noqa: BLE001
        return (quarter, None, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# ZIP / CSV helpers
# ---------------------------------------------------------------------------
def open_file_from_zip_case_insensitive(
    zf: ZipFile, name_options: tuple[str, ...]
):
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
    acct_desc_handle = open_file_from_zip_case_insensitive(
        zf, FILE_CANDIDATES_ACCT_DESC
    )
    if acct_desc_handle is None:
        return {}

    acct_desc_df = pd.read_csv(
        acct_desc_handle, encoding="ISO-8859-1", dtype="string"
    )
    acct_desc_df.columns = [str(c).strip() for c in acct_desc_df.columns]
    column_lookup = {column.upper(): column for column in acct_desc_df.columns}
    account_col = column_lookup.get("ACCOUNT")
    name_col = column_lookup.get("ACCTNAME")
    if not account_col or not name_col:
        return {}

    acct_desc_df = acct_desc_df[[account_col, name_col]].dropna(
        subset=[account_col]
    )
    acct_desc_df[account_col] = (
        acct_desc_df[account_col].str.strip().str.upper()
    )
    acct_desc_df[name_col] = acct_desc_df[name_col].fillna("").str.strip()
    return dict(zip(acct_desc_df[account_col], acct_desc_df[name_col]))


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: str(column).strip().strip('"').upper() for column in df.columns
    }
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


# ---------------------------------------------------------------------------
# Aggregation helpers  (with cached numeric detection)
# ---------------------------------------------------------------------------
_numeric_cache: dict[str, bool] = {}


def mostly_numeric(
    series: pd.Series,
    column_name: str | None = None,
    threshold: float = 0.95,
) -> bool:
    """Check whether *series* is predominantly numeric.

    Results are cached by *column_name* so repeated quarters don't
    re‑evaluate the same FS220 account columns.
    """
    if column_name and column_name in _numeric_cache:
        return _numeric_cache[column_name]

    non_null = series.dropna()
    if non_null.empty:
        result = False
    else:
        numeric_values = pd.to_numeric(non_null, errors="coerce")
        result = bool(numeric_values.notna().mean() >= threshold)

    if column_name:
        _numeric_cache[column_name] = result
    return result


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


def collapse_multi_record_table(
    df: pd.DataFrame, key_columns: tuple[str, ...]
) -> pd.DataFrame:
    value_columns = [c for c in df.columns if c not in key_columns]
    if not value_columns:
        return df.drop_duplicates(subset=list(key_columns))

    aggregation_functions = {}
    for column in value_columns:
        is_num = mostly_numeric(df[column], column_name=column)
        aggregation_functions[column] = numeric_sum if is_num else join_unique_values

    return (
        df.groupby(list(key_columns), as_index=False, dropna=False)
        .agg(aggregation_functions)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Merge quarter tables
# ---------------------------------------------------------------------------
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
            rename_map = {
                column: f"{column}__{table_id}" for column in overlapping_non_key
            }
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


# ---------------------------------------------------------------------------
# Column formatting
# ---------------------------------------------------------------------------
def format_column_name(
    column: str, account_name_map: dict[str, str]
) -> str:
    description = account_name_map.get(column, "").strip()
    if not description:
        description = column.lower()
    return f"{column} - {description}"


# ---------------------------------------------------------------------------
# Process a single quarter payload (extracted from main loop for clarity)
# ---------------------------------------------------------------------------
def _process_quarter_payload(
    quarter: str,
    payload: bytes,
    *,
    all_columns_seen: set[str],
    ordered_columns: list[str],
    global_account_name_map: dict[str, str],
    quarter_temp_dir: Path,
    multirecord_output_file: str,
    multirecord_header_written: list[bool],  # mutable flag list [bool]
) -> Path:
    """Parse one quarter's ZIP *payload*, write temp CSV, return its path.

    Multirecord rows are **streamed** directly to *multirecord_output_file*
    instead of being accumulated in memory.
    """
    quarter_temp_path = quarter_temp_dir / f"quarter-{quarter}.csv"

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
            raise FileNotFoundError(
                "No FS220 schedule files found in quarter zip."
            )

        quarter_tables: list[tuple[str, pd.DataFrame]] = []
        for schedule_file in schedule_files:
            with zf.open(schedule_file) as handle:
                table_df = read_ncua_csv(handle)
            table_df = normalize_dataframe_columns(table_df)

            missing_keys = [
                key for key in KEY_COLUMNS if key not in table_df.columns
            ]
            if missing_keys:
                print(
                    f"  SKIP TABLE: {quarter} {schedule_file} "
                    f"missing keys {missing_keys}"
                )
                continue

            duplicate_rows = table_df.duplicated(
                subset=list(KEY_COLUMNS)
            ).sum()
            if duplicate_rows:
                # Stream multirecord rows to disk immediately
                raw_multi_df = table_df.copy()
                raw_multi_df["SOURCE_TABLE"] = Path(schedule_file).stem.upper()
                raw_multi_df["SOURCE_QUARTER"] = quarter
                write_header = not multirecord_header_written[0]
                raw_multi_df.to_csv(
                    multirecord_output_file,
                    mode="a",
                    header=write_header,
                    index=False,
                )
                multirecord_header_written[0] = True
                del raw_multi_df

                table_df = collapse_multi_record_table(
                    table_df, key_columns=KEY_COLUMNS
                )
                print(
                    f"  COLLAPSED: {quarter} {schedule_file} "
                    f"{duplicate_rows} duplicate‑key rows aggregated."
                )

            quarter_tables.append((schedule_file, table_df))

        if not quarter_tables:
            raise ValueError(
                "No mergeable FS220 tables were found in this quarter."
            )

        merged_quarter_df = merge_quarter_tables(
            quarter_tables, key_columns=KEY_COLUMNS
        )

        for column in merged_quarter_df.columns:
            if column not in all_columns_seen:
                all_columns_seen.add(column)
                ordered_columns.append(column)

        merged_quarter_df.to_csv(quarter_temp_path, index=False)
        print(
            f"  OK: {quarter} -> "
            f"{merged_quarter_df.shape[0]:,} rows x "
            f"{merged_quarter_df.shape[1]:,} cols"
        )

    return quarter_temp_path


# ---------------------------------------------------------------------------
# Lightweight quarter processing (no temp files, no multirecord tracking)
# ---------------------------------------------------------------------------
def _process_quarter_to_dataframe(
    quarter: str, payload: bytes
) -> pd.DataFrame:
    """Parse one quarter's ZIP *payload* and return the merged FS220 DataFrame.

    Unlike :func:`_process_quarter_payload` this keeps everything in memory
    and skips multirecord tracking / temp‑file I/O.
    """
    with ZipFile(BytesIO(payload)) as zf:
        schedule_files = sorted(
            [
                name
                for name in zf.namelist()
                if FS220_TABLE_PATTERN.match(Path(name).name)
            ],
            key=str.lower,
        )
        if not schedule_files:
            raise FileNotFoundError(
                "No FS220 schedule files found in quarter zip."
            )

        quarter_tables: list[tuple[str, pd.DataFrame]] = []
        for schedule_file in schedule_files:
            with zf.open(schedule_file) as handle:
                table_df = read_ncua_csv(handle)
            table_df = normalize_dataframe_columns(table_df)

            missing_keys = [
                key for key in KEY_COLUMNS if key not in table_df.columns
            ]
            if missing_keys:
                print(
                    f"    SKIP TABLE: {quarter} {schedule_file} "
                    f"missing keys {missing_keys}"
                )
                continue

            duplicate_rows = table_df.duplicated(
                subset=list(KEY_COLUMNS)
            ).sum()
            if duplicate_rows:
                table_df = collapse_multi_record_table(
                    table_df, key_columns=KEY_COLUMNS
                )
                print(
                    f"    COLLAPSED: {quarter} {schedule_file} "
                    f"{duplicate_rows} duplicate‑key rows aggregated."
                )

            quarter_tables.append((schedule_file, table_df))

        if not quarter_tables:
            raise ValueError(
                "No mergeable FS220 tables were found in this quarter."
            )

        return merge_quarter_tables(
            quarter_tables, key_columns=KEY_COLUMNS
        )


# ---------------------------------------------------------------------------
# Generator: yield one quarter DataFrame at a time (for incremental writes)
# ---------------------------------------------------------------------------
def iter_quarter_dataframes(
    start_year: int,
    end_year: int,
    download_workers: int = 8,
    max_retries: int = DEFAULT_MAX_RETRIES,
    raw_column_names: bool = False,
):
    """Yield ``(quarter, dataframe)`` for each successfully processed quarter.

    All requested quarters are downloaded in parallel first.  Then the
    account-description files from every ZIP are merged into a single
    global name map so that column names are **consistent** across all
    quarters (e.g. ``ACCT_007 - Total Assets``).  Finally each quarter
    is parsed, columns renamed, and yielded one at a time so that peak
    memory stays at roughly one processed DataFrame.

    Parameters
    ----------
    raw_column_names : bool
        If *True*, columns keep their raw ``ACCT_007`` form instead of
        being formatted with descriptive names.
    """
    requested = build_requested_quarters(start_year, end_year)
    ssl_ctx = ssl.create_default_context()

    quarter_urls: dict[str, str] = {}
    try:
        quarter_urls = discover_quarter_urls(ssl_ctx)
        print(f"Discovered {len(quarter_urls)} quarter links from NCUA page.")
    except Exception as err:  # noqa: BLE001
        print(
            f"WARN: Could not discover quarter links ({err}). "
            "Using fallback URLs."
        )

    # ------------------------------------------------------------------
    # Phase 1: Download ALL quarters in parallel
    # ------------------------------------------------------------------
    print(
        f"\nDownloading {len(requested)} quarters "
        f"with {download_workers} workers …"
    )
    payloads: dict[str, tuple[bytes | None, str]] = {}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {
            pool.submit(
                download_quarter,
                q,
                quarter_urls.get(
                    q, FALLBACK_URL_TEMPLATE.format(quarter=q)
                ),
                ssl_ctx,
            ): q
            for q in requested
        }
        done_count = 0
        for future in as_completed(futures):
            quarter, data, err_msg = future.result()
            payloads[quarter] = (data, err_msg)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(requested):
                print(f"  … downloaded {done_count}/{len(requested)}")

    downloaded = sum(1 for d, _ in payloads.values() if d is not None)
    print(f"Downloaded {downloaded}/{len(requested)} quarters.\n")

    # ------------------------------------------------------------------
    # Phase 2: Build a global account-name map from ALL ZIPs
    # ------------------------------------------------------------------
    global_account_name_map: dict[str, str] = {}
    if not raw_column_names:
        for quarter in requested:
            data, _ = payloads[quarter]
            if data is None:
                continue
            try:
                with ZipFile(BytesIO(data)) as zf:
                    acct_map = load_account_name_map(zf)
                    for code, name in acct_map.items():
                        if name:
                            global_account_name_map[code] = name
            except Exception:  # noqa: BLE001
                pass
        print(
            f"Built account name map with "
            f"{len(global_account_name_map)} entries."
        )

    # ------------------------------------------------------------------
    # Phase 3: Process and yield one quarter at a time
    # ------------------------------------------------------------------
    total_yielded = 0
    for quarter in requested:
        data, err_msg = payloads[quarter]
        if data is None:
            print(f"  SKIP: {quarter} -> {err_msg}")
            continue
        try:
            df = _process_quarter_to_dataframe(quarter, data)

            # Rename ACCT_* columns to descriptive names; leave
            # metadata columns (CU_NUMBER, CYCLE_DATE, …) untouched
            # so downstream MERGE keys stay predictable.
            if not raw_column_names and global_account_name_map:
                df.columns = [
                    format_column_name(col, global_account_name_map)
                    if col.startswith("ACCT_")
                    else col
                    for col in df.columns
                ]

            print(
                f"  OK: {quarter} -> "
                f"{df.shape[0]:,} rows x {df.shape[1]:,} cols"
            )
            total_yielded += 1
            yield quarter, df
            del df
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {quarter} -> {type(exc).__name__}: {exc}")
        finally:
            payloads[quarter] = (None, "")  # free ZIP payload
            gc.collect()

    print(
        f"\nDone. Yielded {total_yielded}/{len(requested)} "
        f"requested quarters."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_ingestion(
    start_year: int,
    end_year: int,
    output_file: str,
    multirecord_output_file: str,
    cleanup_temp_files: bool = True,
    resume_from_temp: bool = False,
    raw_column_names: bool = False,
    download_workers: int = DEFAULT_DOWNLOAD_WORKERS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    return_dataframe: bool = False,
) -> "pd.DataFrame | dict[str, str] | None":
    """Download and merge NCUA call‑report data.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive range of years to retrieve.
    output_file : str
        Path for the merged FS220 CSV.
    multirecord_output_file : str
        Path for raw duplicate‑key rows from schedules like FS220CUSO.
    cleanup_temp_files : bool
        Remove per‑quarter temp CSVs after final assembly.
    resume_from_temp : bool
        Re‑use existing temp CSVs and only download missing quarters.
    raw_column_names : bool
        If *True*, keep raw account codes as column headers (e.g. ``ACCT_881``)
        instead of ``ACCT_881 - Total Loans and Leases``.  Useful when loading
        directly into Delta tables where special characters break.
    download_workers : int
        Number of parallel threads for downloading quarter ZIPs.
    max_retries : int
        Max HTTP retry attempts per quarter download.
    return_dataframe : bool
        If *True*, return the assembled pandas DataFrame directly instead
        of writing a final CSV.  This avoids filesystem access entirely,
        which is required on Databricks serverless compute.

    Returns
    -------
    pd.DataFrame | dict[str, str] | None
        When *return_dataframe* is True, returns the assembled DataFrame.
        When *raw_column_names* is True (and *return_dataframe* is False),
        returns the global account‑name mapping.
        Returns *None* when no data was retrieved.
    """
    requested_quarters = build_requested_quarters(
        start_year=start_year, end_year=end_year
    )
    ssl_ctx = ssl.create_default_context()

    # -- Discover available quarter URLs from the NCUA page --
    quarter_urls: dict[str, str] = {}
    try:
        quarter_urls = discover_quarter_urls(ssl_ctx=ssl_ctx)
        print(f"Discovered {len(quarter_urls)} quarter links from NCUA page.")
    except Exception as error:  # noqa: BLE001
        print(
            f"WARN: Could not discover quarter links from NCUA page ({error})."
        )
        print(
            "      Falling back to default modern URL format for "
            "requested quarters."
        )

    # -- Prepare temp directory --
    quarter_temp_dir = Path(".ncua_quarter_temp")
    quarter_temp_dir.mkdir(exist_ok=True)
    if not resume_from_temp:
        for temp_csv in quarter_temp_dir.glob("quarter-*.csv"):
            temp_csv.unlink()

    # Clear multirecord output for a fresh run
    multi_path = Path(multirecord_output_file)
    if not resume_from_temp and multi_path.exists():
        multi_path.unlink()
    multirecord_header_written: list[bool] = [
        resume_from_temp and multi_path.exists()
    ]

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    quarter_temp_files: list[tuple[str, Path]] = []
    all_columns_seen: set[str] = set()
    ordered_columns: list[str] = []
    global_account_name_map: dict[str, str] = {}

    # -- Identify which quarters still need downloading --
    quarters_to_download: list[str] = []
    for quarter in requested_quarters:
        quarter_temp_path = quarter_temp_dir / f"quarter-{quarter}.csv"
        if resume_from_temp and quarter_temp_path.exists():
            header_columns = pd.read_csv(
                quarter_temp_path, nrows=0
            ).columns.tolist()
            for column in header_columns:
                if column not in all_columns_seen:
                    all_columns_seen.add(column)
                    ordered_columns.append(column)
            quarter_temp_files.append((quarter, quarter_temp_path))
            successful.append(quarter)
            print(f"RESUME: {quarter} from existing temp file.")
        else:
            quarters_to_download.append(quarter)

    # -- Download in parallel, process sequentially --
    if quarters_to_download:
        print(
            f"Downloading {len(quarters_to_download)} quarters "
            f"with {download_workers} workers …"
        )
        download_results: dict[str, tuple[bytes | None, str]] = {}

        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures = {
                pool.submit(
                    download_quarter,
                    q,
                    quarter_urls.get(
                        q, FALLBACK_URL_TEMPLATE.format(quarter=q)
                    ),
                    ssl_ctx,
                ): q
                for q in quarters_to_download
            }
            for future in as_completed(futures):
                quarter, payload, error_msg = future.result()
                download_results[quarter] = (payload, error_msg)

        # Process in chronological order (not completion order)
        for quarter in quarters_to_download:
            payload, error_msg = download_results[quarter]
            if payload is None:
                failed.append((quarter, error_msg))
                print(f"SKIP: {quarter} -> {error_msg}")
                continue

            try:
                temp_path = _process_quarter_payload(
                    quarter,
                    payload,
                    all_columns_seen=all_columns_seen,
                    ordered_columns=ordered_columns,
                    global_account_name_map=global_account_name_map,
                    quarter_temp_dir=quarter_temp_dir,
                    multirecord_output_file=multirecord_output_file,
                    multirecord_header_written=multirecord_header_written,
                )
                quarter_temp_files.append((quarter, temp_path))
                successful.append(quarter)
            except Exception as error:  # noqa: BLE001
                failed.append((quarter, f"{type(error).__name__}: {error}"))
                print(f"ERROR: {quarter} -> {type(error).__name__}: {error}")
            finally:
                # Free the (potentially large) ZIP payload immediately
                del payload
                gc.collect()

    if not quarter_temp_files:
        print("No data was retrieved for the requested quarter range.")
        return None

    # -- Assemble final CSV from temp files --
    final_columns = [
        column for column in KEY_COLUMNS if column in all_columns_seen
    ] + [column for column in ordered_columns if column not in KEY_COLUMNS]

    if raw_column_names:
        final_headers = final_columns
    else:
        final_headers = [
            format_column_name(column, global_account_name_map)
            for column in final_columns
        ]

    # -- Helper: cleanup temp files --
    def _cleanup():
        if cleanup_temp_files:
            for _, qtp in quarter_temp_files:
                if qtp.exists():
                    qtp.unlink()
            if quarter_temp_dir.exists() and not any(quarter_temp_dir.iterdir()):
                quarter_temp_dir.rmdir()

    # -- Return in-memory DataFrame (for Databricks serverless) --
    if return_dataframe:
        chunks = []
        for _, quarter_temp_path in quarter_temp_files:
            quarter_df = pd.read_csv(quarter_temp_path, low_memory=False)
            quarter_df = quarter_df.reindex(columns=final_columns)
            quarter_df.columns = final_headers
            chunks.append(quarter_df)
            del quarter_df
        result_df = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(columns=final_headers)
        )
        del chunks
        gc.collect()
        _cleanup()

        print(
            f"\nDone. Retrieved {len(successful)}/{len(requested_quarters)} "
            f"requested quarters."
        )
        print(
            f"Assembled {len(result_df):,} rows x "
            f"{len(result_df.columns):,} columns in memory."
        )
        if failed:
            print("Missing/failed quarters:")
            for quarter, reason in failed:
                print(f"  - {quarter}: {reason}")
        return result_df

    # -- Write final CSV to disk --
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
        del quarter_df
        gc.collect()

    # -- Format multirecord output headers to match (if file exists) --
    if multi_path.exists() and multi_path.stat().st_size > 0 and not raw_column_names:
        multi_df = pd.read_csv(multi_path, low_memory=False)
        multi_headers = [
            format_column_name(column, global_account_name_map)
            if column not in ("SOURCE_TABLE", "SOURCE_QUARTER")
            else column
            for column in multi_df.columns
        ]
        multi_df.columns = multi_headers
        multi_df.to_csv(multirecord_output_file, index=False)
        print(
            f"Saved multi‑record raw rows ({len(multi_df):,}) "
            f"to {multirecord_output_file}."
        )
        del multi_df
        gc.collect()

    _cleanup()

    print()
    print(
        f"Done. Retrieved {len(successful)}/{len(requested_quarters)} "
        f"requested quarters."
    )
    print(
        f"Saved {total_rows:,} rows x {len(final_headers):,} columns "
        f"to {output_file}."
    )
    if failed:
        print("Missing/failed quarters:")
        for quarter, reason in failed:
            print(f"  - {quarter}: {reason}")

    return global_account_name_map if raw_column_names else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NCUA quarterly call report data and build a maximally "
            "wide FS220 dataset by merging all available FS220* schedules."
        )
    )
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-file", default="NCUA_Call_Report.csv")
    parser.add_argument(
        "--multirecord-output-file",
        default="NCUA_Call_Report_Multirecord.csv",
        help=(
            "Output for raw rows from FS220 schedules that contain duplicate "
            "keys (for example FS220CUSO in older years)."
        ),
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary per‑quarter CSV files used to assemble the final output.",
    )
    parser.add_argument(
        "--resume-from-temp",
        action="store_true",
        help=(
            "Reuse existing .ncua_quarter_temp/quarter‑YYYY‑MM.csv files and "
            "only download/process missing quarters before producing final "
            "outputs."
        ),
    )
    parser.add_argument(
        "--raw-column-names",
        action="store_true",
        help=(
            "Keep raw account codes as headers (e.g. ACCT_881) instead of "
            "formatted names.  Useful for loading into Delta / database tables."
        ),
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help=f"Parallel download threads (default: {DEFAULT_DOWNLOAD_WORKERS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"HTTP retry attempts per quarter (default: {DEFAULT_MAX_RETRIES}).",
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
        raw_column_names=cli_args.raw_column_names,
        download_workers=cli_args.download_workers,
        max_retries=cli_args.max_retries,
    )
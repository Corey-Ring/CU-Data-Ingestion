# Credit_Union_Scrape

This repo contains ingestion tooling for NCUA quarterly call report data.

## What It Retrieves

`ingest_ncua_call_report.py` downloads and merges all available `FS220*` schedules for each requested quarter, not only `FS220.txt`.

- Primary output: one wide CSV with maximal column coverage across schedules.
- Secondary output: raw rows from multi-record schedules (for example historical `FS220CUSO`) so no detail is silently lost.

## Why This Version

The previous workflow only pulled `FS220.txt` and missed most available NCUA call report columns.  
The new workflow:

- Discovers quarter ZIP URLs directly from NCUA's quarterly data page (supports historical link patterns back to 1994).
- Merges all `FS220*` tables by `CU_NUMBER`, `CYCLE_DATE`, `JOIN_NUMBER`.
- Handles duplicate-key schedules by collapsing into the wide dataset and also exporting raw multi-record rows.
- Uses an inclusive year range (`start_year` through `end_year`).

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ingestion:

```bash
python ingest_ncua_call_report.py --start-year 1994 --end-year 2026
```

Optional arguments:

- `--output-file` (default: `NCUA_Call_Report.csv`)
- `--multirecord-output-file` (default: `NCUA_Call_Report_Multirecord.csv`)
- `--keep-temp-files` (keeps intermediate per-quarter files)
- `--resume-from-temp` (reuses existing `.ncua_quarter_temp` quarter files and only backfills missing quarters)

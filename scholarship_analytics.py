

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure the console can print special characters (e.g. the "n-tilde" in
# "Las Pinas"/"Paranaque"). Windows consoles default to cp1252 which cannot
# encode some characters; reconfigure to UTF-8 where supported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_NAME = "NCR_Scholarship_Analytics_Sample_Dataset.xlsx"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_default_input():
    """Find the dataset automatically.

    Looks first next to this script, then in the user's Downloads folder.
    """
    candidates = [
        os.path.join(_SCRIPT_DIR, DATASET_NAME),
        os.path.join(os.path.expanduser("~"), "Downloads", DATASET_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


DEFAULT_INPUT = _resolve_default_input()
CLEANED_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "NCR_Scholarship_Cleaned.xlsx"
)

# Mapping used to repair corrupted (mojibake) city names found in the raw file.
CITY_FIXES = {
    "Las Pi\ufffdas": "Las Pi\u00f1as",   # Las Pi(?)as -> Las Pinas (Pi-ny-as)
    "Las Pinas": "Las Pi\u00f1as",
    "Para\ufffdaque": "Para\u00f1aque",   # Para(?)aque -> Paranaque
    "Paranaque": "Para\u00f1aque",
}

# Columns expected to be numeric after cleaning.
NUMERIC_COLS = [
    "age", "monthly_household_income", "household_size", "weekly_work_hours",
    "commute_time_minutes", "applications_submitted", "monthly_grant_amount",
    "gpa", "units_enrolled", "financial_need_score", "year_level",
]

# Yes/No columns that become boolean.
YESNO_COLS = [
    "first_gen_college", "working_student", "scholarship_applied",
]


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
def load_data(path=DEFAULT_INPUT):
    """Load the 'Dataset' sheet from the Excel workbook.

    Returns the raw DataFrame, or None if the file cannot be read.
    """
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return None
    try:
        df = pd.read_excel(path, sheet_name="Dataset", engine="openpyxl")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [ERROR] Could not read Excel file: {exc}")
        return None
    print(f"  Loaded '{os.path.basename(path)}'")
    print(f"  Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    return df


# ===========================================================================
# 2. DATA CLEANING
# ===========================================================================
def _rename_columns(df):
    """Rename original columns to clean snake_case names."""
    rename_map = {
        "Student_ID": "student_id",
        "Student_Name": "student_name",
        "Age": "age",
        "Gender": "gender",
        "Region": "region",
        "City": "city",
        "NCR_Area": "ncr_area",
        "Course_Program": "course_program",
        "Year_Level": "year_level",
        "School_Type": "school_type",
        "Monthly_Household_Income": "monthly_household_income",
        "Household_Size": "household_size",
        "First_Gen_College": "first_gen_college",
        "Working_Student": "working_student",
        "Weekly_Work_Hours": "weekly_work_hours",
        "Commute_Time_Minutes": "commute_time_minutes",
        "Scholarship_Applied": "scholarship_applied",
        "Applications_Submitted": "applications_submitted",
        "Scholarship_Status": "scholarship_status",
        "Scholarship_Type": "scholarship_type",
        "Scholarship_Provider": "scholarship_provider",
        "Monthly_Grant_Amount": "monthly_grant_amount",
        "GPA": "gpa",
        "Units_Enrolled": "units_enrolled",
        "Renewal_Status": "renewal_status",
        "Financial_Need_Score": "financial_need_score",
        "Risk_Category": "risk_category",
    }
    return df.rename(columns=rename_map)


def clean_data(raw_df):
    """Run the full cleaning pipeline, printing what each step does.

    Returns the cleaned DataFrame.
    """
    print("\n" + "-" * 64)
    print("DATA CLEANING REPORT")
    print("-" * 64)

    df = raw_df.copy()

    # --- Step 1: Rename columns -------------------------------------------
    df = _rename_columns(df)
    print("[1] Renamed columns to clean snake_case names.")

    # --- Step 2: Correct inconsistent text entries ------------------------
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
    fixed_cities = df["city"].isin(CITY_FIXES.keys()).sum()
    df["city"] = df["city"].replace(CITY_FIXES)
    print(f"[2] Trimmed whitespace on {len(text_cols)} text columns; "
          f"fixed {fixed_cities} corrupted city names "
          f"(e.g. Las Pi\ufffdas -> Las Pi\u00f1as).")

    # --- Step 3: Handle missing values ------------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("[3] Missing values: 0 found.")
    else:
        for col, cnt in missing.items():
            if col in ("scholarship_type", "scholarship_provider"):
                df[col] = df[col].fillna("No Scholarship")
                print(f"[3] Imputed {cnt} missing '{col}' values with "
                      f"'No Scholarship' (blank only for non-scholars).")
            elif df[col].dtype.kind in "biufc":
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"[3] Imputed {cnt} missing numeric '{col}' with "
                      f"median ({med}).")
            else:
                df[col] = df[col].fillna("Unknown")
                print(f"[3] Imputed {cnt} missing '{col}' with 'Unknown'.")

    # --- Step 4: Remove duplicates ----------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    dup_rows = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset=["student_id"], keep="first")
    dup_ids = before - len(df)
    print(f"[4] Removed {dup_rows} full-row duplicate(s) and {dup_ids} "
          f"duplicate Student_ID(s).")

    # --- Step 5: Convert data types ---------------------------------------
    converted = []
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted.append(col)
    for col in YESNO_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.lower().map(
                    {"yes": True, "no": False}
                )
            )
    cat_cols = [
        "gender", "region", "ncr_area", "course_program", "school_type",
        "scholarship_status", "scholarship_type", "scholarship_provider",
        "renewal_status", "risk_category",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    print(f"[5] Converted {len(converted)} columns to numeric, "
          f"{len(YESNO_COLS)} Yes/No columns to boolean, and "
          f"{len(cat_cols)} columns to category.")

    # --- Step 6: Filter invalid data --------------------------------------
    before = len(df)
    valid = (
        df["age"].between(16, 30)
        & df["gpa"].between(1.0, 5.0)
        & (df["monthly_household_income"] >= 0)
        & (df["units_enrolled"] >= 0)
        & df["financial_need_score"].between(0, 100)
    )
    invalid_count = int((~valid).sum())
    df = df[valid].copy()
    print(f"[6] Filtered {invalid_count} invalid row(s) violating domain rules "
          f"(Age 16-30, GPA 1.0-5.0, non-negative income/units, "
          f"need score 0-100).")

    # --- Step 7: Handle outliers (IQR capping / winsorize) ----------------
    outlier_cols = [
        "monthly_household_income", "financial_need_score",
        "commute_time_minutes", "weekly_work_hours",
    ]
    total_capped = 0
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (df[col] < low) | (df[col] > high)
        n = int(mask.sum())
        total_capped += n
        df[col] = df[col].clip(lower=low, upper=high)
    print(f"[7] Handled outliers on {len(outlier_cols)} numeric columns using "
          f"the IQR method (capped {total_capped} extreme value(s)).")

    df = df.reset_index(drop=True)
    print("-" * 64)
    print(f"Cleaning complete. Final shape: {df.shape[0]} rows x "
          f"{df.shape[1]} columns.")
    print("-" * 64)
    return df


# ===========================================================================
# 3. COMPUTATIONS / ANALYSIS
# ===========================================================================
def show_descriptive_stats(df):
    """Mean, median, mode, std, min, max for the main numeric columns."""
    print("\n" + "=" * 64)
    print("DESCRIPTIVE STATISTICS (numeric columns)")
    print("=" * 64)
    cols = [
        "age", "monthly_household_income", "household_size",
        "weekly_work_hours", "commute_time_minutes", "monthly_grant_amount",
        "gpa", "units_enrolled", "financial_need_score",
    ]
    rows = []
    for col in cols:
        s = df[col].dropna()
        mode_val = s.mode()
        rows.append({
            "column": col,
            "mean": round(s.mean(), 2),
            "median": round(s.median(), 2),
            "mode": round(mode_val.iloc[0], 2) if len(mode_val) else np.nan,
            "std": round(s.std(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
        })
    stats = pd.DataFrame(rows).set_index("column")
    print(stats.to_string())


def show_sorting(df):
    """Sorting demonstration: top students by financial need and by GPA."""
    print("\n" + "=" * 64)
    print("SORTING")
    print("=" * 64)
    cols = ["student_id", "student_name", "ncr_area",
            "financial_need_score", "gpa"]
    print("\nTop 10 students by Financial Need Score (highest need):")
    print(df.sort_values("financial_need_score", ascending=False)
            .head(10)[cols].to_string(index=False))
    print("\nTop 10 academic performers (lowest GPA = best on PH scale):")
    print(df.sort_values("gpa", ascending=True)
            .head(10)[cols].to_string(index=False))


def show_filtering(df):
    """Filtering demonstration: scholars and high-risk subsets."""
    print("\n" + "=" * 64)
    print("FILTERING")
    print("=" * 64)
    scholars = df[df["scholarship_status"] == "Scholar"]
    high_risk = df[df["risk_category"] == "High"]
    working = df[df["working_student"] == True]  # noqa: E712
    print(f"Scholars                : {len(scholars)} students")
    print(f"High-risk students      : {len(high_risk)} students")
    print(f"Working students        : {len(working)} students")
    print("\nSample of high-risk students:")
    cols = ["student_id", "student_name", "gpa",
            "financial_need_score", "risk_category"]
    print(high_risk.head(8)[cols].to_string(index=False))


def show_grouping(df):
    """Grouping + aggregation by NCR area, school type, and course."""
    print("\n" + "=" * 64)
    print("GROUPING & AGGREGATION")
    print("=" * 64)

    by_area = df.groupby("ncr_area", observed=True).agg(
        students=("student_id", "count"),
        avg_income=("monthly_household_income", "mean"),
        avg_gpa=("gpa", "mean"),
        avg_grant=("monthly_grant_amount", "mean"),
    ).round(2)
    print("\nBy NCR Area:")
    print(by_area.to_string())

    by_school = df.groupby("school_type", observed=True).agg(
        students=("student_id", "count"),
        avg_income=("monthly_household_income", "mean"),
        avg_need=("financial_need_score", "mean"),
    ).round(2)
    print("\nBy School Type:")
    print(by_school.to_string())

    by_course = df.groupby("course_program", observed=True).agg(
        students=("student_id", "count"),
        avg_gpa=("gpa", "mean"),
    ).round(2).sort_values("students", ascending=False)
    print("\nBy Course Program:")
    print(by_course.to_string())


def show_totals_percentages(df):
    """Totals, averages, percentages, and rankings."""
    print("\n" + "=" * 64)
    print("TOTALS, AVERAGES, PERCENTAGES & RANKINGS")
    print("=" * 64)
    total = len(df)
    applicants = int((df["scholarship_applied"] == True).sum())  # noqa: E712
    scholars = int((df["scholarship_status"] == "Scholar").sum())
    working = int((df["working_student"] == True).sum())  # noqa: E712

    print(f"Total students            : {total}")
    print(f"Total applicants          : {applicants} "
          f"({applicants / total * 100:.1f}%)")
    print(f"Total scholars            : {scholars} "
          f"({scholars / total * 100:.1f}%)")
    approval = (scholars / applicants * 100) if applicants else 0
    print(f"Approval rate (scholars/applicants) : {approval:.1f}%")
    print(f"Working students          : {working} "
          f"({working / total * 100:.1f}%)")
    print(f"Average household income  : PHP {df['monthly_household_income'].mean():,.0f}")
    print(f"Average GPA               : {df['gpa'].mean():.2f}")
    print(f"Average grant (scholars)  : PHP "
          f"{df[df['scholarship_status'] == 'Scholar']['monthly_grant_amount'].mean():,.0f}")

    # Ranking: cities by approval rate.
    g = df.groupby("city", observed=True)
    rank = pd.DataFrame({
        "students": g.size(),
        "applicants": g.apply(
            lambda x: (x["scholarship_applied"] == True).sum(),  # noqa: E712
            include_groups=False,
        ),
        "scholars": g.apply(
            lambda x: (x["scholarship_status"] == "Scholar").sum(),
            include_groups=False,
        ),
    })
    rank["approval_rate_%"] = (
        rank["scholars"] / rank["applicants"].replace(0, np.nan) * 100
    ).round(1)
    rank = rank.sort_values("approval_rate_%", ascending=False)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    print("\nCity ranking by approval rate:")
    print(rank.to_string())


def show_frequency_table(df):
    """Frequency tables for key categorical columns."""
    print("\n" + "=" * 64)
    print("FREQUENCY TABLES")
    print("=" * 64)
    for col in ["risk_category", "scholarship_status", "scholarship_type"]:
        counts = df[col].value_counts()
        pct = (counts / len(df) * 100).round(1)
        table = pd.DataFrame({"frequency": counts, "percent_%": pct})
        print(f"\n{col}:")
        print(table.to_string())


def show_correlation(df):
    """Correlation analysis among numeric variables."""
    print("\n" + "=" * 64)
    print("CORRELATION ANALYSIS")
    print("=" * 64)
    cols = [
        "monthly_household_income", "gpa", "financial_need_score",
        "monthly_grant_amount", "weekly_work_hours", "commute_time_minutes",
    ]
    corr = df[cols].corr().round(2)
    print(corr.to_string())
    # Highlight strongest relationship (excluding self-correlations).
    c = corr.where(~np.eye(len(corr), dtype=bool))
    strongest = c.abs().stack().idxmax()
    val = corr.loc[strongest[0], strongest[1]]
    print(f"\nStrongest relationship: {strongest[0]} vs {strongest[1]} "
          f"(r = {val:.2f}).")


def show_trend(df):
    """Trend analysis: averages across Year Level."""
    print("\n" + "=" * 64)
    print("TREND ANALYSIS (by Year Level)")
    print("=" * 64)
    trend = df.groupby("year_level", observed=True).agg(
        students=("student_id", "count"),
        avg_gpa=("gpa", "mean"),
        avg_need=("financial_need_score", "mean"),
        avg_income=("monthly_household_income", "mean"),
    ).round(2)
    print(trend.to_string())
    return trend


def run_all_statistics(df):
    """Convenience: run every statistical section."""
    show_descriptive_stats(df)
    show_sorting(df)
    show_filtering(df)
    show_frequency_table(df)


# ===========================================================================
# 4. VISUALIZATIONS
# ===========================================================================
def chart_bar(df):
    """Bar graph: number of scholars by NCR Area."""
    data = (
        df[df["scholarship_status"] == "Scholar"]
        .groupby("ncr_area", observed=True)["student_id"].count()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(9, 5))
    plt.bar(data.index, data.values, color="#2a7fff", edgecolor="black")
    plt.title("Number of Scholars by NCR Area")
    plt.xlabel("NCR Area")
    plt.ylabel("Number of Scholars")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def chart_line(df):
    """Line graph: average Financial Need Score by Year Level."""
    data = df.groupby("year_level", observed=True)["financial_need_score"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(data.index, data.values, marker="o", color="#d6336c",
             linewidth=2)
    plt.title("Average Financial Need Score by Year Level")
    plt.xlabel("Year Level")
    plt.ylabel("Average Financial Need Score")
    plt.xticks(list(data.index))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def chart_pie(df):
    """Pie chart: distribution of Risk Category."""
    data = df["risk_category"].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(data.values, labels=data.index, autopct="%1.1f%%",
            startangle=90, colors=["#69db7c", "#ffd43b", "#ff6b6b"])
    plt.title("Distribution of Students by Risk Category")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def chart_scatter(df):
    """Scatterplot: Household Income vs Financial Need Score."""
    plt.figure(figsize=(9, 5))
    plt.scatter(df["monthly_household_income"], df["financial_need_score"],
                alpha=0.5, color="#7048e8", edgecolor="white", s=30)
    plt.title("Household Income vs Financial Need Score")
    plt.xlabel("Monthly Household Income (PHP)")
    plt.ylabel("Financial Need Score")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def chart_histogram(df):
    """Histogram: distribution of GPA."""
    plt.figure(figsize=(9, 5))
    plt.hist(df["gpa"].dropna(), bins=15, color="#4dabf7",
             edgecolor="black")
    plt.title("Distribution of GPA (Philippine scale: 1.0 = best)")
    plt.xlabel("GPA")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()


def show_all_charts(df):
    """Display all required charts in sequence."""
    print("\nGenerating charts... close each window to see the next.")
    chart_bar(df)
    chart_line(df)
    chart_pie(df)
    chart_scatter(df)
    chart_histogram(df)
    print("All charts displayed.")


# ===========================================================================
# 5. SUMMARY OUTPUT
# ===========================================================================
def print_summary(df):
    """Consolidated headline metrics."""
    total = len(df)
    applicants = int((df["scholarship_applied"] == True).sum())  # noqa: E712
    scholars = int((df["scholarship_status"] == "Scholar").sum())
    approval = (scholars / applicants * 100) if applicants else 0
    top_area = (
        df[df["scholarship_status"] == "Scholar"]
        .groupby("ncr_area", observed=True)["student_id"].count().idxmax()
    )
    print("\n" + "#" * 64)
    print("#  NCR SCHOLARSHIP ANALYTICS - EXECUTIVE SUMMARY")
    print("#" * 64)
    print(f"  Total students              : {total}")
    print(f"  Scholarship applicants      : {applicants} "
          f"({applicants / total * 100:.1f}%)")
    print(f"  Scholars                    : {scholars} "
          f"({scholars / total * 100:.1f}%)")
    print(f"  Approval rate               : {approval:.1f}%")
    print(f"  Avg household income        : PHP "
          f"{df['monthly_household_income'].mean():,.0f}")
    print(f"  Avg GPA (1.0 = best)        : {df['gpa'].mean():.2f}")
    print(f"  Avg financial need score    : {df['financial_need_score'].mean():.1f}")
    print(f"  Area with most scholars     : {top_area}")
    print("#" * 64)


# ===========================================================================
# 5b. EXPORT CLEANED DATA
# ===========================================================================
def export_cleaned(df, path=CLEANED_OUTPUT):
    """Write the cleaned DataFrame to an Excel file."""
    try:
        df.to_excel(path, index=False, sheet_name="Cleaned", engine="openpyxl")
        print(f"  Cleaned data exported to: {path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [ERROR] Could not write Excel file: {exc}")


def show_preview(df, label, show_dtypes=False):
    """Print a console preview of a DataFrame: shape, columns, first 20 rows.

    Used by the 'Load datasets' and 'Display cleaned data' menu options so the
    user can see the actual data in the console.
    """
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 64)
    print(label)
    print("=" * 64)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:")
    print(", ".join(map(str, df.columns)))
    print(f"\nFirst 20 rows:")
    print(df.head(20).to_string())
    if show_dtypes:
        print("\nColumn data types:")
        print(df.dtypes.to_string())


# ===========================================================================
# 6. MENU LOOP
# ===========================================================================
MENU = """
==========================================================
   NCR SCHOLARSHIP ANALYTICS - MAIN MENU
==========================================================
  1) Load datasets
  2) Display cleaned data
  3) Perform computations
  4) Generate charts automatically
  5) Output summarized results
  0) Exit
==========================================================
"""


def _ensure_loaded(state):
    if state["raw"] is None:
        print("  [!] Please load the dataset first (option 1).")
        return False
    return True


def _ensure_cleaned(state):
    if not _ensure_loaded(state):
        return False
    if state["clean"] is None:
        print("  [i] Data not cleaned yet - cleaning now...")
        state["clean"] = clean_data(state["raw"])
    return True


def perform_computations(df):
    """Run the full analytical suite for the 'Perform computations' option."""
    show_descriptive_stats(df)
    show_sorting(df)
    show_filtering(df)
    show_grouping(df)
    show_totals_percentages(df)
    show_frequency_table(df)
    show_correlation(df)
    show_trend(df)


def main():
    state = {"raw": None, "clean": None}
    print("Welcome to the NCR Scholarship Analytics console application.")
    while True:
        print(MENU)
        choice = input("Enter choice: ").strip()

        if choice == "1":
            # Load datasets: load the raw data and display it in the console.
            state["raw"] = load_data(DEFAULT_INPUT)
            state["clean"] = None
            if state["raw"] is not None:
                show_preview(state["raw"], "LOADED DATASET (raw)")

        elif choice == "2":
            # Display cleaned data: clean, show a preview, and export to Excel.
            if _ensure_loaded(state):
                state["clean"] = clean_data(state["raw"])
                show_preview(state["clean"], "CLEANED DATASET",
                             show_dtypes=True)
                export_cleaned(state["clean"])

        elif choice == "3":
            # Perform computations.
            if _ensure_cleaned(state):
                perform_computations(state["clean"])

        elif choice == "4":
            # Generate charts automatically.
            if _ensure_cleaned(state):
                show_all_charts(state["clean"])

        elif choice == "5":
            # Output summarized results.
            if _ensure_cleaned(state):
                print_summary(state["clean"])

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("  [!] Invalid choice. Please enter a number from the menu.")


if __name__ == "__main__":
    main()

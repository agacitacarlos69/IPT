
import os
import sys
import pandas as pd
import numpy
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

class ScholarshipAnalytics:

    def __init__(self):
        # Find the dataset next to this program, or in the Downloads folder.
        folder = os.path.dirname(os.path.abspath(__file__))
        name = "NCR_Scholarship_Analytics_Sample_Dataset.xlsx"
        path1 = os.path.join(folder, name)
        path2 = os.path.join(os.path.expanduser("~"), "Downloads", name)
        if os.path.exists(path1):
            self.input_file = path1
        else:
            self.input_file = path2
        self.output_file = os.path.join(folder, "NCR_Scholarship_Cleaned.xlsx")

        self.df = None          # the working DataFrame
        self.is_cleaned = False  # True once the data has been cleaned

  
    def load_data(self):
        try:
            self.df = pd.read_excel(self.input_file, sheet_name="Dataset")
            self.is_cleaned = False
            print("Loaded:", os.path.basename(self.input_file))
            print("Number of rows:", len(self.df))
            print("Number of columns:", len(self.df.columns))
            self.display_data("LOADED DATASET (raw)")
        except FileNotFoundError:
            print("Error: The dataset file was not found.")
        except Exception:
            print("Error: The dataset could not be loaded.")

    def display_data(self, title):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print()
        print("=" * 60)
        print(title)
        print("=" * 60)
        print("Rows:", len(self.df), " Columns:", len(self.df.columns))
        print()
        print("Columns:")
        print(", ".join(self.df.columns))
        print()
        print("First 20 rows:")
        print(self.df.head(20).to_string())

    
    def clean_data(self):
        print()
        print("-" * 60)
        print("DATA CLEANING REPORT")
        print("-" * 60)

        # --- Step 1: Rename the columns to simple names --------------------
        new_names = {
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
        self.df = self.df.rename(columns=new_names)
        print("[1] Renamed the columns to simple lowercase names.")

        # --- Step 2: Correct inconsistent text entries --------------------
        city_fixes = {
            "Las Pinas": "Las Pi\u00f1as",
            "Paranaque": "Para\u00f1aque",
        }
        text_columns = ["student_name", "gender", "region", "city", "ncr_area",
                        "course_program", "school_type", "first_gen_college",
                        "working_student", "scholarship_applied",
                        "scholarship_status", "scholarship_type",
                        "scholarship_provider", "renewal_status",
                        "risk_category"]
        fixed = 0
        for x in self.df.index:
            for col in text_columns:
                value = self.df.loc[x, col]
                if isinstance(value, str):
                    value = value.strip()
                    if value in city_fixes:
                        value = city_fixes[value]
                        fixed = fixed + 1
                    self.df.loc[x, col] = value
        print("[2] Trimmed spaces and standardized city names. Fixed:", fixed)

        # --- Step 3: Handle missing values --------------------------------
        missing_type = self.df["scholarship_type"].isnull().sum()
        missing_provider = self.df["scholarship_provider"].isnull().sum()
        self.df.fillna({"scholarship_type": "No Scholarship"}, inplace=True)
        self.df.fillna({"scholarship_provider": "No Scholarship"}, inplace=True)
        print("[3] Filled missing scholarship_type:", missing_type,
              "and scholarship_provider:", missing_provider)

        # --- Step 4: Remove duplicates ------------------------------------
        duplicates = self.df.duplicated().sum()
        self.df.drop_duplicates(inplace=True)
        print("[4] Removed duplicate rows:", duplicates)

        # --- Step 5: Convert data types -----------------------------------
        number_columns = ["age", "monthly_household_income", "household_size",
                          "weekly_work_hours", "commute_time_minutes",
                          "applications_submitted", "monthly_grant_amount",
                          "gpa", "units_enrolled", "financial_need_score",
                          "year_level"]
        for col in number_columns:
            self.df[col] = pd.to_numeric(self.df[col])
        yes_no_columns = ["first_gen_college", "working_student",
                          "scholarship_applied"]
        for col in yes_no_columns:
            self.df[col] = self.df[col].replace({"Yes": True, "No": False})
        print("[5] Converted number columns and Yes/No columns.")

        # --- Step 6: Filter invalid data ----------------------------------
        removed = 0
        for x in self.df.index:
            age = self.df.loc[x, "age"]
            gpa = self.df.loc[x, "gpa"]
            income = self.df.loc[x, "monthly_household_income"]
            need = self.df.loc[x, "financial_need_score"]
            if age < 16 or age > 30 or gpa < 1.0 or gpa > 5.0 \
                    or income < 0 or need < 0 or need > 100:
                self.df.drop(x, inplace=True)
                removed = removed + 1
        print("[6] Removed invalid rows:", removed)

        # --- Step 7: Handle outliers using the IQR method -----------------
        outlier_columns = ["monthly_household_income", "financial_need_score",
                           "commute_time_minutes", "weekly_work_hours"]
        capped = 0
        for col in outlier_columns:
            q1 = numpy.percentile(self.df[col], 25)
            q3 = numpy.percentile(self.df[col], 75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            for x in self.df.index:
                value = self.df.loc[x, col]
                if value < low:
                    self.df.loc[x, col] = low
                    capped = capped + 1
                elif value > high:
                    self.df.loc[x, col] = high
                    capped = capped + 1
        print("[7] Capped outlier values using IQR:", capped)

        self.is_cleaned = True
        print("-" * 60)
        print("Cleaning complete. Rows:", len(self.df),
              " Columns:", len(self.df.columns))
        print("-" * 60)

    def make_sure_cleaned(self):
        if self.df is None:
            print("Please load the dataset first (option 1).")
            return False
        if not self.is_cleaned:
            print("The data is not cleaned yet. Cleaning now...")
            self.clean_data()
        return True

    # -----------------------------------------------------------------------
    # 3. DISPLAY CLEANED DATA AND EXPORT
    # -----------------------------------------------------------------------
    def display_cleaned(self):
        if self.df is None:
            print("Please load the dataset first (option 1).")
            return
        self.clean_data()
        self.display_data("CLEANED DATASET")
        print()
        print("Column data types:")
        print(self.df.dtypes.to_string())
        self.export_cleaned()

    def export_cleaned(self):
        try:
            self.df.to_excel(self.output_file, index=False)
            print("Cleaned data exported to:", self.output_file)
        except Exception:
            print("Error: The cleaned file could not be saved.")

    # -----------------------------------------------------------------------
    # 4. COMPUTATIONS
    # -----------------------------------------------------------------------
    def perform_computations(self):
        if not self.make_sure_cleaned():
            return
        self.descriptive_statistics()
        self.sorting_example()
        self.filtering_example()
        self.grouping_example()
        self.totals_and_rankings()
        self.frequency_table()
        self.correlation_analysis()
        self.trend_analysis()

    def descriptive_statistics(self):
        print()
        print("=" * 60)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 60)
        columns = ["age", "monthly_household_income", "household_size",
                   "weekly_work_hours", "commute_time_minutes",
                   "monthly_grant_amount", "gpa", "units_enrolled",
                   "financial_need_score"]
        for col in columns:
            values = self.df[col]
            mean = numpy.mean(values)
            median = numpy.median(values)
            mode = values.mode()[0]
            std = numpy.std(values)
            smallest = numpy.min(values)
            largest = numpy.max(values)
            print()
            print("Column:", col)
            print("  Mean   : %.2f" % mean)
            print("  Median : %.2f" % median)
            print("  Mode   : %.2f" % mode)
            print("  Std Dev: %.2f" % std)
            print("  Minimum: %.2f" % smallest)
            print("  Maximum: %.2f" % largest)

    def sorting_example(self):
        print()
        print("=" * 60)
        print("SORTING")
        print("=" * 60)
        columns = ["student_id", "student_name", "ncr_area",
                   "financial_need_score", "gpa"]
        print()
        print("Top 10 students by Financial Need Score (highest need):")
        high_need = self.df.sort_values("financial_need_score",
                                        ascending=False)
        print(high_need.head(10)[columns].to_string(index=False))
        print()
        print("Top 10 students by GPA (1.0 is the best grade):")
        best_gpa = self.df.sort_values("gpa", ascending=True)
        print(best_gpa.head(10)[columns].to_string(index=False))

    def filtering_example(self):
        print()
        print("=" * 60)
        print("FILTERING")
        print("=" * 60)
        scholars = 0
        high_risk = 0
        working = 0
        for x in self.df.index:
            if self.df.loc[x, "scholarship_status"] == "Scholar":
                scholars = scholars + 1
            if self.df.loc[x, "risk_category"] == "High":
                high_risk = high_risk + 1
            if self.df.loc[x, "working_student"] == True:
                working = working + 1
        print("Scholars        :", scholars)
        print("High-risk        :", high_risk)
        print("Working students:", working)

    def grouping_example(self):
        print()
        print("=" * 60)
        print("GROUPING AND AGGREGATION (by NCR Area)")
        print("=" * 60)
        # Use dictionaries to count students and total the income per area.
        count = {}
        income_total = {}
        gpa_total = {}
        for x in self.df.index:
            area = self.df.loc[x, "ncr_area"]
            if area not in count:
                count[area] = 0
                income_total[area] = 0
                gpa_total[area] = 0
            count[area] = count[area] + 1
            income_total[area] = income_total[area] \
                + self.df.loc[x, "monthly_household_income"]
            gpa_total[area] = gpa_total[area] + self.df.loc[x, "gpa"]

        for area in count:
            average_income = income_total[area] / count[area]
            average_gpa = gpa_total[area] / count[area]
            print()
            print("Area:", area)
            print("  Students    :", count[area])
            print("  Avg Income  : %.2f" % average_income)
            print("  Avg GPA     : %.2f" % average_gpa)

    def totals_and_rankings(self):
        print()
        print("=" * 60)
        print("TOTALS, PERCENTAGES AND RANKINGS")
        print("=" * 60)
        total = len(self.df)
        applicants = 0
        scholars = 0
        working = 0
        for x in self.df.index:
            if self.df.loc[x, "scholarship_applied"] == True:
                applicants = applicants + 1
            if self.df.loc[x, "scholarship_status"] == "Scholar":
                scholars = scholars + 1
            if self.df.loc[x, "working_student"] == True:
                working = working + 1

        print("Total students :", total)
        print("Applicants     : %d (%.1f%%)"
              % (applicants, applicants / total * 100))
        print("Scholars       : %d (%.1f%%)"
              % (scholars, scholars / total * 100))
        if applicants > 0:
            print("Approval rate  : %.1f%%" % (scholars / applicants * 100))
        print("Working students: %d (%.1f%%)"
              % (working, working / total * 100))

        # Ranking of cities by approval rate, using dictionaries.
        city_applicants = {}
        city_scholars = {}
        for x in self.df.index:
            city = self.df.loc[x, "city"]
            if city not in city_applicants:
                city_applicants[city] = 0
                city_scholars[city] = 0
            if self.df.loc[x, "scholarship_applied"] == True:
                city_applicants[city] = city_applicants[city] + 1
            if self.df.loc[x, "scholarship_status"] == "Scholar":
                city_scholars[city] = city_scholars[city] + 1

        rates = {}
        for city in city_applicants:
            if city_applicants[city] > 0:
                rates[city] = city_scholars[city] / city_applicants[city] * 100
            else:
                rates[city] = 0
        ranking = pd.Series(rates).sort_values(ascending=False)
        print()
        print("City ranking by approval rate:")
        rank = 1
        for city in ranking.index:
            print("  %2d. %-14s %.1f%%" % (rank, city, ranking[city]))
            rank = rank + 1

    def frequency_table(self):
        print()
        print("=" * 60)
        print("FREQUENCY TABLE (Risk Category)")
        print("=" * 60)
        frequency = {}
        for x in self.df.index:
            risk = self.df.loc[x, "risk_category"]
            if risk not in frequency:
                frequency[risk] = 0
            frequency[risk] = frequency[risk] + 1

        total = len(self.df)
        for risk in frequency:
            percent = frequency[risk] / total * 100
            print("  %-10s %5d  (%.1f%%)" % (risk, frequency[risk], percent))

    def correlation_analysis(self):
        print()
        print("=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        print("(value near 1 = strong positive, near -1 = strong negative)")
        print()
        self.print_correlation("monthly_household_income",
                               "financial_need_score")
        self.print_correlation("financial_need_score", "monthly_grant_amount")
        self.print_correlation("weekly_work_hours", "financial_need_score")

    def print_correlation(self, column1, column2):
        # Compute the Pearson correlation coefficient using numpy.
        list1 = []
        list2 = []
        for x in self.df.index:
            list1.append(self.df.loc[x, column1])
            list2.append(self.df.loc[x, column2])

        mean1 = numpy.mean(list1)
        mean2 = numpy.mean(list2)
        top = 0
        bottom1 = 0
        bottom2 = 0
        i = 0
        while i < len(list1):
            diff1 = list1[i] - mean1
            diff2 = list2[i] - mean2
            top = top + (diff1 * diff2)
            bottom1 = bottom1 + (diff1 * diff1)
            bottom2 = bottom2 + (diff2 * diff2)
            i = i + 1
        r = top / numpy.sqrt(bottom1 * bottom2)
        print("  %s  vs  %s : %.2f" % (column1, column2, r))

    def trend_analysis(self):
        print()
        print("=" * 60)
        print("TREND ANALYSIS (Average Financial Need by Year Level)")
        print("=" * 60)
        need_total = {}
        count = {}
        for x in self.df.index:
            year = self.df.loc[x, "year_level"]
            if year not in count:
                need_total[year] = 0
                count[year] = 0
            need_total[year] = need_total[year] \
                + self.df.loc[x, "financial_need_score"]
            count[year] = count[year] + 1

        for year in sorted(count):
            average = need_total[year] / count[year]
            print("  Year %d : %.1f" % (year, average))

    # -----------------------------------------------------------------------
    # 5. CHARTS
    # -----------------------------------------------------------------------
    def generate_charts(self):
        if not self.make_sure_cleaned():
            return
        print("Generating charts... close each window to see the next one.")
        self.chart_bar()
        self.chart_line()
        self.chart_pie()
        self.chart_scatter()
        self.chart_histogram()
        print("All charts shown.")

    def chart_bar(self):
        # Count the scholars per NCR area using a dictionary.
        count = {}
        for x in self.df.index:
            if self.df.loc[x, "scholarship_status"] == "Scholar":
                area = self.df.loc[x, "ncr_area"]
                if area not in count:
                    count[area] = 0
                count[area] = count[area] + 1
        data = pd.Series(count)
        data.plot(kind="bar", color="blue")
        plt.title("Number of Scholars by NCR Area")
        plt.xlabel("NCR Area")
        plt.ylabel("Number of Scholars")
        plt.tight_layout()
        plt.show()

    def chart_line(self):
        # Average financial need per year level using dictionaries.
        need_total = {}
        count = {}
        for x in self.df.index:
            year = self.df.loc[x, "year_level"]
            if year not in count:
                need_total[year] = 0
                count[year] = 0
            need_total[year] = need_total[year] \
                + self.df.loc[x, "financial_need_score"]
            count[year] = count[year] + 1
        averages = {}
        for year in sorted(count):
            averages[year] = need_total[year] / count[year]
        data = pd.Series(averages)
        data.plot(kind="line", marker="o", color="red")
        plt.title("Average Financial Need Score by Year Level")
        plt.xlabel("Year Level")
        plt.ylabel("Average Financial Need Score")
        plt.tight_layout()
        plt.show()

    def chart_pie(self):
        # Count students per risk category using a dictionary.
        count = {}
        for x in self.df.index:
            risk = self.df.loc[x, "risk_category"]
            if risk not in count:
                count[risk] = 0
            count[risk] = count[risk] + 1
        data = pd.Series(count)
        data.plot(kind="pie", autopct="%1.1f%%")
        plt.title("Distribution of Students by Risk Category")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

    def chart_scatter(self):
        self.df.plot(kind="scatter", x="monthly_household_income",
                     y="financial_need_score", color="purple")
        plt.title("Household Income vs Financial Need Score")
        plt.xlabel("Monthly Household Income")
        plt.ylabel("Financial Need Score")
        plt.tight_layout()
        plt.show()

    def chart_histogram(self):
        self.df["gpa"].plot(kind="hist", bins=15, color="blue",
                            edgecolor="black")
        plt.title("Distribution of GPA (1.0 is the best grade)")
        plt.xlabel("GPA")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # 6. SUMMARY
    # -----------------------------------------------------------------------
    def output_summary(self):
        if not self.make_sure_cleaned():
            return
        total = len(self.df)
        applicants = 0
        scholars = 0
        income_total = 0
        gpa_total = 0
        for x in self.df.index:
            if self.df.loc[x, "scholarship_applied"] == True:
                applicants = applicants + 1
            if self.df.loc[x, "scholarship_status"] == "Scholar":
                scholars = scholars + 1
            income_total = income_total \
                + self.df.loc[x, "monthly_household_income"]
            gpa_total = gpa_total + self.df.loc[x, "gpa"]

        print()
        print("#" * 60)
        print("#  NCR SCHOLARSHIP ANALYTICS - SUMMARY")
        print("#" * 60)
        print("Total students    :", total)
        print("Applicants        : %d (%.1f%%)"
              % (applicants, applicants / total * 100))
        print("Scholars          : %d (%.1f%%)"
              % (scholars, scholars / total * 100))
        if applicants > 0:
            print("Approval rate     : %.1f%%" % (scholars / applicants * 100))
        print("Average income    : %.2f" % (income_total / total))
        print("Average GPA       : %.2f" % (gpa_total / total))
        print("#" * 60)

    # -----------------------------------------------------------------------
    # 7. MENU
    # -----------------------------------------------------------------------
    def run(self):
        print("Welcome to the NCR Scholarship Analytics application.")
        choice = ""
        while choice != "0":
            print()
            print("=" * 50)
            print("   NCR SCHOLARSHIP ANALYTICS - MAIN MENU")
            print("=" * 50)
            print("  1) Load datasets")
            print("  2) Display cleaned data")
            print("  3) Perform computations")
            print("  4) Generate charts automatically")
            print("  5) Output summarized results")
            print("  0) Exit")
            print("=" * 50)

            choice = input("Enter choice: ")

            if choice == "1":
                self.load_data()
            elif choice == "2":
                self.display_cleaned()
            elif choice == "3":
                self.perform_computations()
            elif choice == "4":
                self.generate_charts()
            elif choice == "5":
                self.output_summary()
            elif choice == "0":
                print("Goodbye!")
            else:
                print("Invalid choice. Please enter a number from the menu.")


# ===========================================================================
# START THE PROGRAM
# ===========================================================================
if __name__ == "__main__":
    program = ScholarshipAnalytics()
    program.run()

import pandas as pd

# Load the entire Excel workbook
file_path = "Dev/ApplicationCatalog.xlsx"
xlsx = pd.ExcelFile(file_path)

# Loop through each sheet
for sheet in xlsx.sheet_names:
    print(f"\n==================== SHEET: {sheet} ====================\n")
    
    # Read sheet
    df = pd.read_excel(file_path, sheet_name=sheet)
    
    # Show columns and dtypes
    print("Columns and Data Types:")
    print(df.dtypes)
    
    print("\n Unique Values (first 20 for each column):")
    for col in df.columns:
        print(f"\n--- {col} ---")
        try:
            print(df[col].unique()[:20])  # limit output
        except Exception as e:
            print(f"Could not display unique values: {e}")
    
    print("\n========================================================\n")

# Load your file
AppDescriptions = pd.read_excel("Dev/ApplicationCatalog.xlsx", sheet_name="Application Descriptions")

# Define the full list of business lines
all_business_lines = [
    "Global Wealth Management",
    "Retail Banking",
    "Canadian Banking",
    "US Banking",
    "Global Banking and Markets",
    "International Banking",
    "Global Finance"
]

# Function to expand rows that contain "All 7"
def expand_all_7(value):
    if pd.isna(value):
        return value
    text = str(value)

    # If it contains "All 7", replace the entire value with all business lines
    if "All 7" in text:
        return ", ".join(all_business_lines)

    return value

# Apply transformation
AppDescriptions["Business Line(s)"] = AppDescriptions["Business Line(s)"].apply(expand_all_7)

# Show result
print(AppDescriptions["Business Line(s)"].head())

# Export to a new Excel file
output_file = "ApplicationCatalog_Cleaned.xlsx"
AppDescriptions.to_excel(output_file, index=False)

print(f"File exported successfully: {output_file}")
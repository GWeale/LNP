import pandas as pd
import os
import re
import sys

def check_dependencies():
    try:
        import xlrd
    except ImportError:
        sys.exit(1)

def parse_specimen(specimen_rows):
    result = []
    specimen_name = specimen_rows[0]['Name']
    total_cells = specimen_rows[0]['#Cells']
    
    for row in specimen_rows:
        statistic = row.get('Statistic')
        if pd.notna(statistic):
            statistic_str = str(statistic)
            parsed_statistic = None
            
            if 'Mean : Comp-FITC-A =' in statistic_str:
                match = re.search(r'Mean : Comp-FITC-A = ([\d.]+)', statistic_str)
                if match:
                    parsed_statistic = float(match.group(1))
            elif statistic_str.replace('.', '', 1).isdigit():
                parsed_statistic = float(statistic_str)
            
            if parsed_statistic is not None:
                result.append({
                    'Specimen': specimen_name,
                    'Subset': row['Name'],
                    'Statistic': parsed_statistic,
                    'Count': row['#Cells'],
                    'Total_Cells': total_cells
                })
    
    return result

def process_excel_file(file_path):
    """Process a single Excel file and return a list of dictionaries."""
    try:
        df = pd.read_excel(file_path, header=None, engine='xlrd')
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

    print(f"Processing file: {file_path}")

    header_row = df[df.iloc[:, 0] == 'Depth'].index
    if len(header_row) == 0:
        print(f"Error: Could not find header row in {file_path}")
        return None
    
    header_row = header_row[0]
    
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row+1:].reset_index(drop=True)

    required_columns = ['Depth', 'Name', 'Statistic', '#Cells']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in {file_path}")
            return None

    specimens = []
    current_specimen = []
    
    for _, row in df.iterrows():
        row_dict = {col: row[col] for col in required_columns}
        if pd.isna(row['Depth']): 
            if current_specimen:
                specimens.extend(parse_specimen(current_specimen))
            current_specimen = [row_dict]
        else:
            current_specimen.append(row_dict)
    
    if current_specimen:
        specimens.extend(parse_specimen(current_specimen))
    
    return specimens

def main():
    check_dependencies()
    
    directory = '/Users/georgeweale/Downloads/Export Excel files 2'
    
    all_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.xls') and not filename.startswith('Summary'):
            file_path = os.path.join(directory, filename)
            data = process_excel_file(file_path)
            if data:
                all_data.extend(data)
    
    if not all_data:
        print("No valid data found in the Excel files.")
        return
    
    combined_df = pd.DataFrame(all_data)
    
    combined_df.to_csv('combined_flow_cytometry_data.csv', index=False)
    print("Data processing complete. Output saved to 'combined_flow_cytometry_data.csv'")

    print("\nFirst few rows of the combined data:")
    print(combined_df.head())

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import re

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('combined_flow_cytometry_data.csv')

# Ensure 'Specimen' and 'Subset' columns are strings
df['Specimen'] = df['Specimen'].astype(str)
df['Subset'] = df['Subset'].astype(str)

# Function to extract ratios from specimen names
def extract_ratio(specimen_name, pattern):
    match = re.search(pattern, specimen_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    else:
        return 0.0

# Function to extract the PEI, NP, and PBA ratios
def extract_ratios(specimen_name):
    # Patterns to match 'cp', 'NP', 'PBA' followed by a number
    pei_pattern = r'cp\s*([0-9]+\.?[0-9]*)'
    np_pattern = r'NP\s*([0-9]+\.?[0-9]*)'
    pba_pattern = r'PBA\s*([0-9]+\.?[0-9]*)'

    pei_ratio = extract_ratio(specimen_name, pei_pattern)
    np_ratio = extract_ratio(specimen_name, np_pattern)
    pba_ratio = extract_ratio(specimen_name, pba_pattern)

    return pei_ratio, np_ratio, pba_ratio

# Label specimens
def label_specimen(specimen_name):
    specimen_name_lower = specimen_name.lower()
    if any(keyword in specimen_name_lower for keyword in ['wt', 'water', 'blank', 'nc']):
        return 'NegControl'
    elif any(keyword in specimen_name_lower for keyword in ['l3k', 'lipo']):
        return 'PosControl'
    else:
        return 'Sample'

# Apply the labeling and ratio extraction functions
df['Label'] = df['Specimen'].apply(label_specimen)

# Extract ratios and add them as new columns
ratios = df['Specimen'].apply(extract_ratios)
df[['PEI Ratio', 'NP Ratio', 'PBA Ratio']] = pd.DataFrame(ratios.tolist(), index=df.index)

# Initialize an empty list to collect row data
rows_list = []

# Define the columns for the final table
columns = ['Specimen', 'PEI Ratio', 'NP Ratio', 'PBA Ratio',
           'Comp-Pacific Blue-A subset', 'q1', 'q2', 'q3', 'q4', 'Mean', 'After Mean']

# Group by Specimen
for specimen, group in df.groupby('Specimen'):
    # Initialize a dictionary to hold row data
    row_data = {'Specimen': specimen}
    
    # Get the ratios from the group (they are the same for all rows in the group)
    row_data['PEI Ratio'] = group['PEI Ratio'].iloc[0]
    row_data['NP Ratio'] = group['NP Ratio'].iloc[0]
    row_data['PBA Ratio'] = group['PBA Ratio'].iloc[0]
    
    # Get 'Statistic' values based on conditions
    row_data['Comp-Pacific Blue-A subset'] = group.loc[group['Subset'].str.contains('Comp-Pacific Blue-A subset', case=False), 'Statistic'].values
    row_data['q1'] = group.loc[group['Subset'].str.contains('q1', case=False), 'Statistic'].values
    row_data['q2'] = group.loc[group['Subset'].str.contains('q2', case=False), 'Statistic'].values
    row_data['q3'] = group.loc[group['Subset'].str.contains('q3', case=False), 'Statistic'].values
    row_data['q4'] = group.loc[group['Subset'].str.contains('q4', case=False), 'Statistic'].values
    row_data['Mean'] = group.loc[group['Subset'].str.contains('mean', case=False), 'Statistic'].values

    # Get the index of the 'mean' row and fetch the 'Statistic' from the next row
    mean_indices = group.index[group['Subset'].str.contains('mean', case=False)].tolist()
    if mean_indices:
        next_index = mean_indices[0] + 1
        if next_index in group.index:
            row_data['After Mean'] = group.loc[next_index, 'Statistic']
        else:
            row_data['After Mean'] = np.nan
    else:
        row_data['After Mean'] = np.nan
    #implements new methods


    # Flatten arrays and handle missing data
    for key in ['Comp-Pacific Blue-A subset', 'q1', 'q2', 'q3', 'q4', 'Mean']:
        if isinstance(row_data[key], np.ndarray) and len(row_data[key]) > 0:
            row_data[key] = row_data[key][0]
        else:
            row_data[key] = np.nan

    # Append the row data to the list
    rows_list.append(row_data)

# Create the final DataFrame from the list of row data
final_table = pd.DataFrame(rows_list, columns=columns)

# Display the final table
print(final_table)

# Optional: Save the final table to a CSV file
final_table.to_csv('flow_cytometry_summary.csv', index=False)

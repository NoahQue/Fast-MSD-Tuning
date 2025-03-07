import pandas as pd
import os
# Read all CSV files
folder_path = "./fast_gain_len4"
file_paths = [
    "result_sequence_0-90.csv",
    "result_sequence_90-180.csv",
    "result_sequence_180-270.csv",
    "result_sequence_270-360.csv"
]

# Create an empty DataFrame to merge data
merged_df = pd.DataFrame()

# Iterate through file paths, read each CSV file, and append it to the merged DataFrame
for file in file_paths:
    df = pd.read_csv(os.path.join(folder_path, file), dtype={"Task Sequence": str})
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# Remove duplicate entries based on the 'Task Sequence' column, keeping the first occurrence
merged_df = merged_df.drop_duplicates(subset='Task Sequence', keep='first')

# Write the merged DataFrame to a new CSV file
merged_df.to_csv(os.path.join(folder_path, "all_result_sequence_temp.csv"), index=False)

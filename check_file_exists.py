import csv
import os

# Set the path to your CSV file
csv_path = "./data/test_org.csv"
filtered_csv_path = "./data/test_org_filtered.csv"

# List to store valid file paths
valid_file_paths = []

# Open the original CSV, read paths, and check their existence
with open(csv_path, mode="r", newline="") as file:
    reader = csv.reader(file)
    path_count = 0
    for row in reader:
        file_path = row[0]  # Assuming each row contains one file path
        path_count += 1
        # Check if the file exists
        if os.path.exists(f"./data/{file_path}"):
            valid_file_paths.append(
                [file_path]
            )  # Append as a list to match CSV writer format

    print(f"{len(valid_file_paths)}/{path_count} files exist")

# Write the valid file paths back to a new CSV file
with open(filtered_csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(valid_file_paths)

print(
    f"Checked and filtered file paths. Valid paths are saved to '{filtered_csv_path}'"
)

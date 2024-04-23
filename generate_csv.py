import os
import pandas as pd

# Set the base directory where the subdirectories are located
base_dir = "wav48"

# List and sort all subdirectories in the base directory
subdirectories = sorted(
    [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
)

# Split into train and test
train_dirs = subdirectories[:100]
test_dirs = subdirectories[100:]


# Function to get all wav file paths in a list of directories
def get_file_paths(directories):
    file_paths = []
    for directory in directories:
        full_dir_path = os.path.join(base_dir, directory)
        for file in os.listdir(full_dir_path):
            if file.endswith(".wav"):
                file_paths.append(os.path.join(base_dir, directory, file))
    return file_paths


if __name__ == "__main__":
    # Get file paths for train and test
    train_file_paths = get_file_paths(train_dirs)
    test_file_paths = get_file_paths(test_dirs)

    # Convert lists to DataFrame
    train_df = pd.DataFrame(train_file_paths, columns=["file_path"])
    test_df = pd.DataFrame(test_file_paths, columns=["file_path"])

    # Save to CSV
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

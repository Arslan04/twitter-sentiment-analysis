import pandas as pd

# Load the dataset
file_path = "D:/7TH SEM/Twitter sentimental analysis/TWITTER/training.1600000.processed.noemoticon.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path, header=None, encoding='latin1')

# Filter rows based on the first column (assuming column 0 has 0 and 4 values)
data_0 = data[data[0] == 0]
data_4 = data[data[0] == 4]

# Randomly sample 5000 rows from each class
sampled_0 = data_0.sample(n=5000, random_state=42)
sampled_4 = data_4.sample(n=5000, random_state=42)

# Combine the samples
balanced_data = pd.concat([sampled_0, sampled_4]).sample(frac=1, random_state=42)  # Shuffle the data

# Save the balanced data to a new CSV file
output_path = "balanced_data.csv"  # Replace with your desired output file path
balanced_data.to_csv(output_path, index=False, header=False)

print(f"Balanced data saved to {output_path}")

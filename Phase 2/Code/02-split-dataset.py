import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset/02-prepared/covid-dataset.csv')

used_percent = 0.001

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=1-used_percent, random_state=42)
train_df , test_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Save the resulting datasets to CSV files
train_df.to_csv('dataset/03-splited/train.csv', index=False)
test_df.to_csv('dataset/03-splited/test.csv', index=False)
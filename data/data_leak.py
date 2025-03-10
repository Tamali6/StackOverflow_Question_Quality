import pandas as pd

# Load the datasets
train_df = pd.read_csv("stack_overflow_questions_train.csv")
test_df = pd.read_csv("stack_overflow_questions_test.csv")

# Merge both datasets to find common rows
common_rows = train_df.merge(test_df, how="inner")

# Check if there are any common rows
if not common_rows.empty:
    print("There are common rows between train and test data.")
    print(common_rows)
else:
    print("No common rows found between train and test data.")


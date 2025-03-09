import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os

class StackOverflowDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load the dataset
            self.data = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_columns = ['Title', 'Body', 'Y']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Combine 'Title' and 'Body' to create 'text'
            self.data['text'] = self.data['Title'] + " " + self.data['Body']
            
            # Ensure there are no missing values in 'Type' column
            if self.data['Y'].isnull().any():
                raise ValueError("The 'Y' column contains missing values. Please clean the data.")
            
            # Convert 'Type' to categorical labels
            self.labels = self.data['Y'].astype('category').cat.codes

            # Initialize tokenizer and max length
            self.tokenizer = tokenizer
            self.max_length = max_length

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: The file is empty: {file_path}")
            raise
        except pd.errors.ParserError:
            print(f"Error: There was an error parsing the file: {file_path}")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error while loading the dataset: {e}")
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            # Tokenize the text
            encoding = self.tokenizer(
                self.data['text'][idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Return tokenized inputs and label
            return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(self.labels[idx])

        except KeyError as e:
            print(f"Error: Missing key during tokenization, {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during tokenization: {e}")
            raise


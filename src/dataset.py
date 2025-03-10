import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
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
            self.data['text'] = self.data['Title'].fillna('') + " " + self.data['Body'].fillna('')
            
            # Ensure there are no missing values in 'Y' column
            if self.data['Y'].isnull().any():
                raise ValueError("The 'Y' column contains missing values. Please clean the data.")
            
            # Convert 'Y' to categorical labels using LabelEncoder
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.data['Y'])
            
            # Initialize tokenizer and max length
            self.tokenizer = tokenizer
            self.max_length = max_length

        except FileNotFoundError as e:
            raise e  # Let the caller handle this
        except pd.errors.EmptyDataError:
            raise ValueError(f"Error: The file is empty: {file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error: There was an error parsing the file: {file_path}")
        except ValueError as e:
            raise e  # Let the caller handle this
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading the dataset: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            # Tokenize the text
            encoding = self.tokenizer(
                self.data['text'][idx],
                truncation=True,
                padding='max_length',  # Ensure padding to max_length
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Return tokenized inputs and label
            return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor(self.labels[idx])

        except KeyError as e:
            raise KeyError(f"Error: Missing key during tokenization: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during tokenization: {e}")


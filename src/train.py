import torch
import torch.optim as optim
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import StackOverflowDataset
from model import BERTClassifier

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import BERTClassifier
from dataset import StackOverflowDataset


try:
    # Load config
    print("Loading configuration...")
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: Configuration file not found: {e}")
    raise

except yaml.YAMLError as e:
    print(f"Error: YAML format error: {e}")
    raise

except Exception as e:
    print(f"Unexpected error while loading the configuration: {e}")
    raise



try:
    # Initialize device (GPU or CPU)
    device = torch.device(config["device"])
    print(f"Device set to {device}")

except KeyError as e:
    print(f"Error: Missing key in configuration: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while setting device: {e}")
    raise


try:
    # Load the tokenizer
    print(f"Loading tokenizer from {config['model']['pretrained_model']}...")
    tokenizer = BertTokenizer.from_pretrained(config["model"]["pretrained_model"])
    print("Tokenizer loaded successfully.")

except KeyError as e:
    print(f"Error: Missing key in configuration: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while loading tokenizer: {e}")
    raise



try:
    # Load dataset
    print(f"Loading dataset from {config['data_path'] + config['train_file']}...")
    train_dataset = StackOverflowDataset(config["data_path"] + config["train_file"], tokenizer, config["model"]["max_length"])
    print('Dataset loaded successfully.')

except FileNotFoundError as e:
    print(f"Error: Dataset file not found: {e}")
    raise
except ValueError as e:
    print(f"Error: Invalid data in dataset: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while loading dataset: {e}")
    raise



try:
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)
    print(f"DataLoader created with batch size {config['model']['batch_size']}.")

except KeyError as e:
    print(f"Error: Missing key in configuration: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while creating DataLoader: {e}")
    raise


try:
    # Load pretrained model
    print(f"Loading model {config['model']['pretrained_model']}...")
    model = BERTClassifier(config["model"]["pretrained_model"], config["model"]["num_labels"]).to(device)
    print("Model loaded successfully.")

except KeyError as e:
    print(f"Error: Missing key in configuration: {e}")
    raise
except Exception as e:
    print(f"Unexpected error while loading model: {e}")
    raise
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["model"]["learning_rate"])

# Training Loop
for epoch in range(config["model"]["epochs"]):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Save model
model.bert.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])


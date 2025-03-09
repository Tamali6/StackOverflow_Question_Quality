import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import StackOverflowDataset
from src.util import load_config

# Load configuration
config = load_config()

device = torch.device(config["device"])
tokenizer = BertTokenizer.from_pretrained(config["output_dir"])

# Load test dataset
test_dataset = StackOverflowDataset(config["data_path"] + config["test_file"], tokenizer, config["model"]["max_length"])
test_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"])

# Load model
model = BertForSequenceClassification.from_pretrained(config["output_dir"], num_labels=config["model"]["num_labels"])
model.to(device)
model.eval()

# Evaluate
correct, total = 0, 0
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")


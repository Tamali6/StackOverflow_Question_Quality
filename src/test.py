import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from dataset import StackOverflowDataset
from model import BERTClassifier
from util import load_config
import pandas as pd

# Load configuration
config = load_config()

# Select device
device = torch.device(config["device"])

# Determine which tokenizer to load based on the model configuration
if config["model"]["eval_model"] == "pretrained":
    tokenizer_name = config["model"]["pretrained_model"]
else:
    tokenizer_name = config["output_dir"]

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# Load train and test datasets
train_df = pd.read_csv(config["data_path"] + config["train_file"])
test_df = pd.read_csv(config["data_path"] + config["test_file"])

# Check for overlap
common_rows = train_df.merge(test_df, how="inner")
if not common_rows.empty:
    print("Warning: There are common rows between train and test data. Evaluation aborted.")
else:
    print("No common rows found. Proceeding with evaluation.")
    
    # Load test dataset
    test_dataset = StackOverflowDataset(config["data_path"] + config["test_file"], tokenizer, config["model"]["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"])

    # Determine which model to load
    if config["model"]["eval_model"] == "pretrained":
        model_name = config["model"]["pretrained_model"]
    else:
        model_name = config["output_dir"]
    
    # Load model
    model = BERTClassifier(model_name, config["model"]["num_labels"])
    model.to(device)
    model.eval()

    # Evaluate
    correct, total = 0, 0
    positive_sample_outputs = []
    negative_sample_outputs = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Analyze output if enabled in config
            if config.get("analyse_output", False):
                for i in range(len(predictions)):
                    decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    
                    # Use the updated label encoding mapping from the LabelEncoder
                    category_mapping = dict(enumerate(test_dataset.label_encoder.classes_))
                    predicted_label = category_mapping[predictions[i].item()]
                    actual_label = category_mapping[labels[i].item()]

                    if predictions[i] != labels[i]:
                        if len(negative_sample_outputs) < 3:
                            negative_sample_outputs.append({
                                'Text': decoded_text,
                                'Predicted': predicted_label,
                                'Actual': actual_label
                            })
                    else:
                        if len(positive_sample_outputs) < 3:
                            positive_sample_outputs.append({
                                'Text': decoded_text,
                                'Predicted': predicted_label,
                                'Actual': actual_label
                            })

    # Print the accuracy
    print(f"Test Finetuned Accuracy: {correct / total:.4f}")

    # Display examples if enabled
    if config.get("analyse_output", False):
        print("\nPositive Sample Outputs:")
        for example in positive_sample_outputs:
            print(f"\nText: {example['Text']}\nPredicted: {example['Predicted']}\nActual: {example['Actual']}")
        
        print("\nNegative Sample Outputs:")
        for example in negative_sample_outputs:
            print(f"\nText: {example['Text']}\nPredicted: {example['Predicted']}\nActual: {example['Actual']}")


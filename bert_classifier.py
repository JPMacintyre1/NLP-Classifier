import os
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast






# Load your dataset (replace with actual file paths or data sources)
train_data = pd.read_csv("bert_data\\train.csv")
dev_data = pd.read_csv("bert_data\\dev.csv")
test_data = pd.read_csv("bert_data\\test.csv")

train_data["label"] = train_data["label"].astype(int)
dev_data["label"] = dev_data["label"].astype(int)
test_data["label"] = test_data["label"].astype(int)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
dev_dataset = Dataset.from_pandas(dev_data)
test_dataset = Dataset.from_pandas(test_data)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dev_dataset = dev_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Use a data collator for proper batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)
dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)

# Load the saved model and tokenizer
saved_model_dir = "saved_cased_model"
model = BertForSequenceClassification.from_pretrained(saved_model_dir)
tokenizer = BertTokenizer.from_pretrained(saved_model_dir)

# Move the model to the appropriate device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Example: Retest on the test dataset
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_predictions))



"""
# Load your dataset (replace with actual file paths or data sources)
train_data = pd.read_csv("bert_data\\train.csv")
dev_data = pd.read_csv("bert_data\\dev.csv")
test_data = pd.read_csv("bert_data\\test.csv")

train_data["label"] = train_data["label"].astype(int)
dev_data["label"] = dev_data["label"].astype(int)
test_data["label"] = test_data["label"].astype(int)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
dev_dataset = Dataset.from_pandas(dev_data)
test_dataset = Dataset.from_pandas(test_data)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dev_dataset = dev_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")  # Rename 'label' to 'labels'
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Assuming binary classification (num_labels=2)
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# Use a data collator for proper batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)
dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)



optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

epochs = 3
progress_bar = tqdm(range(num_training_steps))

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type):
            outputs = model(**batch)
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
    lr_scheduler.step()

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in dev_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_predictions))




# Define a directory to save the model
output_dir = "saved_cased_model"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

"""




















"""
def get_target_values(data_set):
    output_id = []
    output_rating = []
    for doc_id in data_set:
        rating = doc_id.split('_')[1].split('.')[0]
        output_id.append(doc_id)
        output_rating.append(rating)
    return output_id, output_rating

def read_data(data_set, data_set_rating, data_path):
    output_set = [("label", "text")]
    for rating_id, id in enumerate(data_set):
        if int(data_set_rating[rating_id]) < 5:
            with open(data_path + "neg\\" + id, "r", encoding="utf8") as f:
                output_set.append((0, f.read()))
        else:
            with open(data_path + "pos\\" + id, "r", encoding="utf8") as f:
                output_set.append((1, f.read()))
    return output_set

# path for input text files
data_path = "data\\data\\"

# separate into positive and negative
neg_list = os.listdir(data_path + "neg")
pos_list = os.listdir(data_path + "pos")

all_reviews = neg_list + pos_list

all_doc_id, all_doc_rating = get_target_values(all_reviews)  # Retrieve all document ID's and associated ratings in 2 lists

train_id, X_temp, train_ratings, y_temp = train_test_split(all_doc_id, all_doc_rating, test_size=0.2, random_state=42, stratify=all_doc_rating) # Split to training and temp

dev_id, test_id, dev_ratings, test_ratings = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # Split temp into dev and test

# Populate data sets
training_set = read_data(train_id, train_ratings, data_path)
dev_set = read_data(dev_id, dev_ratings, data_path)
test_set = read_data(test_id, test_ratings, data_path)

with open("bert_data\\train.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(training_set)

with open("bert_data\\dev.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(dev_set)

with open("bert_data\\test.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(test_set)
"""









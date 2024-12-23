import json

from torch import nn
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


class DifficultyModel(nn.Module):
    def __init__(self, base, vocab_size, num_classes=1):
        super(DifficultyModel, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, num_classes, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_answer = self.data_js[idx]['prompt_answer']
        label = self.data_js[idx]['label']

        encoded_pair = self.tokenizer.encode_plus(
            prompt_answer,
            padding='max_length',
            max_length=max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'label': label
        }


def read_jsonl(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

# Load training set, validation set, and test set data
train_js = 'data/train_en.json'
test_js = 'data/test_en.json'
val_js = 'data/valid_en.json'
train_json = read_jsonl(train_js)  # This section uses a CSV file as an example to describe how to load data
val_json = read_jsonl(val_js)
test_json = read_jsonl(test_js)

# Load the pre-trained ChatGLM3-6b model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/ckpt/chatglm3-6b", trust_remote_code=True)
base_model = AutoModel.from_pretrained("/workspace/ckpt/chatglm3-6b",
                                       trust_remote_code=True).bfloat16().cuda()

# Create a custom dataset
train_dataset = MyDataset(train_json, tokenizer)
val_dataset = MyDataset(val_json, tokenizer)
test_dataset = MyDataset(test_json, tokenizer)

# Create data loaders
batch_size = 3  # Set batch size
max_length = 1024
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, '\n')
vocab_size = base_model.config.padded_vocab_size
print(vocab_size)
DM = DifficultyModel(base_model, vocab_size, 1)

DM.to(device)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = AdamW(DM.parameters(), lr=3e-6)
num_epochs = 2
# Training and validation loop
best_val_loss = 10000000
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    print(f"{epoch}/{num_epochs} training")
    # Training
    DM.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].bfloat16().to(device)

        optimizer.zero_grad()
        outputs = DM(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    DM.eval()
    val_loss = 0.0
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].bfloat16().to(device)
            outputs = DM(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_labels.extend(labels.tolist())

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(DM.state_dict(), "checkpoints/VM_best_checkpoint.pt")

print("Training complete!")

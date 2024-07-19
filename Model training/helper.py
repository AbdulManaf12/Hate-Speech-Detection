import re
import os
import csv
import nltk
import torch
import string
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

nltk.download('stopwords')

with open('../resources/stopwords-ur.txt', 'r', encoding='utf-8') as f:
    urdu_stopwords = set(f.read().splitlines())
    
with open('../resources/stopwords-sd.txt', 'r', encoding='utf-8') as f:
    sindhi_stopwords = set(f.read().splitlines())

# with open('/content/drive/MyDrive/Hate Speech_Multilingual /Code/Model training/Hate_VS_Non-Hate/resources/stopwords-ur.txt', 'r', encoding='utf-8') as f:
#     urdu_stopwords = set(f.read().splitlines())
    
# with open('/content/drive/MyDrive/Hate Speech_Multilingual /Code/Model training/Hate_VS_Non-Hate/resources/stopwords-sd.txt', 'r', encoding='utf-8') as f:
#     sindhi_stopwords = set(f.read().splitlines())

### Data Preprocessing and Exploration ###
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        if self.labels.shape == (self.labels.shape[0], ):
            self.type = 'binary'
        else:
            self.type = 'multi'

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.type == 'binary':
            temp_type = torch.tensor(label, dtype=torch.long)
        else:
            temp_type = torch.tensor(label, dtype=torch.float)
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': temp_type
        }

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )

def preprocess(texts, tokenizer, max_len):
    encodings = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encodings

def preprocess_text(text, language='english'):
    text = re.sub(r'@user', '', text)
    text = re.sub(r'#(\w+)', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()

    if language == 'english':
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    elif language == 'urdu':
        stop_words = urdu_stopwords
    elif language == 'sindhi':
        stop_words = sindhi_stopwords
    else:
        raise ValueError(f"Unsupported language: {language}")

    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text.strip()

def shuffle_data(texts, labels):
    indices = np.arange(len(texts))
    np.random.shuffle(indices)
    return [texts[i] for i in indices], [labels[i] for i in indices]
### End of Data Preprocessing and Exploration ###

### Model Loading and Saving ###
def save_model(model, tokenizer, path, epoch):
    model_save_path = os.path.join(path, f'model_epoch_{epoch}')
    tokenizer_save_path = os.path.join(path, 'tokenizer')

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(tokenizer_save_path):
        os.makedirs(tokenizer_save_path)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f'Model and tokenizer saved at epoch {epoch}')

### Model Evaluation ###
def log_metrics_to_csv(path, epoch, train_acc, train_loss, val_acc, val_loss):
    csv_path = os.path.join(path, 'training_log.csv')

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Validation Accuracy', 'Validation Loss'])
        writer.writerow([epoch, train_acc, train_loss, val_acc, val_loss])
    print(f'Metrics logged at epoch {epoch}')

def load_training_history(csv_path):
    if not os.path.isfile(csv_path):
        return {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}, 0

    df = pd.read_csv(csv_path)
    history = {
        'train_acc': df['Train Accuracy'].tolist(),
        'train_loss': df['Train Loss'].tolist(),
        'val_acc': df['Validation Accuracy'].tolist(),
        'val_loss': df['Validation Loss'].tolist()
    }
    last_epoch = df['Epoch'].max()
    return history, last_epoch

### Model Training ###
def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0
    # Wrap data_loader with tqdm for a progress bar
    progress_bar = tqdm(data_loader, desc="Training Progress", total=len(data_loader))
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update the progress bar with the latest loss
        progress_bar.set_postfix({'loss': loss.item()})

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels

def train_model(train_data_loader, val_data_loader, model, optimizer, device, epochs, tokenizer, path, start_epoch=0):
    history, last_epoch = load_training_history(os.path.join(path, 'training_log.csv'))

    best_val_acc = 0.0 
    early_stopping_patience = 5 
    no_improvement_epochs = 0 

    if start_epoch == 0:
        start_epoch = last_epoch + 1

    for epoch in range(start_epoch, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss, _, _ = eval_model(model, val_data_loader, device)
        print(f'Validation loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, tokenizer, path, epoch)  
            print(f"Checkpoint saved: Improved validation accuracy at epoch {epoch}: {val_acc}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation accuracy for {no_improvement_epochs} epochs.")

        if no_improvement_epochs >= early_stopping_patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

        log_metrics_to_csv(path, epoch, train_acc, train_loss, val_acc, val_loss)
    return history
### End of Model Training ###

#### train multi model training ###
def train_epoch_multi(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training Progress", total=len(data_loader))
    for d in progress_bar:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["label"].to(device)  

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits
        preds = (logits.sigmoid() > 0.5).int()  
        
        loss.backward()
        optimizer.step()

        correct_predictions += (preds == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)
        losses.append(loss.item())

        progress_bar.set_postfix({'loss': loss.item()})

    train_accuracy = correct_predictions / total_samples
    average_loss = np.mean(losses)
    return train_accuracy, average_loss


def train_model_multi(train_data_loader, val_data_loader, model, optimizer, device, epochs, tokenizer, path, start_epoch=0):
    history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    last_epoch = start_epoch
    
    best_val_acc = 0.0
    early_stopping_patience = 5
    no_improvement_epochs = 0

    for epoch in range(start_epoch, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch_multi(model, train_data_loader, optimizer, device)
        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss, _, _ = eval_model_multi(model, val_data_loader, device)  
        print(f'Validation loss {val_loss} accuracy {val_acc}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, tokenizer, path, epoch)  
            print(f"Checkpoint saved: Improved validation accuracy at epoch {epoch}: {val_acc}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            print(f"No improvement in validation accuracy for {no_improvement_epochs} epochs.")

        if no_improvement_epochs >= early_stopping_patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

    return history

def eval_model_multi(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["label"].to(device)  

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            preds = (logits.sigmoid() > 0.5).int()  

            losses.append(loss.item())
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    validation_accuracy = correct_predictions / total_samples
    average_loss = np.mean(losses)
    return validation_accuracy, average_loss, all_preds, all_labels
### End of Model Evaluation ###

### Model prediction ###
def predict(texts, model, tokenizer, max_len, device):
    model = model.eval()
    encodings = preprocess(texts, tokenizer, max_len)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    logits = outputs.logits
    preds = logits.argmax(dim=1).cpu().numpy()
    softmax_probs = torch.softmax(logits, dim=1).cpu().numpy()

    return preds, softmax_probs

## plot confusion matrix
def plot_history(history):
    train_acc = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in history['train_acc']]
    val_acc = [acc.item() if isinstance(acc, torch.Tensor) else acc for acc in history['val_acc']]
    train_loss = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in history['train_loss']]
    val_loss = [loss.item() if isinstance(loss, torch.Tensor) else loss for loss in history['val_loss']]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy', marker='o')
    plt.plot(val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    print("Accuracy:", round(accuracy_score(y_true, y_pred),2))
    print("F1 Score:", round(f1_score(y_true, y_pred, average='weighted'), 2))
    print("Precision:", round(precision_score(y_true, y_pred, average='weighted'), 2))
    print("Recall:", round(recall_score(y_true, y_pred, average='weighted'), 2))
    print()
    print("Classification Report:")
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    if len(labels) == 2:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    else:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
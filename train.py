from parser import CustomDataset, get_data
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sermodel import SERModel, SERModelOneLL
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch.optim
import warnings
import numpy as np

# Warning about nested tensors
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

def train_model(model, train_loader, val_loader, num_epochs, optimizer, device, loss_weight):
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    model.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        preds = []
        labels = []

        for batch in train_loader:
            inputs, label, mask = batch

            inputs = inputs.to(device)
            label = label.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, padding_mask=mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            y_pred = outputs.argmax(dim=1)
            preds.extend(y_pred.cpu().tolist())
            labels.extend(label.cpu().tolist())

        average_loss = total_loss / len(train_loader.dataset)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)

        cm = confusion_matrix(labels, preds)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(cm)
        # Validation
        val_loss = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)


def evaluate_model(model, data_loader, criterion, device, test=False):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, label, mask = batch

            inputs = inputs.to(device)
            label = label.to(device)
            mask = mask.to(device)

            outputs = model(inputs, padding_mask=mask)
            loss = criterion(outputs, label)
            total_loss += loss.item() * inputs.size(0)
            
            y_pred = outputs.argmax(dim=1)
            preds.extend(y_pred.cpu().tolist())
            labels.extend(label.cpu().tolist())

    loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    if test:
        print(f"Test loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print(f"Validation loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(cm)
    return loss
    
    
            

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data()
    model = SERModel(d_model=512, nhead=2, hidden_dim = 100, num_layers=1, dropout=.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=.01)

    for batch in train_loader:
        _, labels, _ = batch
    labels = np.array(list(labels), dtype=np.int64)
    class_count = np.bincount(labels)
    class_weights = 1.0 / class_count
    class_weights /= class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("Two linear layers:")
    train_model(model, train_loader, val_loader, num_epochs=35, optimizer=optimizer, device=device, loss_weight=class_weights_tensor)
    evaluate_model(model, test_loader, nn.CrossEntropyLoss(class_weights_tensor), device=device, test=True)

    model = SERModelOneLL(d_model=512, nhead=2, hidden_dim = 100, num_layers=1, dropout=.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=.01)

    print("\nOne linear layer:")
    train_model(model, train_loader, val_loader, num_epochs=35, optimizer=optimizer, device=device, loss_weight=class_weights_tensor)
    evaluate_model(model, test_loader, nn.CrossEntropyLoss(class_weights_tensor), device=device, test=True)
    
    # match label:
    #    case 0:
    #        label = "anger"
    #    case 1:
    #        label = "disgust"
    #    case 2:
    #        label = "fear"
    #    case 3:
    #        label = "happy"
    #    case 4:
    #        label = "neutral"
    #    case 5:
    #        label = "sad"
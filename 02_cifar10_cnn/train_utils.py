import torch
from sklearn.model_selection import KFold, StratifiedKFold
from utils import get_model, get_optimizer

def train(epochs, patience, device, train_data, val_data, model, loss_function, optimizer):
    best_loss = float('inf')
    count = 0
    val_sum_loss = 0
    val_sum_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for images, labels in train_data:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            #update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_size = labels.size(0)
            total += batch_size

            #loss
            total_loss += loss.item() * batch_size #loss.item() is batch mean
            
            #accuracy
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()

        train_loss = total_loss/total
        train_acc = correct/total * 100
        val_loss, val_acc = evaluate(device, val_data, model, loss_function)
        
        val_sum_loss += val_loss
        val_sum_acc += val_acc

        if best_loss > val_loss:
            best_loss = val_loss
            count = 0
            #torch.save(model.state_dict(), "best.pth")
        else:
            count += 1

        if count >= patience:
            print("Early Stopping\n")
            break

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        print("---------------------------------------------------")

    return val_sum_loss/(epoch+1), val_sum_acc/(epoch+1)

def evaluate(device, data, model, loss_function):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            batch_size = labels.size(0)
            total += batch_size

            #loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * batch_size

            #accuracy
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()
            
    accuracy = correct/total * 100
    avg_loss = total_loss/total

    return avg_loss, accuracy

def cross_validate(full_train_datasets, epochs, patience, device, model_name, loss_function, 
                   opt_name, n_splits=5, batch_size=64, random_state=42):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_sum_loss = 0
    val_sum_acc = 0

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_train_datasets.data, full_train_datasets.targets)):
        model = get_model(model_name).to(device)
        optimizer = get_optimizer(opt_name, model)
        
        train_datasets = torch.utils.data.Subset(full_train_datasets, train_indices)
        val_datasets = torch.utils.data.Subset(full_train_datasets, val_indices)

        train_data = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle = True)
        val_data = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = False)

        val_loss, val_acc = train(epochs, patience, device, train_data, val_data, model, loss_function, optimizer)

        val_sum_loss += val_loss
        val_sum_acc += val_acc

    print(f"Val Mean Loss: {val_sum_loss / n_splits:.4f}")
    print(f"Val Mean Acc: {val_sum_acc / n_splits:.2f}%")

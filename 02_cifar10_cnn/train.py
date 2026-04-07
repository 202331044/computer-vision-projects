import torch
from sklearn.model_selection import KFold, StratifiedKFold
from utils import get_model, get_optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR

def train(train_data, val_data, model, loss_function, device, optimizer, epochs=100, patience=7):
    
    best_loss = float('inf')
    count = 0
    val_total_loss = 0
    val_total_acc = 0

    # scheduler = StepLR(optimizer, step_size=10, gamma=0.01)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_data), epochs=epochs)
    
    for epoch in range(epochs):
        
        model.train()
        
        train_total_loss = 0
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
            #scheduler.step() #OneCycleLR
            
            batch_size = labels.size(0)
            total += batch_size

            #loss
            train_total_loss += loss.item() * batch_size
            
            #accuracy
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()

        val_loss, val_acc = evaluate(val_data, model, loss_function, device)
        
        #scheduler.step() #stepLR, CosineAnnealingLR
    
        val_total_loss += val_loss
        val_total_acc += val_acc

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_total_loss/total:.4f} Train Acc: {correct/total * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        print("---------------------------------------------------")

        if best_loss > val_loss:
            best_loss = val_loss
            count = 0
            #torch.save(model.state_dict(), "best.pth")
        else:
            count += 1

        if count >= patience:
            print("Early Stopping\n")
            break

    return val_total_loss/(epoch+1), val_total_acc/(epoch+1)

def evaluate(data, model, loss_function, device):

    model.eval()

    total_loss = 0
    total = 0
    correct = 0

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
            
    avg_loss = total_loss/total
    accuracy = correct/total * 100

    return avg_loss, accuracy

def cross_validate(datasets, model_name, loss_function, device, batch_size=64,
                n_splits=5, epochs=100, patience=7, opt_name="Adam"):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_total_loss = 0
    val_total_acc = 0

    for fold, (train_indices, val_indices) in enumerate(skf.split(datasets.data, datasets.targets)):
        
        model = get_model(model_name).to(device)
        optimizer = get_optimizer(opt_name, model)

        train_datasets = torch.utils.data.Subset(datasets, train_indices)
        val_datasets = torch.utils.data.Subset(datasets, val_indices)

        train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
        val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

        print(f"-------fold {fold+1}-------")
        
        val_loss, val_acc = train(train_data, val_data, model, loss_function, device, optimizer, epochs, patience)

        val_total_loss += val_loss
        val_total_acc += val_acc

    print(f"Val Mean Loss: {val_total_loss / n_splits:.4f}")
    print(f"Val Mean Acc: {val_total_acc / n_splits:.2f}%")

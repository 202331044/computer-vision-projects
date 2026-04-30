import torch
from sklearn.model_selection import KFold, StratifiedKFold
import utils as u
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
import numpy as np

def train(train_data, val_data, model, loss_function, device, optimizer, epochs=10, patience=5, is_early_stopping=False):
    
    train_loss = 0
    train_acc = 0

    best_loss = float('inf')
    acc_at_best_loss = 0

    loss_at_best_acc  = float('inf')
    best_acc  = 0

    count = 0

    # scheduler = StepLR(optimizer, step_size=10, gamma=0.01)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_data), epochs=epochs)
    
    for epoch in range(epochs):
        
        model.train()
        
        train_sum_loss = 0
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
            train_sum_loss += loss.item() * batch_size
            
            #accuracy
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()

        val_loss, val_acc = evaluate(val_data, model, loss_function, device)
        
        #scheduler.step() #stepLR, CosineAnnealingLR
    
        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_sum_loss/total:.4f} Train Acc: {correct/total * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        print("-------------------------")

        if best_loss > val_loss:
            best_loss = val_loss
            acc_at_best_loss = val_acc

            train_loss = train_sum_loss/total
            train_acc = correct/total * 100
            
            count = 0
            #torch.save(model.state_dict(), "best.pth")
        else:
            count += 1

        if  val_acc > best_acc:
            best_acc = val_acc
            loss_at_best_acc = val_loss

        if(is_early_stopping):
            if count >= patience:
                print("-----------Early Stopping-----------")
                break

    return {
    "best_loss": best_loss,
    "acc_at_best_loss": acc_at_best_loss,
    "best_acc": best_acc,
    "loss_at_best_acc": loss_at_best_acc,
    "train_loss_at_best_loss": train_loss,
    "train_acc_at_best_loss": train_acc
    } 

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


def cross_validate(datasets, model_name, loss_function, device, batch_size=32,
                n_splits=5, epochs=10, patience=5, opt_name="Adam", is_early_stopping=False):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    val_losses = []
    val_accs = []

    for fold, (train_indices, val_indices) in enumerate(skf.split(datasets.data, datasets.targets)):
        
        u.set_seed(42)
        g = torch.Generator()
        g.manual_seed(42)

        model = u.get_model(model_name).to(device)
        optimizer = u.get_optimizer(opt_name, model)

        train_datasets = torch.utils.data.Subset(datasets, train_indices)
        val_datasets = torch.utils.data.Subset(datasets, val_indices)

        train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, generator=g)
        val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

        print(f"-------fold {fold+1}-------")
        
        val_loss, val_acc = train(train_data, val_data, model, loss_function, device, optimizer, epochs, patience, is_early_stopping=is_early_stopping)
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"fold {fold + 1} - Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    print("-------------------------")    
    print(f"Val Mean Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Val Mean Acc: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")


def run_cross_validate(datasets, model_name, loss_function, device, batch_size=32, 
                      n_splits=5, epochs=10, patience=5, opt_name="Adam", 
                      is_early_stopping=False, load_file="splits.pkl"):
    
    splits = u.load_train_val_data(load_file)

    best_losses = []
    accs_at_best_loss = []

    losses_at_best_acc = []
    best_accs = []

    train_losses = []
    train_accs = []

    for fold, (train_idx, val_idx) in enumerate(splits):

        u.set_seed(42)
        g = torch.Generator()
        g.manual_seed(42)

        train_datasets = torch.utils.data.Subset(datasets, train_idx)
        val_datasets = torch.utils.data.Subset(datasets, val_idx)

        train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, generator=g)
        val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

        model = u.get_model(model_name).to(device)
        optimizer = u.get_optimizer(opt_name, model)

        print(f"--------fold {fold + 1}--------")
             
        results = train(train_data, val_data, model, loss_function, device, optimizer,
                            epochs, patience, is_early_stopping=is_early_stopping)
        
        best_losses.append(results["best_loss"])
        accs_at_best_loss.append(results["acc_at_best_loss"])

        losses_at_best_acc.append(results["loss_at_best_acc"])
        best_accs.append(results["best_acc"])

        train_losses.append(results["train_loss_at_best_loss"])
        train_accs.append(results["train_acc_at_best_loss"])

        print(f"fold {fold + 1} - Val Loss By Loss: {results['best_loss']:.4f} "
        f"Val Acc By Loss: {results['acc_at_best_loss']:.2f}%")

        print(f"fold {fold + 1} - Val Loss By Acc: {results['loss_at_best_acc']:.4f} "
        f"Val Acc By Acc: {results['best_acc']:.2f}%")

    print("-------------------------")
    print("[Best Accuracy Based Performance]")
    print(f"Val Mean Loss By Acc: {np.mean(losses_at_best_acc):.4f} ± {np.std(losses_at_best_acc):.4f}")
    print(f"Val Mean Acc By Acc: {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")
    
    print("-------------------------")
    print("[Best Loss Based Performance (More Stable)]")
    print(f"Val Mean Loss By Loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}")
    print(f"Val Mean Acc By Loss: {np.mean(accs_at_best_loss):.2f}% ± {np.std(accs_at_best_loss):.2f}%")

    print("-------------------------")
    print("[Train]")
    print(f"Train loss: {np.mean(train_losses):.4f} ± {np.std(train_losses):.4f}")
    print(f"Train Acc: {np.mean(train_accs):.2f}% ± {np.std(train_accs):.2f}%")
    
    print("-------------------------")
    print("[Generalization Gap]")
    print(f"Gap (Best Loss Based): {np.mean(train_accs) - np.mean(accs_at_best_loss):.2f}%")
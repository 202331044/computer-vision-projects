import torch

def train(model, train_loader, device, criterion, optimizer):

    model.train()
    
    total_size = 0
    total_correct = 0
    total_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        batch_size = imgs.size(0)

        outputs = model(imgs)

        loss = criterion(outputs, labels)
        total_size += batch_size
        total_loss += loss.item() * batch_size

        pred = torch.argmax(outputs, dim=-1)
        total_correct += (pred == labels).sum().item()

        loss.backward()
        optimizer.step()
    
    train_loss = total_loss/total_size
    train_acc = total_correct/total_size*100

    return train_loss, train_acc

def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_size = 0
    total_correct = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.size(0)
            total_size += batch_size

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size

            pred = torch.argmax(outputs, dim=-1)
            total_correct += (pred == labels).sum().item()
        
    val_loss = total_loss/total_size
    val_acc = total_correct/total_size*100

    return val_loss, val_acc

def train_model(model, train_loader, device, criterion, optimizer, epochs, 
                val_loader, save_path):

    history = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = train(model, 
                                      train_loader,
                                      device, 
                                      criterion, 
                                      optimizer)

        val_loss, val_acc = validate(model, 
                                     val_loader, 
                                     device, 
                                     criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        print()

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            checkpoints = {"epoch": epoch,
                           "model_state_dict": model.state_dict(),
                           "optimizer_state_dic": optimizer.state_dict(),
                           "best_val_loss": best_val_loss}
                            
            torch.save(checkpoints, save_path)

    return history


def evaluate(model, test_loader, device, criterion):
    model.eval()
    total_loss = 0
    total_size = 0
    total_correct = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = imgs.size(0)
            total_size += batch_size

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size

            pred = torch.argmax(outputs, dim=-1)
            total_correct += (pred == labels).sum().item()
        
    test_loss = total_loss/total_size
    test_acc = total_correct/total_size*100

    return test_loss, test_acc
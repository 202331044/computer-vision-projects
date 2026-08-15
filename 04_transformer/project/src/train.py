import torch

def train(model, device, train_loader, optimizer, criterion):

    model.train()

    total_loss = 0
    total_correct = 0
    total_size = 0

    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(texts)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, prediction = torch.max(output, dim = 1)
        total_correct += (prediction == labels).sum().item()

        batch_size = labels.size(0)
        total_size += batch_size
        total_loss += loss.item() * batch_size

    train_loss = total_loss / total_size
    train_acc = total_correct / total_size * 100

    return train_loss, train_acc
   
def validation(model, device, val_loader, criterion):
    model.eval()

    with torch.no_grad():

        total_loss = 0
        total_correct = 0
        total_size = 0

        for texts, labels in val_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)
            total_size += batch_size

            output = model(texts)
            loss = criterion(output, labels)

            total_loss += loss.item() * batch_size
            _, prediction = torch.max(output, dim = 1)
            total_correct += (prediction == labels).sum().item()

        val_loss = total_loss / total_size
        val_acc = total_correct / total_size * 100

        return val_loss, val_acc

def train_model(epochs, model, device, train_loader, val_loader, 
                optimizer, criterion, scheduler):

    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(model, 
                                      device,
                                      train_loader, 
                                      optimizer, 
                                      criterion)

        val_loss, val_acc = validation(model, 
                                       device, 
                                       val_loader, 
                                       criterion)
        
        scheduler.step()

        print(f"Epoch [{epoch + 1} / {epochs}]")
        print(f"Train Loss:{train_loss:.4f}")
        print(f"Train Accuracy:{train_acc:.2f}")
        print(f"Val Loss:{val_loss:.4f}")
        print(f"Val Accuracy:{val_acc:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
import torch

def train(epochs, device, train_data, val_data, model, loss_function, optimizer):
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

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

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
import torch

def train(epochs, device, train_data, model, loss_function, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total = 0
        for images, labels in train_data:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        print(f"epoch: {epoch+1} loss: {total_loss/total: .4f}")

def test(device, test_data, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct/total * 100:.2f}")
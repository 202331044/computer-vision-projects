import torch

def train(model, epochs, train_data, device, loss_function, optimizer):
  model.train()                 # train mode: enables dropout, batchnorm

  for epoch in range(epochs):
    total_loss = 0
    total = 0
    for images, labels in train_data:
      images = images.to(device)
      labels = labels.to(device)
      batch_size = labels.size(0)

      outputs = model(images)
      loss = loss_function(outputs, labels)

      optimizer.zero_grad()     # reset gradients
      loss.backward( )          # backpropagation
      optimizer.step()          # update parameters

      total_loss += loss.item() * batch_size # item(): convert tensor scalar to float
      total += batch_size

    print(f"epoch: {epoch + 1} loss: {total_loss/total:.4f}") 

def test(model, test_data, device):
  correct = 0
  total = 0
  model.eval()

  with torch.no_grad():                             # no gradient computation -> no learning
    for images, labels in test_data:
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      _, prediction = torch.max(outputs, 1)         # maxValues, class

      total += labels.size(0)                       # batch_size
      correct += (prediction == labels).sum().item()

  print(f"Accuracy: {100 * correct / total: .2f}%") # round to 2 decimal places

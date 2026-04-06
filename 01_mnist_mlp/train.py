import torch
from sklearn.model_selection import StratifiedKFold
from model import get_model, get_optimizer

def train(train_data, val_data, model, loss_function, device, epochs, patience, optimizer):
  
  best_loss = float('Inf')
  count = 0
  val_total_loss = 0
  val_total_acc = 0

  for epoch in range(epochs):

    model.train()                 # train mode: enables dropout, batchnorm
    
    total_loss = 0
    total = 0
    correct = 0

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
      
      _, prediction = torch.max(outputs, 1)
      correct += (prediction == labels).sum().item()
    
    
    val_loss, val_acc = evaluate(val_data, model, loss_function, device)

    val_total_loss += val_loss
    val_total_acc += val_acc

    print(f"epoch: {epoch + 1}")
    print(f"Train Loss: {total_loss/total:.4f} Train Acc: {correct/total*100 :.2f}%")
    print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
    print("--------------------------")

    #early stopping
    if best_loss > val_loss:
      count = 0
      best_loss = val_loss
    else:
      count += 1
    
    if count >= patience:
      print("Early Stopping")
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

      batch_size = labels.size(0)

      outputs = model(images)
      loss = loss_function(outputs, labels)

      total_loss += loss.item() * batch_size
      total += batch_size

      _, prediction = torch.max(outputs, 1)
      correct += (prediction == labels).sum().item()

  eval_loss = total_loss/total
  eval_acc = correct/total*100

  return eval_loss, eval_acc

def cross_validate( datasets, model_name, loss_function, device, batch_size = 32, n_splits=5,
                   epochs=100, patience=5, opt_name=None):

  skf = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)
  val_total_loss = 0
  val_total_acc = 0

  for fold, (train_idx, val_idx) in enumerate(skf.split(datasets.data, datasets.targets)):
    
    train_datasets = torch.utils.data.Subset(datasets, train_idx)
    val_datasets = torch.utils.data.Subset(datasets, val_idx)

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    
    model = get_model(model_name).to(device)
    optimizer = get_optimizer(opt_name, model)

    print(f"--------fold {fold + 1}--------")
    
    val_loss, val_acc = train(train_data, val_data, model, loss_function, device, epochs, patience, optimizer)
    val_total_loss += val_loss
    val_total_acc += val_acc

  print(f"Val Mean Loss: {val_total_loss/(n_splits):.4f} Val Mean Acc: {val_total_acc/(n_splits):.2f}%")
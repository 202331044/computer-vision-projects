import torch
from sklearn.model_selection import StratifiedKFold
from utils import get_model, get_optimizer, make_train_val_data, load_train_val_data, set_seed
import numpy as np

def train(train_data, val_data, model, loss_function, device, optimizer, epochs=100, patience=5, is_early_stopping=False):
  
  best_val_loss = float('Inf')
  best_val_acc = 0
  count = 0

  for epoch in range(epochs):

    model.train()                 # train mode: enables dropout, batchnorm
    
    train_sum_loss = 0
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

      train_sum_loss += loss.item() * batch_size
      total += batch_size
      
      _, prediction = torch.max(outputs, 1)
      correct += (prediction == labels).sum().item()
    
    val_loss, val_acc = evaluate(val_data, model, loss_function, device)

    print(f"epoch: {epoch + 1}")
    print(f"Train Loss: {train_sum_loss/total:.4f} Train Acc: {correct/total*100 :.2f}%")
    print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    #early stopping
    if(is_early_stopping):
      if best_val_loss > val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        count = 0
      else:
        count += 1
      
      if count >= patience:
        print("Early Stopping")
        print("--------------------------")
        break

  return best_val_loss, best_val_acc


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

  avg_loss = total_loss/total
  accuracy = correct/total*100

  return avg_loss, accuracy


def cross_validate(datasets, model_name, loss_function, device, batch_size=32, n_splits=5,
                   epochs=100, patience=5, opt_name="Adam"):

  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
  val_total_loss = 0
  val_total_acc = 0

  for fold, (train_idx, val_idx) in enumerate(skf.split(datasets.data, datasets.targets)):
    
    set_seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    train_datasets = torch.utils.data.Subset(datasets, train_idx)
    val_datasets = torch.utils.data.Subset(datasets, val_idx)

    train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, generator=g)
    val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    
    model = get_model(model_name).to(device)
    optimizer = get_optimizer(opt_name, model)

    print(f"--------fold {fold + 1}--------")
    
    val_loss, val_acc = train(train_data, val_data, model, loss_function, device, optimizer, epochs, patience)
    val_total_loss += val_loss
    val_total_acc += val_acc

  print(f"Val Mean Loss: {val_total_loss/(n_splits):.4f} Val Mean Acc: {val_total_acc/(n_splits):.2f}%")


def run_cross_validate(datasets, model_name, loss_function, device, batch_size=32, 
                      n_splits=5, epochs=100, patience=5, opt_name="Adam", load_file="splits.pkl"):
    
    splits = load_train_val_data(load_file)
    val_losses = []
    val_accs = []

    for fold, (train_idx, val_idx) in enumerate(splits):

      set_seed(42)
      g = torch.Generator()
      g.manual_seed(42)

      train_datasets = torch.utils.data.Subset(datasets, train_idx)
      val_datasets = torch.utils.data.Subset(datasets, val_idx)

      train_data = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
      val_data = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

      model = get_model(model_name).to(device)
      optimizer = get_optimizer(opt_name, model)

      print(f"--------fold {fold + 1}--------")

      val_loss, val_acc = train(train_data, val_data, model, loss_function, device, optimizer,
                          epochs, patience, is_early_stopping=True)
      
      val_losses.append(val_loss)
      val_accs.append(val_acc)

      print(f"fold {fold + 1} - Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}")
      
    print(f"Val Mean Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Val Mean Acc: {np.mean(val_accs):.2f} ± {np.std(val_accs):.2f}")
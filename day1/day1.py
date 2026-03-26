import torch                                  # pytorch
import torchvision                            # image datasets + preprocessing tools
import torchvision.transforms as transforms   # preprocessing tools
import torch.nn as nn                         # layer definitions
import torch.nn.functional as F               # functional operations

# Tensor: multi-dimensional array for holding data in ML/DL
class SimpleNN(nn.Module):# All PyTorch models must inherit nn.Module
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28*28, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    flat_x = x.view(-1, 28*28)
    x1 = F.relu(self.fc1(flat_x))
    x2 = self.fc2(x1)
    return x2

def train(model, train_data, device, loss_function, optimizer):
  model.train()                 # train mode: enables dropout, batchnorm

  for epoch in range(3):
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

def run():
  
  # cuda = NVIDIA GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # DL models accept only tensors, so convert image data (0~255 -> 0.0~1.0)
  transform = transforms.ToTensor()

  train_datasets = torchvision.datasets.MNIST (
      root = "./data", # location to store dataset
      train = True,
      transform = transform,
      download = True
  )

  test_datasets = torchvision.datasets.MNIST(
      root = "./data",
      train = False,
      transform = transform
  )

 
  train_data = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
  test_data = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = False)

  model = SimpleNN().to(device) # in PyTorch, model and data must be on the same device

  loss_function = nn.CrossEntropyLoss()                       # internally applies log-softmax
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)# Adam: adaptive learning rate, fast convergence, easy tuning

  train(model, train_data, device, loss_function, optimizer)
  test(model, test_data, device)

if __name__ == "__main__":
  run()
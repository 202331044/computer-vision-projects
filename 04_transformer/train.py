import torch
def train(epochs, 
          model,
          criterion, 
          optimizer, 
          label, 
          src, 
          tgt, 
          tgt_vocab_size, 
          causal_mask,
          PAD_ID):
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(src, tgt, causal_mask = causal_mask)
        loss = criterion(out.reshape(-1, tgt_vocab_size), 
                        label.reshape(-1))
        
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            mask = (label != PAD_ID)
            _, predictions = torch.max(out, dim = 2)

            correct = ((predictions == label) & mask).sum().item()
            total = mask.sum().item()

            print(f"---------epoch: {epoch + 1}---------")
            print(f"loss: {loss.item() :.4f} accuracy: {correct / total :.2f}")

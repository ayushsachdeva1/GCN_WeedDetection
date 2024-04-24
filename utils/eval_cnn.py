import torch

def eval_cnn(model, cost_function, val_loader, device):

  # Set the model in evaluation mode.
  model.eval()

  # initialize control variables.
  correct = 0
  cumulative_loss = 0
  n_samples = 0

  # No need to keep track of gradients for this part.
  with torch.no_grad():
    # Run the model on the validation set to keep track of accuracy there.
    for (batch_id, (xb, yb)) in enumerate(val_loader):

      # Move data to GPU if needed.
      xb = xb.to(device)
      yb = yb.to(device)

      # Compute predictions.
      predicted = model(xb)

      # Compute loss.
      loss = cost_function(predicted, yb)
      cumulative_loss += loss.item()

      # Count how many correct in batch.
      predicted_ = predicted.detach().softmax(dim = 1)
      max_vals, max_ids = predicted_.max(dim = 1)
      correct += (max_ids == yb).sum().cpu().item()
      n_samples += xb.size(0)

      n_batches = 1 + batch_id

    return cumulative_loss / n_batches, correct / n_samples
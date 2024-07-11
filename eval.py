import torch
import matplotlib.pyplot as plt


def calc_loss_batch(batch, model, device, pad_token_id=0):
    # Move the batch to the device
    batch = batch.to(device)
    
    # Define the input and target batches
    input_batch = batch[:, :-1]
    target_batch = batch[:, 1:].long()
    
    # Get the model's predictions
    logits = model(input_batch)
    
    # Create a mask for non-padding tokens
    mask = (target_batch != pad_token_id).float()
    
    # Flatten the logits and target_batch for cross_entropy calculation
    logits_flat = logits.reshape(-1, logits.size(-1))
    target_flat = target_batch.reshape(-1)
    
    # Calculate the loss with ignore_index
    loss = torch.nn.functional.cross_entropy(
        logits_flat, target_flat, ignore_index=pad_token_id, reduction='none')
    
    # Apply the mask and calculate the mean loss
    loss = (loss.reshape_as(target_batch) * mask).sum() / mask.sum()
    
    return loss


def calc_loss_loader(data_loader, model, device, sample_size=10):
    total_loss = 0.
    for i, batch in enumerate(data_loader):
        loss = calc_loss_batch(batch, model, device)
        total_loss += loss.item()
        if i >= sample_size:
            break
    return total_loss / len(data_loader)


def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()
import torch
import sentencepiece as spm
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from safetensors.torch import save_model
import json
import argparse
import os


# Import from local files
from model import GPT
from data import create_dataloader
from eval import calc_loss_batch, evaluate_model, plot_losses


def train(model, train_loader, train_subset_loader, val_loader, device, train_config,
                       eval_freq):
    
    total_steps = len(train_loader) * train_config["n_epochs"]
    
    warmup_steps = int(train_config["warmup_ratio"] * total_steps)
    warmup_steps = warmup_steps or 1
    lr_increment = train_config["peak_lr"] / warmup_steps

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0, 
        weight_decay=train_config["weight_decay"]
    )

    # Tracking
    train_losses, val_losses, track_tokens_seen = [], [], []
    grad_norms_before_clip, grad_norms_after_clip = [], []
    tokens_seen = 0
    step = 1

    print(f"Begin training for {train_config['n_epochs']} epochs\n")

    # Main training loop
    for epoch in range(train_config["n_epochs"]):
        
        model.train()

        for batch in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            if step < warmup_steps:
                lr = step * lr_increment
            
            else: # Cosine decay
                progress = ((step - warmup_steps) / 
                            (total_steps - warmup_steps))
                lr = 0.1 * train_config["peak_lr"] + (0.9 * train_config["peak_lr"]) * 0.5 * (
                    1 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(batch, model, device)
            loss.backward()
            
            # Calculate gradient norm before clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            grad_norms_before_clip.append(grad_norm_before.item())

            # Perform gradient clipping
            grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms_after_clip.append(grad_norm_after.item())            
            
            optimizer.step()
            tokens_seen += batch.numel()

            # Optional evaluation step
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_subset_loader, val_loader, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                clip_rate = sum([before > after for before, after in zip(grad_norms_before_clip, grad_norms_after_clip)]) / len(grad_norms_before_clip)
                with open(f"{save_directory}train.log", "a") as f:
                    f.write(f"{epoch} | {step:06d} | Train {train_loss:.5f} | Val {val_loss:.5f} | Clip Rate {clip_rate:.3f}\n")
            
            step += 1

        save_model(model, f"{save_directory}model.safetensors")
        print(f"\nEpoch {epoch} complete\ncheckpoint saved\n")

    return train_losses, val_losses, track_tokens_seen, (grad_norms_before_clip, grad_norms_after_clip)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a GPT model on TinyNews data')

    parser.add_argument('-c', '--config', default="", help=f'Directory of the config file')
    parser.add_argument('-d', '--data', default="", help='Directory of the training data')
    parser.add_argument('-o', '--out', default="", help='Directory to save the trained model')

    args = parser.parse_args()

    save_directory = args.out.rstrip("/") + "/" if args.out else ""
    data_directory = args.data.rstrip("/") + "/" if args.data else ""
    config_directory = args.config.rstrip("/") + "/" if args.config else ""

    if not os.path.isfile(f"{config_directory}config.json"):
        print("Error: config.json not found\n\nExiting\n")
        exit()

    with open(f"{config_directory}config.json", "r") as f:
        config = json.load(f)
    
    print(f"Using config:\n\n{config}\n")

    MODEL_CONFIG = config["model"]
    TRAIN_CONFIG = config["train"]

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    torch.manual_seed(sum([(ord(char)) for char in "TinyNews"]))

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    model = GPT(MODEL_CONFIG)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    try:
        save_model(model, f"{save_directory}model.safetensors")
        print("Initialized model saved successfully\n")
    except:
        print("There was an error saving the model\nExiting\n")
        exit()

    ##############################
    # Data
    ##############################

    if not all((os.path.isfile(f"{data_directory}train.txt"), os.path.isfile(f"{data_directory}validation.txt"))):
        print("Error: training.txt and/or validation.txt not found\n\nExiting\n")
        exit()

    print("Loading data... (this may take a few minutes)\n")

    if not os.path.isfile(f"{data_directory}train_subset.txt"):

        with open(f"{data_directory}train.txt", 'r') as f:
            lines = f.readlines()
                
        train_subset = []
        
        for i in torch.randint(0, 1_070_564, (1, 10_815)).tolist()[0]:
            train_subset.append(lines[i])
        
        with open(f"{data_directory}train_subset.txt", 'w') as f:
            for line in train_subset:
                f.write(line)
    
    train_loader = create_dataloader(
        f"{data_directory}train.txt",
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=0
    )

    train_subset_loader = create_dataloader(
        f"{data_directory}train_subset.txt",
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=0
    )

    val_loader = create_dataloader(
        f"{data_directory}validation.txt",
        batch_size=TRAIN_CONFIG["batch_size"],
        num_workers=0
    )

    print("Data loaded.\n")

    ##############################
    # Train
    ##############################

    tokenizer = spm.SentencePieceProcessor(
        model_file=f'tokenizers/tinynewstokenizer{MODEL_CONFIG["vocab_size"]}.model')

    train_losses, val_losses, tokens_seen, grad_norms = train(
        model, train_loader, train_subset_loader, val_loader, device,
        TRAIN_CONFIG, eval_freq=100
    )

    # Plot results
    epochs_tensor = torch.linspace(0, TRAIN_CONFIG["n_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(f"{save_directory}loss.png")

    clip_rate = sum([before > after for before, after in zip(*grad_norms)]) / len(grad_norms[0])
    print(f"Gradient clip rate: {clip_rate:.3f}")

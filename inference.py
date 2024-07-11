import torch


def text_to_token_ids(text, tokenizer):
    encoded = [1]
    encoded.extend(tokenizer.encode(text))
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate(model, idx, context_length, max_new_tokens=None, temperature=0.0, top_k=None, eos_id=None):

    max_new_tokens = max_new_tokens or context_length - idx.shape[1]

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_length:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]

        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:
            break

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

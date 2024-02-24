import logging
import torch
import torch.nn.functional as F

# This should work for any HuggingFace transformers model and tokenizer


def add_token(model, tokenizer, token, description):
    # First attempt to add the token to the tokenizer
    tokens_before = len(tokenizer)
    tokenizer.add_tokens([token])
    if tokens_before == len(tokenizer):
        logging.info("Token already in tokenizer")
        return

    # Next expand the dimension of the model embeddings (NOTE: This already updates the final linear layer size too)
    new_token_embeddings = model.resize_token_embeddings(len(tokenizer))

    # Tokenize the description, get embeddings for each token, and sum
    description_tokens = torch.tensor(tokenizer(description)["input_ids"])
    embeddings_sum = new_token_embeddings(description_tokens).sum(dim=0)

    # Do the same for the LM head
    new_lm_head = model.get_output_embeddings()
    weights_t = new_lm_head.weight.t()
    lm_head_embeddings_sum = F.embedding(
        description_tokens, weights_t).sum(dim=0)

    # Set the new token's embedding to the sum of the description's token embeddings
    new_token_embeddings_module = model.get_input_embeddings()
    with torch.no_grad():
        new_token_embeddings_module.weight[-1, :] = embeddings_sum
        new_lm_head.weight[:, -1] = lm_head_embeddings_sum
    model.set_input_embeddings(new_token_embeddings)
    model.set_output_embeddings(new_lm_head)

    return model, tokenizer


def add_tokens(model, tokenizer, tokens, descriptions):
    for token, description in zip(tokens, descriptions):
        add_token(model, tokenizer, token, description)
    return model, tokenizer

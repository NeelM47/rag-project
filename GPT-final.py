# The model’s training loss improves initially, reaching below 0.001 by epoch 101, but fluctuates, rising after epoch 110 and decreasing again after epoch 150. Future updates will identify which specific words contribute to the loss. The tokenize method uses the padding token `<pad>` for unknown words. Punctuation is separated from words, and carriage returns are handled as a distinct case, replacing them with the `<cr>` token (though this feature is untested). During detokenization, spaces preceding punctuation are not automatically removed, as spaces are added to all tokens. The sequence length (`training_input_seq_len`) can be adjusted over time, but it is better managed dynamically during batch creation using a custom `collate_fn` in the `DataLoader`.

from icecream import ic
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

# print("loading libraries") 
import os        # to get filename of this script
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import the learning rate scheduler
import math
import inspect
#import string # replaced with self.punctuation_list = ['.', ',', '/', '\\', '[', ']', '<', '?', '>', '-']]  # Specific list of punctuations

# print("Hardcoding Memorized_Speech = Gettysburg Address") #(for simplicity in this toy example)
Memorized_Speech = """
Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate - we can not consecrate - we can not hallow-this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us - that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion - that we here highly resolve that these dead shall not have died in vain - that this nation, under God, shall have a new birth of freedom - and that government of the people, by the people, for the people, shall not perish from the earth.
"""
# print(f'Length of Memorized_Speech = {len(Memorized_Speech)} characters, as follows:')
# print(Memorized_Speech)

# Add special tokens here. "<pad>" is also used for unknown words. The carriage-return specialtoken will be auto-inserted into the received text before tokenization.  But  tabs and newlines are not implemented/supported.
# Hyperparameters
hyperparameters = {
    "vocab_size": 152,  # Estimated vocabulary size for Gettysburg Address + special tokens
    "special_tokens": ["<FreetheLLM>", "<cr>",  "<pad>"], 
    "n_embd": 512,      # Embedding dimension
    "n_layer": 4,       # Number of layers
    "n_head": 16,        # Number of attention heads
    "n_inner": 4 * 512,   # Inner dimension of feedforward network (4 times n_embd)
    "max_sequence_len": 264, #  Maximum sequence length
    "epochs": 200,      # Number of training epochs
    "learning_rate": 1e-3, # [Initial] Learning rate
    "batch_size": 1,      # Batch size (since the dataset is small)
    "dropout": 0.2     # Dropout probability
}
# More Script/Training parameters:
min_training_input_seq_len = 32
Early_stopping_loss = 0.003

Per_token_loss_threshold = 0.5  
# Adjust this Per_token_loss_threshold as needed  # Per-Token Loss: torch.nn.CrossEntropyLoss with reduction='none' is used to get the loss for each individual token.  Reshaping Loss: The per-token loss is reshaped to have the same dimensions as target_seq for easier indexing.  Threshold for Error Tokens: A threshold is defined to filter tokens with significant errors. You can adjust this threshold value (e.g., 0.5) based on your observations.  Identifying Error Tokens: The code iterates through the per-token loss, and tokens with loss values above the threshold are identified.

def print_with_line(message):
  #ic(message)
  frame = inspect.currentframe().f_back    #  needs import inspect
  line_number = frame.f_lineno
  print(f"{message} at script line {line_number}")

# --- Tokenizer and Detokenizer ---
class Tokenizer:
    def __init__(self, text, special_tokens, vocab_size_hyperparameter):
        self.special_tokens = special_tokens
        self.cr_token = special_tokens[1]
        self.punctuation_list = ['.', ',', '/', '\\', '[', ']', '<', '?', '>', '-']  # Specific list of punctuations
        estimated_vocab_size = vocab_size_hyperparameter #hyperparameters["vocab_size"]

        # Preprocess text to separate existing punctuation from words, and then auto-inserts <cr> special tokens at carriage returns.
        text = self.separate_punctuation(text)

        in_text_words = []
        in_text_punctuations = []
        for candidate in text.split():  # Split into tokens (space-separated words and punctuation; includes words attached to punctuation)
           # ic(candidate)
           cleaned_words = ''.join(c for c in candidate if c not in self.punctuation_list)  #strip punctuation from words
           # ic(cleaned_words)
           if cleaned_words:
              in_text_words.append(cleaned_words.lower())
              # ic(in_text_words)
           for char in candidate:  # Iterate through each character in the candidates
              if char in self.punctuation_list:
                 in_text_punctuations.append(char)      # Add in-text punctuation as separate tokens

        # Ensure unique and sorted word and punctuation tokens
        in_text_words = list(set(in_text_words))
        in_text_words.sort()
        in_text_punctuations = list(set(in_text_punctuations))
        in_text_punctuations.sort()
        
        self.vocab = self.special_tokens + in_text_punctuations + in_text_words   # Vocab starts with special tokens, then punctuation, then whole words.
        self.vocab_size = len(self.vocab)  # Calculate vocabulary size dynamically
        # Alert if vocab_size is different from a predefined hyperparameter estimate (optional)
        if self.vocab_size != estimated_vocab_size:
            ic(f"Calculated vocab_size({self.vocab_size}) differs from estimated size ({estimated_vocab_size}).")
            # print(f"Warning: Calculated vocab_size ({self.vocab_size}) differs from estimated size ({estimated_vocab_size}).")

        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for i, word in enumerate(self.vocab)}

    def separate_punctuation(self, text):   # text passed to the tokenize method is also preprocessed to have separated punctuation before tokenization  #separate_punctuation(self, text) method, as currently implemented, does not directly affect carriage returns (\r) in the original text.
        # Adds spaces around punctuation to separate them from words.
        for char in self.punctuation_list:
            text = text.replace(char, f' {char} ')
        #Replace carriage returns (backslash-r) in the input text with a special token (e.g., <cr>).
        text = text.replace('\r', f' {self.cr_token} ')  # Replace \r with <cr> token and pad with spaces.
        return text

    def tokenize(self, text):
        # Apply punctuation separation before tokenizing
        text = self.separate_punctuation(text)
        words = text.lower().split() #preserves special tokens like the auto-inserted <cr> 
        token_ids = []
        for word in words:
          if word in self.word_to_index:
            token_ids.append(self.word_to_index[word])
          else:
            # token_ids.append(self.word_to_index['<pad>'])
            token_ids.append(self.word_to_index[self.special_tokens[-1]])  # Use last special token as default (e.g., <pad>)  # The tokenize method now uses the last special token in the self.special_tokens list (which is assumed to be the padding token <pad> in this case) as the default token for unknown words.
        return token_ids

    def detokenize(self, tokens):
        return " ".join([self.index_to_word[token] for token in tokens if token in self.index_to_word])

# --- GPT-2 Model ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config["n_embd"], 3 * config["n_embd"])
        # output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"])
        # regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.register_buffer("bias", torch.tril(torch.ones(config["max_sequence_len"], config["max_sequence_len"]))
                                    .view(1, 1, config["max_sequence_len"], config["max_sequence_len"]))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        ic(x.shape)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(config["n_embd"], config["n_inner"]),
            nn.GELU(),
            nn.Linear(config["n_inner"], config["n_embd"]),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ToyGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.position_embedding_table = nn.Embedding(config["max_sequence_len"], config["n_embd"])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

        # Initialize weights to be small for better training
        self.apply(self._init_weights)

        # Tie the weights of the embedding and the output layer
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
      #if isinstance(module, nn.Linear):
      #  torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        # print("isinstance(module, nn.Linear) and module.bias is not None")
        torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #print("torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation during generation
            for _ in range(max_new_tokens):
                # Limit input_ids to the last max_sequence_len tokens
                input_ids_truncated = input_ids[:, -self.config["max_sequence_len"]:]

                # Get logits from the model
                logits, _ = self(input_ids_truncated)  # No need for loss during generation

                # Focus on the logits for the last time step (next token prediction)
                logits = logits[:, -1, :] / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample the next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append next token to input sequence
                input_ids = torch.cat((input_ids, next_token), dim=1)

        self.train()  # Return model to training mode
        return input_ids

# --- Dataset ---
class Dataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = self.tokenizer.tokenize(data)
        # print(f"DEBUG: Total tokens: {len(self.tokens)} in Dataset(")  # Add this line

        # Calculate token counts
        self.token_counts = self._calculate_token_counts()  # Store counts in the object

        # Create input-target pairs
        self.data = []
        for i in range(0, len(self.tokens) - seq_len - 1, seq_len):
          input_seq = self.tokens[i:i + seq_len]
          target_seq = self.tokens[i + 1:i + seq_len + 1]
          # ic(torch.tensor(input_seq))
          # ic(torch.tensor(target_seq))
          self.data.append((torch.tensor(input_seq), torch.tensor(target_seq)))

        # print(f"DEBUG: Number of data samples created in class Dataset(Dataset): {len(self.data)}")  # Add this line

        # Print token-vocabulary information
        # print_with_line("# Print token-vocabulary information:")
        self.print_vocabulary_info()  # Call the new method

    def _calculate_token_counts(self):
        #Calculates the frequency of each token in self.tokens.
        counts = {}
        for token in self.tokens:
            if token in counts:
                counts[token] += 1
                # print(f"token {token} count has been incremented to {counts[token]}")
            else:
                counts[token] = 1
        return counts

    def print_vocabulary_info(self):
        # print_with_line("# Print token-vocabulary information:")
        for token_id in range(self.tokenizer.vocab_size):  # Iterate through indices
            token = self.tokenizer.index_to_word[token_id]  # Get token string from index
            count = self.token_counts.get(token_id, 0)  # Correct: token_id is an integer ID  # Get count, default to 0 if not found
            #print(f"  Token {token_id}: '{token}' occurs {count} times in the dataset")
            #print(f"  Token {token_id}:'{token}' \t\t occurs {count} times in the dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Return the pre-processed tensor pairs

# --- Trainer ---
class Trainer:
    def __init__(self, model, tokenizer, train_loader, hyperparameters, device):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader  # notice this change
        self.hyperparameters = hyperparameters
        self.Per_token_loss_threshold = Per_token_loss_threshold  # Assign global to instance
        self.Early_stopping_loss = Early_stopping_loss  # Set Early stopping loss
        self.device = device  # Store the device

        self.optimizer = optim.AdamW(self.model.parameters(), lr=hyperparameters["learning_rate"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.99, patience=10) 
        # mode='min': Indicates that you want to minimize the loss.
        # factor=0.1: The factor by which the learning rate is reduced (e.g., 0.1 means reduce to 10%).
        # patience=10: Number of epochs with no improvement after which the learning rate will be reduced.
        # verbose=True: Prints a message when the learning rate is adjusted.
        # Step the Scheduler: Call self.scheduler.step(average_loss) after calculating average_loss. This tells the scheduler to update the learning rate based on the current loss.
        # Automated Adjustment: The scheduler automatically adjusts the learning rate, removing the need for manual tuning during training.
        # Improved Convergence: Can help the model converge more smoothly and potentially reach a better solution.
        # Reduced Fluctuations: Helps reduce the fluctuations in the loss.

    def train(self):
        self.model.train()  # Set model to training mode
        for epoch in range(self.hyperparameters["epochs"]):
            total_loss = 0
            for batch_idx, (input_seq, target_seq) in enumerate(self.train_loader):  # Use enumerate to get batch index # Directly use the loaded batches
                input_seq = input_seq.to(self.device)  # Move to device
                target_seq = target_seq.to(self.device) # Move to device

                self.optimizer.zero_grad()
                logits, loss = self.model(input_seq, targets=target_seq) # logits are the raw predictions

                # <DISABLED the non-working feature>
                """
                # Per-token loss calculation (using cross-entropy as an example)
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # 'none' to get per-token loss
                per_token_loss = loss_fn(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                per_token_loss = per_token_loss.view(target_seq.size())  # Reshape to match target_seq shape

                # Move error reporting INSIDE the batch loop
                if loss.item() < 0.01:  # Check loss for current batch
                    print("Tokens with significant errors (per-token loss > threshold): [feature not working]")
                    for i in range(target_seq.size(0)):  # Iterate over elements in the batch
                        for token_idx in range(target_seq.size(1)): 
                            if per_token_loss[i, token_idx] > self.Per_token_loss_threshold:
                                target_token_id = target_seq[i, token_idx].item()
                                target_word = self.tokenizer.index_to_word[target_token_id]
                                print(f"  Batch item {i}, Token {token_idx}: Word '{target_word}' (ID: {target_token_id}), Loss: {per_token_loss[i, token_idx].item():.4f}")
                """
                # <DISABLED the non-working feature>

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)  # Consider number of batches
            print(f"Epoch {epoch+1}/{self.hyperparameters['epochs']}, Loss: {average_loss:.4f}")

            #if loss < 0.01:  # Check loss for current batch
            #    ic("LOSS IS BELOW 0.01")
            #    print("           LOSS IS BELOW 0.01")
            #if loss < 0.001:  # Check loss for current batch
            #    ic("LOSS IS BELOW 0.001")
            #    print("                               LOSS IS BELOW 0.001")

            last_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(average_loss)  # Update the lossrate-scheduler with the current loss

            # Check if the learning rate has changed and print it
            current_lr = self.optimizer.param_groups[0]['lr']
            # last_lr = self.scheduler.get_last_lr()[0]  # Get the last learning rate
            if current_lr != last_lr:
                # print(f"Learning rate reduced to {last_lr:.6f}")
                ic(f"Learning rate reduced to: {current_lr:.6f}")
                ic(f"Previous Learning rate was: {last_lr:.6f}")

            if epoch!=0 and epoch%100==0:
                current_lr = self.optimizer.param_groups[0]['lr']  # Get the current learning rate from the optimizer
                print(f"Epoch {epoch + 1}: Current learning rate: {current_lr:.6f}")  #current_lr Retrieval: Inside the if (epoch % 100 == 0) block, the current learning rate is obtained using self.optimizer.param_groups[0]['lr']. This is the standard way to access the learning rate of the first (and often only) parameter group in PyTorch optimizers.

                ic('saving model checkpoint')
                self.save_checkpoint(f"model_checkpoint_epoch_{epoch + 1}.pth", epoch, average_loss)  # Pass epoch and average_loss

            # Early stopping condition
            if average_loss < self.Early_stopping_loss:
                print(f"Early stopping: Average loss {average_loss:.4f} is below the threshold ({self.Early_stopping_loss}).")
                self.save_checkpoint(f"model_checkpoint_early_stop.pth", epoch, average_loss)  # Save checkpoint
                break  # Exit the training loop

    def save_checkpoint(self, path, epoch, average_loss):
        # Get the current script's filename
        script_filename = os.path.basename(__file__)  # Get filename from the current script path
    
        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
        # Construct the new filename
        base_filename, extension = os.path.splitext(path)  # Split original filename
        new_filename = f"{base_filename}_{script_filename}_{current_datetime}{extension}"
        # ic(new_filename)
    
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': average_loss,
            'hyperparameters': self.hyperparameters
        }, new_filename)


# --- Main Execution ---
def main():
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # print_with_line("# Initialize tokenizer")
    ic("here")
    tokenizer = Tokenizer(Memorized_Speech, hyperparameters["special_tokens"], hyperparameters["vocab_size"])  # The special_tokens list is now defined in the hyperparameters dictionary.
    # print(f"Vocabulary Size: {tokenizer.vocab_size}")
    dataset = Dataset(Memorized_Speech, tokenizer, min_training_input_seq_len)  # Common values of min_training_input_seq_len for smaller models or experiments are 32, 64, 128, or 256. 
    train_loader = DataLoader(dataset, batch_size=hyperparameters["batch_size"])

    # print(f"HyperParamters = {hyperparameters}")
    model = ToyGPT2(hyperparameters).to(device)

    # print_with_line("# Initialize trainer")
    trainer = Trainer(model, tokenizer, train_loader, hyperparameters, device)

    # print_with_line("# Train the model")
    trainer.train()

    print("") # space
    model.eval()

    # Example 1: Recite the Gettysburg Address
    # print_with_line("# Example 1: Recite the Gettysburg Address")
    start_text = "four score"
    start_tokens = torch.tensor(tokenizer.tokenize(start_text)).unsqueeze(0).to(device)
    print("Prompt:", start_text)
    generated_tokens = model.generate(start_tokens, max_new_tokens=len(dataset.tokens)-len(start_tokens), temperature=1.0) # Generate a completion for the whole dataset
    generated_text = tokenizer.detokenize(generated_tokens.squeeze().tolist())
    print("\nResponse:\n", generated_text)
    
    print("")  # space
    # Example 2: Free text generation after encountering <FreetheLLM>  #### Eventually, modify to request user text inxlusinf  only Gettysburg vocabulary]
    # print_with_line("# Example 2: Free text generation after encountering <FreetheLLM>")
    
    start_text = "we here highly resolve that these dead shall not have died in vain and that this nation under god shall have a new "
    special_token = tokenizer.special_tokens[0] # Get the <FreetheLLM> token
    start_text += special_token # Append the special token directly to the string
    print("Prompt:", start_text)
    
    start_tokens = torch.tensor(tokenizer.tokenize(start_text)).unsqueeze(0).to(device) # Tokenize the combined string

    generated_tokens = model.generate(start_tokens, max_new_tokens=100, temperature=1.0)
    generated_text = tokenizer.detokenize(generated_tokens.squeeze().tolist())
    print("\nFreestyle Generation:\n", generated_text)


if __name__ == "__main__":
    main()

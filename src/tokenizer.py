import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def batch_iterator(file_path, batch_size=1000):
    """Yields batches of lines from the dataset to train the tokenizer efficiently."""
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def train_tokenizer(data_path, vocab_size=5000, save_path="../models/tokenizer.json"):
    """Trains a BPE tokenizer on the given dataset and saves it."""
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File '{data_path}' not found.")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
    )

    # Train the tokenizer using a memory-efficient iterator
    tokenizer.train_from_iterator(batch_iterator(data_path), trainer)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    return tokenizer

if __name__ == "__main__":
    data_file = "data/cleaned_constitution.txt"
    model_save_path = "models/tokenizer.json"
    tokenizer = train_tokenizer(data_file, vocab_size=5000, save_path=model_save_path)
import os
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

from logger.logger import get_logger

log = get_logger(__name__)

class BPE_Tokenizer:
    def __init__(self,
                data = None,
                step  = None,
                save_path = "./bpevocab.json",
                max_vocab_size = 7000,
                pad_token = "<pad>",
                bos_token = "<bos>",
                eos_token = "<eos>",
                unk_token = "<unk>"):
        
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
    
        self.special_tokens = [pad_token, bos_token, eos_token, unk_token]
        
        if os.path.isfile(save_path):
            log.info(f"Loading trained bpe tokenizer from {save_path}")
            self.tokenizer = self.load_vocab(save_path)
        
        else:
            log.info(f"Creating bpe tokenizer with {max_vocab_size} max vocab")

            self.tokenizer = BPE_Tokenizer.create_BPEtokenizer(max_vocab_size, data, step, self.special_tokens, self.unk_token)
            
            self.save_vocab(save_path)
        
        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
    
    def __call__(self, text, max_length = None, padding=True, add_special_tokens = True):

        if type(text)==list:
            return self.batch_encode(text, max_length, padding, add_special_tokens)
        
        return self.encode(text, max_length, padding, add_special_tokens)
    
    def encode(self, text, max_length = None, padding=True, add_special_tokens = True):
        if add_special_tokens:
            text = self.bos_token + text + self.eos_token
            encoding = self.tokenizer.encode(text).ids

            if max_length and padding:
                encoding += [self.pad_id]*(max_length - len(encoding))

            return encoding
        
        return self.tokenizer.encode(text).ids
        
    
    def batch_encode(self, text, max_length = None, padding=True, add_special_tokens = True):
        if add_special_tokens:
            text = [self.bos_token + t + self.eos_token for t in text]
            encoding = [t.ids for t in self.tokenizer.encode_batch(text)]

            if max_length and padding:
                for i in range(len(encoding)):
                    encoding[i] += [self.pad_id]*(max_length - len(encoding[i]))

            return encoding
        
        text = [self.bos_token + t + self.eos_token for t in text]
        encoding = [t.ids for t in self.tokenizer.encode_batch(text)]

        return encoding
    
    def decode(self, ids):
        return self.tokenizer.decode(ids).strip()
    
    def batch_decode(self, ids):
        return [out.strip() for out in self.tokenizer.decode_batch(ids)]

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def load_vocab(self, path):
        return Tokenizer.from_file(path)
    
    def save_vocab(self, path = None):
        self.tokenizer.save(path)
    
    def create_BPEtokenizer(vocab_size, data, step, special_tokens, unk_token):
        tokenizer = Tokenizer(models.BPE(unk_token=unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token=unk_token)
        tokenizer.train_from_iterator(BPE_Tokenizer.get_training_corpus(data, step), trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer
    
    def get_training_corpus(data, batch):
        data = [i for i in data]
        for i in range(0, len(data), batch):
            yield data[i : i + batch]
    
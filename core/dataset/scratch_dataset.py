import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from logger.logger import get_logger

log = get_logger(__name__)

class Scratch_LexDataset(Dataset):
    def __init__(   self,
                    data_path,
                    tokenizer,
                    batch = 256,
                    src_max_token_len = 256,
                    trg_max_token_len = 256):
        super().__init__()

        self.tokenizer = tokenizer
        self.src_max_token_len = src_max_token_len
        self.trg_max_token_len = trg_max_token_len

        self.data = list()

        dataframe = pd.read_csv(data_path)

        try:
            dataframe = dataframe[["original", "normalized"]]
        except:
            dataframe = dataframe[["ceg", "normalized"]]
        
        dataframe.columns = ["src", "trg"]

        self.prepare_io(dataframe, batch)

    def __len__(self):
        return len(self.data)
    
        
    def prepare_io(self, dataframe, batch):
        self.src = list(dataframe['src'])
        self.trg = list(dataframe['trg'])
        src_ids, trg_ids= self.encoding(dataframe, batch)
            
        for index in range(len(dataframe)):
            src_id = torch.tensor(src_ids[index], dtype=torch.int32)
            trg_id = torch.tensor(trg_ids[index], dtype=torch.int32)
            src_attention_mask = self._create_padding_mask(src_id, self.tokenizer.pad_id)
            label_attention_mask = self._create_padding_mask(trg_id, self.tokenizer.pad_id)


            self.data.append({'input_ids': src_id.flatten(), 'labels': trg_id.flatten(),
                            "src_attention_mask":src_attention_mask.flatten(), "label_attention_mask": label_attention_mask.flatten(), })
            
            if index + 1 == 1 or (index + 1) % 1000 == 0 or index+1 == len(dataframe):
                log.info(f"Indexing... {index+1}/{len(dataframe)}")

    
    def encoding(self, dataframe, batch):
        src_ids = []
        trg_ids = []
        
        for i in range(0, len(dataframe), batch):

            srcs = [question.strip() for question in list(dataframe['src'][i:i+batch])]
            
            trgs = [ans.strip() for ans in list(dataframe['trg'][i:i+batch])]

            src_encoding = self.tokenizer(srcs,
                                        max_length = self.src_max_token_len,)
            trg_encoding = self.tokenizer(trgs,
                                        max_length = self.trg_max_token_len,)

            src_ids += src_encoding
            trg_ids += trg_encoding

            if i + 1 == 1 or (i - batch) % 1000 == 0 or i+batch == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")

        return src_ids, trg_ids
    
    def _create_padding_mask(self, ids, pad_token_id):
        return ids == pad_token_id

    def __getitem__(self, index: int):
        return self.data[index]

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from logger.logger import get_logger

log = get_logger(__name__)

class Pretrained_LexDataset(Dataset):
    def __init__(   self,
                    data_path,
                    tokenizer,
                    modeltype = "t5",
                    batch = 256,
                    src_max_token_len = 256,
                    trg_max_token_len = 256):
        super().__init__()

        self.tokenizer = tokenizer
        self.modeltype = modeltype
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
        src_ids, trg_ids, src_masks, trg_masks = self.encoding(dataframe, batch)

        
        for index in range(len(dataframe)):
            src_id = torch.tensor(src_ids[index], dtype=torch.int32)
            trg_id = torch.tensor(trg_ids[index], dtype=torch.int32)
            src_attention_mask = torch.tensor(src_masks[index], dtype=torch.int32)
            label_attention_mask = torch.tensor(trg_masks[index], dtype=torch.int32)


            self.data.append({'input_ids': src_id.flatten(), 'labels': trg_id.flatten(),
                            "src_attention_mask":src_attention_mask.flatten(), "label_attention_mask": label_attention_mask.flatten(), })
            
            if index + 1 == 1 or (index + 1) % 1000 == 0 or index+1 == len(dataframe):
                log.info(f"Indexing... {index+1}/{len(dataframe)}")


    
    def encoding(self, dataframe, batch):
        src_ids = []
        trg_ids = []
        src_masks = []
        trg_masks = []
        
        for i in range(0, len(dataframe), batch):

            srcs = [question.strip() for question in list(dataframe['src'][i:i+batch])]
            
            if self.modeltype == "t5":
                trgs = [self.tokenizer.pad_token + ans.strip() for ans in list(dataframe['trg'][i:i+batch])]
            else:
                trgs = [ans.strip() for ans in list(dataframe['trg'][i:i+batch])]

            src_encoding = self.tokenizer(srcs,
                                            padding='max_length',
                                            max_length = self.src_max_token_len,
                                            truncation = True)
            trg_encoding = self.tokenizer(trgs,
                                            padding='max_length',
                                            max_length = self.trg_max_token_len,
                                            truncation = True)

            
            src_ids += src_encoding["input_ids"]
            src_masks += src_encoding["attention_mask"]

            trg_ids += trg_encoding["input_ids"]
            trg_masks += trg_encoding["attention_mask"]

            if i + 1 == 1 or (i - batch) % 1000 == 0 or i+batch == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")

        return src_ids, trg_ids, src_masks, trg_masks

    def __getitem__(self, index: int):
        return self.data[index]

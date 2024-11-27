import os
import sys
from typing_extensions import override
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader

from .base_executor import Base_Executor

from core.dataset import Scratch_LexDataset
from core.model import Seq2SeqTransformer
from core.tokenizer import *

from timeit import default_timer as timer

from logger.logger import get_logger

log = get_logger(__name__)


class Transformer_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")
    
    
    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                
                pred = self.model.generate( src = batch['input_ids'].to(self.config.DEVICE),
                                            src_key_padding_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                            start_symbol = self.tokenizer.bos_id,
                                            end_symbol = self.tokenizer.eos_id,
                                            pad_symbol = self.tokenizer.pad_id,
                                            max_len = max_length)
                
                decoded_preds += self.tokenizer.batch_decode(self._infer_post_processing(pred.tolist()), skip_special_tokens=True)

                log.info(f"|===| Inferring... {it+1} it |===|")

        return decoded_preds
    
    @override
    def _infer_post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.tokenizer.eos_id)])
            except:
                res.append(out)

        return res
    
    @override
    def _init_training_properties(self):
        if self.config.DO_PRETRAINING:
            self.pretrain_optim = torch.optim.Adam(self.model.parameters(), lr=self.config.pretrain_LR, betas=self.config.pretrain_BETAS, eps=1e-9)

            self.pretrain_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
            
            self.pretrain_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.pretrain_optim, total_iters = self.config.pretrain_warmup_step)


        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, betas=self.config.BETAS, eps=1e-9)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = self.config.warmup_step)

        self.SAVE = self.config.SAVE

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
            try:
                log.info(f"\t- Last train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Last train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
            self.optim.load_state_dict(ckp['optimizer'])
            self.scheduler.load_state_dict(ckp['scheduler'])
            self.best_score = ckp['best_score']
 
    def _create_data_utils(self):

        self._create_tokenizer()

        log.info("# Creating Datasets")

        if self.config.DO_PRETRAINING:
            self.pretrain_data = Scratch_LexDataset(data_path = self.config.pretrain_data_path,
                                            tokenizer = self.tokenizer,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
        
        self.train_data = Scratch_LexDataset(data_path = self.config.train_path,
                                            tokenizer = self.tokenizer,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
        self.val_data = Scratch_LexDataset(data_path = self.config.val_path,
                                            tokenizer = self.tokenizer,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
    

    def _init_eval_predict_mode(self):
        self._create_tokenizer(load_trained_bpe = True)

        if self.mode == "eval":
            log.info("###Load eval data ...")
            self.val_data = Scratch_LexDataset(data_path = self.config.val_path,
                                            tokenizer = self.tokenizer,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
            
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
            
            self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)

        elif self.mode == "predict":
            log.info("###Load predict data ...")
            self.predict_data = Scratch_LexDataset(data_path = self.config.predict_path,
                                            tokenizer = self.tokenizer,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
            

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    
    def _pretrain_step(self):
        assert self.config.NUM_PRETRAIN_STEP is not None
        assert self.config.NUM_PRETRAIN_STEP > 0

        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.train()

        losses = 0
        current_step = 0

        log.info(f"#----------- START PRE-TRAINING -----------------#")
        log.info(f"(!) Show pre-train loss after each {self.config.show_loss_after_pretrain_steps} steps")
        log.info(f"(!) Save model after each {self.config.save_after_pretrain_steps} steps")
        s_train_time = timer()

        while True:
            for batch in self.pretrainiter:
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                logits = self.model(src = batch['input_ids'].to(self.config.DEVICE),
                                    trg = trg_input,
                                    src_padding_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    trg_padding_mask = label_attention_mask)


                self.pretrain_optim.zero_grad()

                trg_out = labels[:, 1:]

                loss = self.pretrain_loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()

                self.pretrain_optim.step()

                self.pretrain_scheduler.step()
                
                losses += loss.data.item()

                current_step += 1

                if current_step % self.config.show_loss_after_pretrain_steps == 0:
                    log.info(f"[Step {current_step} | {int(current_step/self.config.NUM_PRETRAIN_STEP*100)}% completed] Train Loss: {losses / current_step}")

                if current_step % self.config.save_after_pretrain_steps == 0 or current_step >= self.config.NUM_PRETRAIN_STEP:
                    if self.SAVE:
                        lstatedict = {
                                    "state_dict": self.model.state_dict(),
                                    "optimizer": self.pretrain_optim.state_dict(),
                                    "scheduler": self.pretrain_scheduler.state_dict(),
                                    "step": current_step,
                                    "best_score": self.best_score
                                }

                        lfilename = f"last_ckp.pth"
                        torch.save(lstatedict, os.path.join(folder,lfilename))

                if current_step >= self.config.NUM_PRETRAIN_STEP:   
                    e_train_time = timer()
                    log.info(f"#----------- PRE-TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
                    return


    def _train_step(self):
        assert self.config.NUM_TRAIN_STEP is not None
        assert self.config.NUM_TRAIN_STEP > 0

        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        self.model.train()

        losses = 0
        current_step = 0

        m_err = 0
        m_step = 0

        log.info(f"#----------- START TRAINING -----------------#")
        log.info(f"(!) Show train loss after each {self.config.show_loss_after_steps} steps")
        log.info(f"(!) Evaluate after each {self.config.eval_after_steps} steps")
        s_train_time = timer()

        while True:
            for batch in self.trainiter:
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].type(torch.long).to(self.config.DEVICE)


                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                logits = self.model(src = batch['input_ids'].to(self.config.DEVICE),
                                    trg = trg_input,
                                    src_padding_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    trg_padding_mask = label_attention_mask)


                self.optim.zero_grad()

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()

                self.optim.step()

                self.scheduler.step()
                
                losses += loss.data.item()

                current_step += 1

                if current_step % self.config.show_loss_after_steps == 0:
                    log.info(f"[Step {current_step} | {int(current_step/self.config.NUM_TRAIN_STEP*100)}% completed] Train Loss: {losses / current_step}")

                if current_step % self.config.eval_after_steps == 0 or current_step >= self.config.NUM_TRAIN_STEP:
                    eval_loss = self._evaluate()
                    res = self._evaluate_metrics()
                    err = res["ERR"]
                    log.info(f'\tTraining Step {current_step}:')
                    log.info(f'\tTrain Loss: {losses / current_step} - Val. Loss: {eval_loss:.4f}')
                    log.info(res)
                    
                    if m_err < err:
                        m_err = err
                        m_step = current_step

                    if self.SAVE:
                        if self.best_score < err:
                            self.best_score = err
                            statedict = {
                                "state_dict": self.model.state_dict(),
                                "optimizer": self.optim.state_dict(),
                                "scheduler": self.scheduler.state_dict(),
                                "step": current_step,
                                "best_score": self.best_score
                            }

                            filename = f"best_ckp.pth"
                            torch.save(statedict, os.path.join(folder,filename))
                            log.info(f"!---------Saved {filename}----------!")

                        lstatedict = {
                                    "state_dict": self.model.state_dict(),
                                    "optimizer": self.optim.state_dict(),
                                    "scheduler": self.scheduler.state_dict(),
                                    "step": current_step,
                                    "best_score": self.best_score
                                }

                        lfilename = f"last_ckp.pth"
                        torch.save(lstatedict, os.path.join(folder,lfilename))

                if current_step >= self.config.NUM_TRAIN_STEP:
                    if m_err < self.best_score:
                        m_err = self.best_score
                        m_step = -1
                    e_train_time = timer()
                    log.info(f"\n# BEST RESULT:\n\tStep: {m_step}\n\tBest ERR: {m_err:.4f}")
                    log.info(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
                    return

    
    def _evaluate(self):
        self.model.eval()
        losses = 0
        
        with torch.no_grad():
            for it, batch in enumerate(self.valiter):
                label_attention_mask = batch['label_attention_mask'].to(self.config.DEVICE)
                labels = batch['labels'].type(torch.long).to(self.config.DEVICE)

                trg_input = labels[:, :-1]
                label_attention_mask = label_attention_mask[:, :-1]

                logits = self.model(src = batch['input_ids'].to(self.config.DEVICE),
                                    trg = trg_input,
                                    src_padding_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    trg_padding_mask = label_attention_mask)

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / self.valiter_length
    
    def _build_model(self):
        self.model = Seq2SeqTransformer(num_encoder_layers = self.config.NumEncoderLayers,
                                        num_decoder_layers = self.config.NumDecoderLayers,
                                        emb_size = self.config.EmbedSize,
                                        nhead = self.config.NHEAD,
                                        src_vocab_size = len(self.tokenizer),
                                        trg_vocab_size = len(self.tokenizer),
                                        dim_feedforward = self.config.FFW)

        self.model = self.model.to(self.config.DEVICE)
    
    def build_class(self, classname):
        """
        convert string -> class
        """
        return getattr(sys.modules[__name__], classname)
    
    def _create_tokenizer(self, load_trained_bpe=False):
        if "BPE" in self.config.Tokenizer:
            if not load_trained_bpe:
                data = self._prepare_bpe_frames()
            else:
                data = None
            
            self.tokenizer = self.build_class(self.config.Tokenizer)(data, 
                                                                    self.config.bpe_step, 
                                                                    self.config.vocab_save_path, 
                                                                    self.config.max_vocab_size)

        else:
            self.tokenizer = self.build_class(self.config.Tokenizer)()

    def _prepare_bpe_frames(self):
        data = []
        for f in self.config.bpe_path:
            df = pd.read_csv(f)
            try:
                data += df["original"].tolist() + df["normalized"].tolist()
            except:
                data += df["ceg"].tolist() + df["normalized"].tolist()
        return data

   

  
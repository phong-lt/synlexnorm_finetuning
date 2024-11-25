import os
import torch
import math
from torch.utils.data import DataLoader

from .base_executor import Base_Executor

from dataset import Pretrained_LexDataset
from model import LexBARTModel, LexT5Model

from timeit import default_timer as timer

from transformers import AutoTokenizer

from logger.logger import get_logger

log = get_logger(__name__)


class Pretrained_Executor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")
    
    
    def infer(self, dataloader, max_length):
        self.model.eval()

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                
                pred = self.model.generate( input_ids = batch['input_ids'].to(self.config.DEVICE),
                                            max_length = max_length)
                
                if self.config.modeltype == "t5":
                    decoded_preds += self.tokenizer.batch_decode(self._infer_post_processing(pred.tolist()), skip_special_tokens=True)
                else:
                    decoded_preds += self.tokenizer.batch_decode(pred, skip_special_tokens=True)

                log.info(f"|===| Inferring... {it+1} it |===|")

        return decoded_preds
            
    def _create_data_utils(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_name)


        log.info("# Creating Datasets")

        if self.config.DO_PRETRAINING:
            self.pretrain_data = Pretrained_LexDataset(data_path = self.config.pretrain_data_path,
                                            tokenizer = self.tokenizer,
                                            modeltype = self.config.modeltype,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
        
        self.train_data = Pretrained_LexDataset(data_path = self.config.train_path,
                                            tokenizer = self.tokenizer,
                                            modeltype = self.config.modeltype,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
        self.val_data = Pretrained_LexDataset(data_path = self.config.val_path,
                                            tokenizer = self.tokenizer,
                                            modeltype = self.config.modeltype,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
    

    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_name)

        if self.mode == "eval":
            log.info("###Load eval data ...")
            self.val_data = Pretrained_LexDataset(data_path = self.config.val_path,
                                            tokenizer = self.tokenizer,
                                            modeltype = self.config.modeltype,
                                            batch = 256,
                                            src_max_token_len = self.config.src_max_token_len,
                                            trg_max_token_len = self.config.trg_max_token_len)
            
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
            
            self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)

        elif self.mode == "predict":
            log.info("###Load predict data ...")
            self.predict_data = Pretrained_LexDataset(data_path = self.config.predict_path,
                                            tokenizer = self.tokenizer,
                                            modeltype = self.config.modeltype,
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

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_attention_mask = label_attention_mask)


                self.optim.zero_grad()

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()

                self.optim.step()

                self.scheduler.step()
                
                losses += loss.data.item()

                current_step += 1

                if current_step % self.config.show_loss_after_pretrain_steps == 0:
                    log.info(f"[Step {current_step} | {int(current_step/self.config.NUM_PRETRAIN_STEP*100)}% completed] Train Loss: {losses / current_step}")

                if current_step % self.config.save_after_pretrain_steps == 0 or current_step >= self.config.NUM_PRETRAIN_STEP:
                    if self.SAVE:
                        lstatedict = {
                                    "state_dict": self.model.state_dict(),
                                    "optimizer": self.optim.state_dict(),
                                    "scheduler": self.scheduler.state_dict(),
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

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_attention_mask = label_attention_mask)


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

                logits = self.model(input_ids = batch['input_ids'].to(self.config.DEVICE),
                                    label_ids = trg_input,
                                    src_attention_mask = batch['src_attention_mask'].to(self.config.DEVICE),
                                    label_attention_mask = label_attention_mask)

                trg_out = labels[:, 1:]

                loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / len(list(self.valiter))
    
    def _build_model(self):
        if self.config.modeltype == "t5":
            self.model = LexT5Model(self.config.pretrained_name)
        else:
            self.model = LexBARTModel(self.config.pretrained_name)

        self.model = self.model.to(self.config.DEVICE)

   

  
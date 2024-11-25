import os
import json
import torch
import math
from torch.utils.data import DataLoader

from core.dataset import *
from core.model import *

from evaluation.err import compute_err_metrics

from transformers import set_seed
import random

from logger.logger import get_logger

log = get_logger(__name__)


class Base_Executor():
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        log.info("---Initializing Executor---")

        set_seed(config.SEED)
        random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        self.mode = mode
        self.config = config
        self.evaltype = evaltype
        self.predicttype = predicttype

        self.best_score = 0

        if self.mode == "train":
            self._create_data_utils()
            self._build_model()
            self._create_dataloader()
            self._init_training_properties()       

        if self.mode in ["eval", "predict"]:
            self._init_eval_predict_mode()
            self._build_model()
    
    def infer(self, dataloader, max_length):
        raise NotImplementedError
    
    def _create_data_utils(self):
        raise NotImplementedError

    def _init_eval_predict_mode(self):
        raise NotImplementedError  

    def _train_epoch(self, epoch):
        pass
    
    def _evaluate(self):
        raise NotImplementedError

    def _train_step(self):
        pass

    def _pretrain_step(self):
        pass

    def _build_model(self):
        raise NotImplementedError

    
    def run(self):
        if self.mode =='train':
            if self.config.DO_PRETRAINING:
                self._pretrain_step()
            self._train_step()
        elif self.mode == 'eval':
            self.evaluate()
        elif self.mode == 'predict':
            self.predict()
        else:
            exit(-1)


    def evaluate(self):
        log.info("###Evaluate Mode###")

        self._load_trained_checkpoint(self.evaltype)
        
        with torch.no_grad():
            log.info(f'Evaluate val data ...')

            res = self._evaluate_metrics()
            log.info(res)
    
    def predict(self): 
        log.info("###Predict Mode###")

        self._load_trained_checkpoint(self.predicttype)

        log.info("## START PREDICTING ... ")

        if self.config.get_predict_score:
            results, scores = self._evaluate_metrics()
            log.info(f'\t#PREDICTION:\n')
            log.info(f'\t{scores}')
        else:
            preds = self.infer(self.predictiter, self.config.max_predict_length)
            results = [{"pred": p} for p in preds]


        if self.config.SAVE_PATH:
            with open(os.path.join(self.config.SAVE_PATH, "results.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            log.info("Saved Results !")
        else:
            with open(os.path.join(".","results.csv"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            log.info("Saved Results !")
    
    def _create_dataloader(self):
        log.info("# Creating DataLoaders")

        if self.config.DO_PRETRAINING:
            self.pretrainiter = DataLoader(dataset = self.pretrain_data, 
                                    batch_size=self.config.PRETRAIN_BATCH_SIZE, 
                                    shuffle=True)
       
        self.trainiter = DataLoader(dataset = self.train_data, 
                                    batch_size=self.config.TRAIN_BATCH_SIZE, 
                                    shuffle=True)
        self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)
    
    
    def _init_training_properties(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, betas=self.config.BETAS, eps=1e-9)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
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

    def _load_trained_checkpoint(self, loadtype):
        if os.path.isfile(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth"))
            try:
                log.info(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])

        elif os.path.isfile(os.path.join('./models', f"{loadtype}_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join('./models', f"{loadtype}_ckp.pth"))
            try:
                log.info(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
        
        else:
            raise Exception(f"(!) {loadtype}_ckp.pth is required (!)")
    

    def _infer_post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.tokenizer.eos_token_id)])
            except:
                res.append(out)

        return res


    def _evaluate_metrics(self):
        if self.mode == "predict":
            preds = self.infer(self.predictiter, self.config.max_predict_length)
            gts = [i.strip() for i in self.predict_data.trg]
            raw_srcs = [i.strip() for i in self.predict_data.src]
        else:
            preds = self.infer(self.valiter, self.config.max_eval_length)
            gts = [i.strip() for i in self.val_data.trg]
            raw_srcs = [i.strip() for i in self.val_data.src]

        preds = [i.strip() for i in preds]

        if self.mode == "predict":
            result = [{
                "pred": pred,
                "gt": gt,
                "raw_src": raw_src
            } for pred, gt, raw_src in zip(preds, gts, raw_srcs)]

            return result, compute_err_metrics(raw_srcs, gts, preds)

        return compute_err_metrics(raw_srcs, gts, preds)


  
import torch
import editdistance
from edit_distance import SequenceMatcher
import pandas as pd
#from core.model.utils.greedy import normalize
from tqdm import tqdm

def compute_precision_recall(y_true, y_pred):
  # create a SequenceMatcher object with y_true and y_pred
  sm = SequenceMatcher(a=y_true, b=y_pred)

  # get the opcodes from the SequenceMatcher object
  opcodes = sm.get_opcodes()

  # initialize the counts for true positives, false positives, and false negatives
  tp = 0
  fp = 0
  fn = 0

  # loop through the opcodes
  for op, i1, i2, j1, j2 in opcodes:
    # if the operation is equal, increment tp by the number of words
    if op == "equal":
      tp += i2 - i1
    # if the operation is replace, increment fp and fn by the number of words
    elif op == "replace":
      fp += j2 - j1
      fn += i2 - i1
    # if the operation is insert, increment fp by the number of words
    elif op == "insert":
      fp += j2 - j1
    # if the operation is delete, increment fn by the number of words
    elif op == "delete":
      fn += i2 - i1

  # compute precision and recall
  if tp + fp == 0:
      precision = 0
  else:
      precision = tp / (tp + fp)

  if tp+fn==0:
      recall = 0
  else:
      recall = tp / (tp + fn)

  return precision, recall

def compute_err_metrics(src_data, trg_data, prediction):
    assert len(src_data) == len(trg_data)
    assert len(src_data) == len(prediction)

    system_error = 0
    baseline_error = 0
    num_word = 0

    recall_total = []
    prec_total = []
    for i in range(len(src_data)):
            
        y_pred = prediction[i].split()  
        y_true = trg_data[i].split()
        y_src = src_data[i].split()

        ## compute word error
        system_error += editdistance.eval(y_true, y_pred)
        baseline_error += editdistance.eval(y_true, y_src)
        num_word += len(y_true)

        prec, recall = compute_precision_recall(y_true, y_pred)

        prec_total.append(prec)
        recall_total.append(recall)
        


    system_accuracy = 1 - system_error/num_word
    baseline_accuracy = 1 - baseline_error/num_word

    err = (system_accuracy - baseline_accuracy)/(1 - baseline_accuracy)

    return {"ERR": err,
            "Precision": sum(prec_total)/len(prec_total),
            "Recall": sum(recall_total)/len(recall_total)}
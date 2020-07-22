from __future__ import print_function

import numpy as np
import torch
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment


def _original_match(flat_preds, flat_targets, preds_k, targets_k):
  # map each output channel to the best matching ground truth (many to one)

  assert (isinstance(flat_preds, torch.Tensor) and
          isinstance(flat_targets, torch.Tensor) and
          flat_preds.is_cuda and flat_targets.is_cuda)

  out_to_gts = {}
  out_to_gts_scores = {}
  for out_c in range(preds_k):
    for gt_c in range(targets_k):
      # the amount of out_c at all the gt_c samples
      tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())
      if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
        out_to_gts[out_c] = gt_c
        out_to_gts_scores[out_c] = tp_score

  return list(out_to_gts.iteritems())


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
# examples
# num_samples-num_correct = [[3533977. 3468352. 3375877. 3450522. 3514784. 3532520. 3533866. 3534090.
#   3534208. 3534266.]
#  [3534297. 3534260. 3529042. 3511714. 3514639. 3528732. 3531849. 3533168.
#   3533994. 3534172.]
#  [3534285. 3534259. 3522231. 3390835. 3199042. 3165567. 3255379. 3494807.
#   3526289. 3531087.]
#  [3534297. 3534068. 3528525. 3519955. 3526861. 3533227. 3534066. 3534219.
#   3534241. 3534290.]
#  [3534261. 3529552. 3512813. 3515477. 3526448. 3531950. 3532818. 3533338.
#   3533519. 3533400.]
#  [3514371. 3281918. 3439775. 3493826. 3526944. 3532172. 3533007. 3533711.
#   3534163. 3534244.]
#  [3182969. 3456450. 3519110. 3525824. 3528869. 3530883. 3532264. 3532374.
#   3533587. 3533543.]
#  [3522316. 3498384. 3479232. 3506129. 3517050. 3521282. 3526896. 3529765.
#   3530473. 3529340.]
#  [3534180. 3521898. 3457591. 3408838. 3396023. 3472560. 3512471. 3517423.
#   3524985. 3526379.]
#  [3467126. 3422835. 3475034. 3499581. 3511382. 3521109. 3524273. 3528253.
#   3531940. 3531216.]]
# match = [[0 2]
#  [1 7]
#  [2 5]
#  [3 8]
#  [4 9]
#  [5 1]
#  [6 0]
#  [7 6]
#  [8 4]
#  [9 3]]
# refer : https://gemfury.com/stream/python:scikit-learn/-/content/utils/linear_assignment_.py
#         https://kite.com/python/docs/sklearn.utils.linear_assignment_.linear_assignment
#         https://brilliant.org/wiki/hungarian-matching/
#         https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
  assert (isinstance(flat_preds, torch.Tensor) and
          isinstance(flat_targets, torch.Tensor) and
          flat_preds.is_cuda and flat_targets.is_cuda)

  num_samples = flat_targets.shape[0]
  print('num_samples',num_samples)

  assert (preds_k == targets_k)  # one to one
  num_k = preds_k
  num_correct = np.zeros((num_k, num_k))

  for c1 in range(num_k):
    for c2 in range(num_k):
      # elementwise, so each sample contributes once
      votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
      num_correct[c1, c2] = votes

  # num_correct is small
  match = linear_assignment(num_samples - num_correct)

  # return as list of tuples, out_c to gt_c
  res = []
  for out_c, gt_c in match:
    res.append((out_c, gt_c))

  return res


def _acc(preds, targets, num_k, verbose=0):
  assert (isinstance(preds, torch.Tensor) and
          isinstance(targets, torch.Tensor) and
          preds.is_cuda and targets.is_cuda)

  if verbose >= 2:
    print("calling acc...")

  assert (preds.shape == targets.shape)
  assert (preds.max() < num_k and targets.max() < num_k)

  acc = int((preds == targets).sum()) / float(preds.shape[0])

  return acc


def _nmi(preds, targets):
  return metrics.normalized_mutual_info_score(targets, preds)


def _ari(preds, targets):
  return metrics.adjusted_rand_score(targets, preds)

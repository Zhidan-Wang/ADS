import copy
import json
import os
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from scipy.sparse import coo_matrix

file1 = 'dataset/book-crossing/book-crossing.inter1'
file2 = 'dataset/book-crossing/book-crossing.inter'

with open(file1, 'r') as f:
    lines = f.readlines()
with open(file2, 'w') as fn:
    fn.write('user_id:token' + '\t' + 'item_id:token' + '\t' + 'rating:token' + '\n')
    for i, line in enumerate(lines):
        u, i, r = line.strip().split(' ')
        print(u)
        print(i)
        print(r)

        fn.write(str(u) + '\t' + str(i) + '\t' + str(r) + '\n')
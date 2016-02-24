# coding: utf-8

# Author: Hussein AL-NATSHEH <hussein.al-natsheh@ish-lyon.cnrs.fr>
# License: BSD 3 clause
# 2016

import pandas as pd
import argparse
import numpy as np

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='sts', type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--glovefile", default='data/glove.6B.300d.tar.gz', type=str)
    args = parser.parse_args()

    X, y = load_dataset (args.dataset, args.verbose)
    gloveb300d = load_glove(args.glovefile, args.verbose)
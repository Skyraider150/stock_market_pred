#Import dependencies

# Provides regular expression matching operations
import re
# Provides various time-related functions
import time

import csv

import sys

import nltk

from functools import reduce
# Supplies classes for manipulating dates and times
from datetime import datetime
# Represents filesystem paths
from pathlib import Path

# Fundamental package for scientific computing
import numpy as np
# Python Data Analysis Library
import pandas as pd
# Calculates a Pearson correlation coefficient for testing non-correlation.
from scipy.stats import pearsonr
# Transforms features by scaling each feature to a given range.
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Data visualization tool
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
# Data visualization tool
import seaborn as sns
sns.set()
# Provides a portable way of using operating system dependent functionality
import os
# Install packages
#!echo Imported
os.system("echo Imported")

# Check files
os.system("ls -R")

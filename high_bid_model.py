#import tensorflow as tf
import numpy as np
import pandas as pd
#import keras as keras
#from keras import Sequential
#from keras.layers import Dense

from database import get_ticker_stream
from utilities import vectorize_ticker_stream

raw_ticker_stream = get_ticker_stream('/home/cole/Code/KrakenTrading/ticker-1')
ticker_stream, all_pairs_sorted = vectorize_ticker_stream(raw_ticker_stream[:100000]) # 200000 is approximately 28 gb of memory, 100000 is approximately 17gb
print(len(ticker_stream))
print('---------------')
print(ticker_stream[0:10])
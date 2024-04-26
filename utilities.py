import requests
import functools
import datetime

import pandas as pd
import numpy as np

SECONDS_IN_DAY = 24*60*60

def get_result(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        error = data['error']

        if error:
            print("API Error:", error)
            return None

        currency_pairs = data['result']
        return currency_pairs

    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
        return None

def flatten_lists(list_of_lists):
    return functools.reduce(lambda x,y: x+y, list_of_lists)

def vectorize_ticker_stream(ticker=[]):
    """
    Until every pair has had an example processed, we won't be able 
    to vectorize the whole thing. This implies that we must throw out 
    some early samples until each pair has been encountered at least once.
    """
    

    all_pairs = set()
    for item in ticker:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        all_pairs.add(pair)

    pairs_so_far = set()
    last_known_pair_info = {}
    for i, item in enumerate(ticker):
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        pairs_so_far.add(pair)

        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        last_known_pair_info[pair] = flattened_data

        if not all_pairs.difference(pairs_so_far):
            break

    all_pairs = sorted(all_pairs) # enforce ordering of data
    # by this point last_known_pair_info has every pair populated.
    vectorized_tickers = []
    for item in ticker[i:]:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        last_known_pair_info[pair] = flattened_data

        timestamp = datetime.datetime.fromisoformat(time_received).timestamp()
        vectorized_data = [timestamp]
        for pair in all_pairs:
            flattened_data = last_known_pair_info[pair]
            vectorized_data.extend(flattened_data)
        vectorized_tickers.append(vectorized_data)
    
    return vectorized_tickers, all_pairs

def equal_time_spacing(X, t_space=0):
    """
    NOT FINISHED

    Assumes X[:, 0] are timestamps, and X is sorted by time.

    Given t, where t_space is a float between (0, inf)

    Warning: If t_space is too large examples will be skipped. 
    
    Default is 0, which will cause t = min(abs(t_i - t_j)) for all i, j
    """
    if t_space < 0:
        raise ValueError("t must be >= 0")
    
    if t_space == 0:
        sorted_X_0 = X[X[:, 0].argsort(), 0]
        X_0_diff = np.diff(sorted_X_0)
        sorted_X_0_diff = X_0_diff[X_0_diff.argsort()]
        t_space = sorted_X_0_diff[0] # minimum diff
    
    num_examples = X.shape[0]

    min_t = X[0][0]
    t = min_t
    i = 0
    X_ret = []
    while i < num_examples:
        x_i = X[i]
        X_ret.append(x_i)
        t = t + t_space
        next_t = X[i+1][0]
        while t < next_t:
            # append previous vector with new t
            x_i_copy = x_i.copy()
            x_i_copy[0] = t
            X_ret.append(x_i_copy)
            t = t + t_space
        # if the next t falls in between t+1 and t+2, just append t_1 as filler
        # if t increment overshot t+1, use t+2? (Can this happen?)

def vectorize_windows(X, window_length, stride=1, drop_extras=True):
    num_examples = X.shape[0]
    
    X_ret = []
    window_base = 0
    while window_base + window_length <= num_examples:
        X_ret.append(np.hstack([x for x in X[window_base:window_base+window_length]]))
        window_base = window_base + stride

    if not drop_extras:
        X_ret.append(np.hstack([x for x in X[window_base:window_base+window_length]]))
        window_base = window_base + stride

    return np.array(X_ret)

def timestamp_to_percent(column):

    if not (isinstance(column, (pd.Series, np.ndarray))):
        raise ValueError("Column must be ndarray or dataframe")
    
    # put timestamp on interval [0, 1]
    percent_of_day_column = (column % SECONDS_IN_DAY) / SECONDS_IN_DAY
    return percent_of_day_column
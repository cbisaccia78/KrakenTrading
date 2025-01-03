import requests
import functools
import datetime

import pandas as pd
import numpy as np

FEATURE_MAP = {
    'best_ask': 0, 'ask_whole_lot_volume': 1, 'ask_lot_volume': 2,
    'best_bid': 3, 'bid_whole_lot_volume': 4, 'bid_lot_volume': 5,
    'close_price': 6, 'close_lot_volume': 7,
    'value_today': 8, 'value_last_24': 9,
    'vol_weight_avg_today': 10, 'vol_weight_avg_last_24': 11,
    'num_trades_today': 12, 'num_trades_last_24': 13,
    'low_price_today': 14, 'low_price_last_24': 15,
    'high_price_today': 16, 'high_price_last_24': 17,
    'open_price_today': 18, 'open_price_last_24': 19
}

FEATURES_PER_PAIR = len(FEATURE_MAP)

SECONDS_IN_DAY = 24*60*60

PING = {
    "event": "ping",
    "reqid": 69420
}

LARGEST_MODEL_SIZE = 4459

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

def update_pair_cache(raw_ticker_stream, cache):
    # populate the cache with the most recent value of each pair
    for item in raw_ticker_stream:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']

        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        cache[pair] = flattened_data

def vectorize_from_cache(cache, raw_ticker_stream):
    """
    for each new ticker, update the cache
    with this new value and then vectorize all of the tickers using the 
    cached values for each ticker. 
    """
    vectorized_tickers = []

    pair_names = cache.keys()

    for item in raw_ticker_stream:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        cache[pair] = flattened_data

        timestamp = datetime.datetime.fromisoformat(time_received).timestamp()
        vectorized_data = [timestamp]
        for pair in pair_names:
            flattened_data = cache[pair]
            vectorized_data.extend(flattened_data)
        vectorized_tickers.append(vectorized_data)
    
    return vectorized_tickers

def vectorize_ticker_stream(raw_ticker_stream=[]):
    """
    First, gather a cache of the most recent ticker info of each pair.

    Then, once this cache is gathered: for each new ticker, update the cache
    with this new value and then vectorize all of the tickers using the 
    cached values for each ticker. 

    Until every pair has had an example processed at least once, we won't be able 
    to vectorize the whole cache. This implies that we wont have labels for all the early
    examples encountered.
    """
    
    # determine all the pairs that exist
    all_pairs = set()
    for item in raw_ticker_stream:
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        all_pairs.add(pair)

    # populate the cache with the most recent value of each pair, breaking once 
    # each pair has been encountered once
    pairs_so_far = set()
    pair_cache = {}
    for i, item in enumerate(raw_ticker_stream):
        time_received = list(item.keys())[0]
        pair = item[time_received]['pair']
        pairs_so_far.add(pair)

        data = item[time_received]['data']
        values = data.values()
        flattened_data = flatten_lists(values)
        pair_cache[pair] = flattened_data

        if not all_pairs.difference(pairs_so_far):
            break
    
    pair_cache = dict(sorted(pair_cache.items())) # enforce ordering of data
    # by this point last_known_pair_info has every pair populated.
    # now for each new example vectorize it using the cache
    vectorized_tickers = vectorize_from_cache(pair_cache, raw_ticker_stream[i:])
    
    return vectorized_tickers, pair_cache

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

def vectorize_window(X, window_length, stride=1, drop_extras=True):
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

def top_correlated_features(X, y, num_to_keep=10):

    # Add a small noise to the data so variances aren't zero
    epsilon = 1e-6
    X_noisy = X + epsilon * np.random.randn(*X.shape)
    y_noisy = y + epsilon * np.random.randn(*y.shape)

    # Calculate the absolute correlation coefficients between each feature and F
    correlations = np.abs(np.corrcoef(X_noisy, y_noisy, rowvar=False)[-1, :-1])

    # Get the indices of the features with the highest correlation coefficients
    top_feature_indices = np.argsort(correlations)[-num_to_keep:]

    top_feature_tuples = [(idx, correlations[idx]) for idx in top_feature_indices]

    return top_feature_tuples

def explore_correlated_features(raw_ticker_stream, target_pair, num_to_keep=10, stride=LARGEST_MODEL_SIZE):
    num_examples = len(raw_ticker_stream)
    i = 0
    end = num_examples - stride

    top_index_scores = {}

    while i <= end:
        ticker_stream, pair_cache = vectorize_ticker_stream(raw_ticker_stream[i:i+stride])

        # get pair index
        pair_index = list(pair_cache.keys()).index(target_pair)
        features_per_pair = len(FEATURE_MAP)
        bid_index = FEATURE_MAP['best_bid']
        offset_index = pair_index*features_per_pair + 1 # +1 because of time index
        target_index = offset_index + bid_index

        # convert to numpy array
        X = np.array(ticker_stream, dtype=np.float32)

        # numeric timestamps
        X[:, 0] = timestamp_to_percent(X[:, 0])

        y = X[:, target_index]

        top_feature_tuples = top_correlated_features(X, y, num_to_keep)

        for idx, corrcoeff in top_feature_tuples:
            if idx not in top_index_scores:
                top_index_scores[idx] = 0.0
            
            top_index_scores[idx] = top_index_scores[idx] + corrcoeff

        i = i + stride
    
    print(dict(sorted(top_index_scores.items(), key=lambda item: item[1], reverse=True)))

def binary_transform(y, predicate):
    """
    For each y_i, set y_i = predicate(y_i)
    """

    result = np.where(predicate(y), 1.0, 0.0)
    return result

def create_regression_labels(X, window_length, feature_index):
    return X[window_length:, feature_index]

def create_classification_labels(X, window_length, feature_index):
    # one hot encode down, stay, up
    y_iter = X[window_length-1:, feature_index] # minus 1 because we need previous value to determine if it changed
    y_num_examples = y_iter.shape[0] - 1
    y = np.zeros(y_num_examples)
    for i in range(0, y_num_examples):
        diff = y_iter[i+1] - y_iter[i]
        if diff < 0:
            y[i] = 0
        elif diff == 0:
            y[i] = 1
        else:
            y[i] = 2
    y = y.astype(int)
    return np.eye(3)[y]
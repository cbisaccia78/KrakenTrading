import numpy as np
import keras as keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from database import get_ticker_stream
from utilities import vectorize_ticker_stream, timestamp_to_percent, vectorize_windows

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

raw_ticker_stream = get_ticker_stream('/Users/dancing_ghosts/Code/WebApps/KrakenTrading/ticker-1-test')
ticker_stream, all_pairs_sorted = vectorize_ticker_stream(raw_ticker_stream) # 200000 is approximately 28 gb of memory, 100000 is approximately 17gb

btc_index = all_pairs_sorted.index('XBT/USD')
num_pairs = len(all_pairs_sorted)
features_per_pair = 20

# create evenly spaced samples over time (not implemented yet)

X = np.array(ticker_stream, dtype=np.float32)

X[:, 0] = timestamp_to_percent(X[:, 0])

# train test validation split
num_examples = X.shape[0]

train_valid_split = int(0.8*num_examples)

X_train_valid = X[0:train_valid_split]

train_split = int(0.8*train_valid_split)
X_train = X_train_valid[0:train_split]
X_valid = X_train_valid[train_split:]

X_test = X[train_valid_split:]

# Standardize features
scalar = StandardScaler()
X_train_std = scalar.fit_transform(X_train)
X_valid_std = scalar.transform(X_valid)
X_test_std = scalar.transform(X_test)

window_len = 5
stride = 1

# create labels
bid_index = FEATURE_MAP['best_bid']
btc_bid_index = btc_index*features_per_pair + bid_index

y_train_std = X_train_std[window_len:, btc_bid_index]
y_valid_std = X_valid_std[window_len:, btc_bid_index]
y_test_std = X_test_std[window_len:, btc_bid_index]

# create windows
X_train_std = vectorize_windows(X_train_std, 5, stride)
X_valid_std = vectorize_windows(X_valid_std, 5, stride)
X_test_std = vectorize_windows(X_test_std, 5, stride)

# remove final row (no label for it)
X_train_std = X_train_std[:-1]
X_valid_std = X_valid_std[:-1]
X_test_std = X_test_std[:-1]

batch_size = 32

# create datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train_std, y_train_std)).batch(batch_size)
valid_ds = tf.data.Dataset.from_tensor_slices((X_valid_std, y_valid_std)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_std, y_test_std)).batch(batch_size)


# create model
num_input_parameters = X_train_std.shape[1]
model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(None, num_input_parameters)),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit training data
history = model.fit(train_ds, epochs=10, validation_data=valid_ds)

# evaluate test data
test_loss, test_mse = model.evaluate(test_ds)
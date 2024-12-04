import numpy as np
import keras as keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping

from utilities import vectorize_ticker_stream, timestamp_to_percent, vectorize_window, create_regression_labels, FEATURE_MAP, FEATURES_PER_PAIR

def create_model(raw_ticker_stream, pair_name, window_len=5, stride=1, generate_y=create_regression_labels, output_activation='linear', loss='mse', metric='mse', output_size=1):

    ticker_stream, pair_cache = vectorize_ticker_stream(raw_ticker_stream) # 200000 is approximately 28 gb of memory, 100000 is approximately 17gb

    pair_index = list(pair_cache.keys()).index(pair_name)

    # create labels
    bid_index = FEATURE_MAP['best_bid']
    pair_bid_index = 1 + pair_index*FEATURES_PER_PAIR + bid_index # add 1 because of timestamp

    # convert to numpy array
    X = np.array(ticker_stream, dtype=np.float32)

    # numeric timestamps
    X[:, 0] = timestamp_to_percent(X[:, 0])

    # train test validation split
    num_examples = X.shape[0]

    test_split_idx = int(0.8*num_examples)

    valid_split_idx = int(0.8*test_split_idx)

    y = generate_y(X, window_len, pair_bid_index)

    # Standardize features
    scalar = StandardScaler()
    X_std = scalar.fit_transform(X)


    BATCH_SIZE = 8

    train_ds = keras.utils.timeseries_dataset_from_array(
        X_std, 
        y,
        sequence_length=window_len,
        batch_size=BATCH_SIZE,
        end_index=valid_split_idx
    )

    valid_ds = keras.utils.timeseries_dataset_from_array(
        X_std, 
        y,
        sequence_length=window_len,
        batch_size=BATCH_SIZE,
        start_index=valid_split_idx,
        end_index=test_split_idx
    )

    test_ds = keras.utils.timeseries_dataset_from_array(
        X_std, 
        y,
        sequence_length=window_len,
        batch_size=BATCH_SIZE,
        start_index=test_split_idx
    )


    # create model
    num_input_parameters = X_std.shape[1]

    inputs = keras.Input(shape=(window_len, num_input_parameters))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.3)(x)
    outputs = keras.layers.Dense(output_size, activation=output_activation)(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss=loss, metrics=[metric])

    early_stopping_callback = EarlyStopping(monitor=f'val_recall', patience=5, restore_best_weights=True)

    original_y_train_labels = np.argmax(y[:valid_split_idx], axis=1)
    class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(original_y_train_labels),
                                                  y=original_y_train_labels)

    # Convert to a dictionary to pass to Keras
    class_weights_dict = dict(enumerate(class_weights))

    # fit training data
    _ = model.fit(train_ds, class_weight=class_weights_dict, epochs=250, validation_data=valid_ds, callbacks=[early_stopping_callback])

    # evaluate test data
    _, test_metric = model.evaluate(test_ds)

    print(f'test_{metric} : {test_metric}')

    return model, test_metric, scalar, pair_cache
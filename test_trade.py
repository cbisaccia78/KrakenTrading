import numpy as np

from high_bid_model import create_model
from utilities import timestamp_to_percent, vectorize_window, vectorize_from_cache, update_pair_cache, FEATURE_MAP, create_regression_labels, create_classification_labels
from database import get_ticker_stream
from keras.metrics import Recall

window_length = 5

pair_cache = {}

pair_name = 'XBT/USD'

examples_processed = 0
examples_received = 0

NUM_EXAMPLES = 4459

total_raw_ticker_stream = get_ticker_stream('/home/cole/Code/KrakenTrading/ticker-2-model')
raw_ticker_stream = total_raw_ticker_stream[:NUM_EXAMPLES]

#create model
model, test_mse, standard_scalar, pair_cache = create_model(
    raw_ticker_stream, pair_name, window_len=window_length, 
    generate_y=create_classification_labels, output_activation='softmax',
    loss='categorical_crossentropy', metric=Recall(), output_size=3)

# save index of pair bid to predict
pair_index = list(pair_cache.keys()).index(pair_name)
features_per_pair = len(FEATURE_MAP)
bid_index = FEATURE_MAP['best_bid']
offset_index = pair_index*features_per_pair + 1 # +1 because of time index
pair_bid_index = offset_index + bid_index
print(pair_bid_index)

# save mean/std of pair bid to predict

mean = standard_scalar.mean_[pair_bid_index]
std = standard_scalar.scale_[pair_bid_index]
print('mean xbt: ', mean)
print('std: xbt', std)

#updated examples_processed
examples_processed = NUM_EXAMPLES
examples_received = NUM_EXAMPLES

last_prediction = None
last_pair_bid = None
errors = []


for i in range(NUM_EXAMPLES, len(total_raw_ticker_stream)):
    # add more data to raw_ticker_stream
    raw_ticker_stream.append(total_raw_ticker_stream[i])
    examples_received = examples_received + 1

    
    _examples_received = examples_received # just in case this thread gets put to sleep and examples received changes, we don't want to get out of sync
    
    # update the pair cache for all examples that came in while model was trading / sleeping,
    # which will not be apart of the window
    last_missed_example_index = _examples_received - window_length
    num_missed_examples = last_missed_example_index - examples_processed
    if num_missed_examples > 0:
        update_pair_cache(raw_ticker_stream[examples_processed:last_missed_example_index], pair_cache)

    # grab the most recent window_length examples
    example_window = raw_ticker_stream[last_missed_example_index:]
    # vectorize examples from the cache
    example_window = vectorize_from_cache(pair_cache, example_window) # this call will also update pair cache with recent examples

    #convert to numpy array
    x = np.array(example_window, dtype=np.float32)
    x[:, 0] = timestamp_to_percent(x[:, 0])

    # standardize using same mean/std from creation of model
    x = standard_scalar.transform(x)

    # concatenate each example from the example window into one large vector
    x = vectorize_window(x, window_length)

    # Reshape the example to add a batch dimension
    x = np.array([x])#np.reshape(x, (1,) + x.shape)

    # get model prediction
    prediction = model(x) # tensor of shape (1,1,1)
    
    
    value = prediction[0].numpy()
    value = np.argmax(value) # since it's one-hot encoded
    # value = (value*std) + mean # need to un-standardize this value to get the actual value to trade
    value = value.item()
    """print('----------------')
    print(f'{pair_name} at next ticker update: ', value)
    print('----------------\n')"""

    current_pair_bid = float(pair_cache[pair_name][bid_index])

    # TODO - this monitoring of the model should happen on a different thread?
    # grab the last predicted value and compare it to the current ticker value
    if last_prediction is not None:
        """ # calculate error from previous prediction with current bid price
        errors.append(float(current_pair_bid) - last_prediction) # positive == underestimated """
        diff = current_pair_bid - last_pair_bid
        if diff < 0:
            actual_y = 0
        elif diff == 0:
            actual_y = 1
        else:
            actual_y = 2

        errors.append(1.0 if actual_y != last_prediction else 0.0)
        if actual_y != 1:
            print('----------------')
            print(f'predicted: {last_prediction}')
            print(f'actual: {actual_y}')
            print(f'mean error: {np.mean(np.array(errors))}')
            print('----------------\n')

    last_prediction = value
    last_pair_bid = current_pair_bid

    # update examples processed
    examples_processed = _examples_received # use local value in case global one was updated while this thread was sleeping
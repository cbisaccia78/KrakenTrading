import websocket
import json
import datetime
import threading
import signal
import time
import sys

import numpy as np

from api import get_all_wsnames
from high_bid_model import create_model
from utilities import vectorize_from_cache, timestamp_to_percent, vectorize_window, update_pair_cache, FEATURE_MAP

# TODO - move these to api module
PUBLIC_URL = "wss://ws.kraken.com/" 
PRIVATE_URL = "wss://ws-auth.kraken.com/"

ACCEPTED_ERROR = 0.01

features_per_pair = len(FEATURE_MAP)
pair_name = 'XBT/USD'

window_length = 5

model = None
model_thread = None
model_thread_stop = False

retrain = False

raw_ticker_stream = []
pair_cache = {}

last_prediction = None
errors = []

examples_processed = 0
examples_received = 0

NUM_EXAMPLES = 4459

def model_thread_func():
    """
    Separate thread of execution which, upon reaching NUM_EXAMPLES messages, will create a model from these examples.
    As the model is being created, more data will be streaming in. Call this extra data X_extra. 
    After the model is created, the thread updates to the most recent X_extra examples and start prediction. 

    Question: Should there be "backfilling" of this unused X_extra data? Presumably this thread could online-train 
    this extra data in between attempts at predicting the most recent data, considering that we are throttled by 
    120 trades per minute. On first cut this data will just be lost. 

    The thread should also be checking the mse score for incoming data, and if the score deteriorates past a certain threshold,
    should tell the model thread to train a new model with more recent data. This implies that data that we run predictions on should
    be stored for future model training. 
    """

    global model
    global model_thread_stop
    global raw_ticker_stream
    global pair_cache
    global examples_processed
    global examples_received
    global retrain
    global errors
    global last_prediction
    
    while not model_thread_stop:

        if model is None: # need to create model
                if examples_received >= NUM_EXAMPLES:
                    #create model
                    model, test_mse, standard_scalar, pair_cache = create_model(raw_ticker_stream, pair_name, window_len=window_length)
                    
                    # save index of pair bid to predict
                    pair_index = list(pair_cache.keys()).index(pair_name)
                    features_per_pair = len(FEATURE_MAP)
                    bid_index = FEATURE_MAP['best_bid']
                    offset_index = pair_index*features_per_pair + 1 # +1 because of time index
                    pair_bid_index = offset_index + bid_index

                    # save mean/std of pair bid to predict

                    mean = standard_scalar.mean_[pair_bid_index]
                    std = standard_scalar.scale_[pair_bid_index]
                    print('mean xbt: ', mean)
                    print('std: xbt', std)

                    #updated examples_processed
                    examples_processed = NUM_EXAMPLES
        
        elif retrain: # model is not performing well
            examples_received = 0
            examples_processed = 0
            model = None

        else: # we can trade

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
            x = np.array([x])

            # get model prediction
            prediction = model(x) # tensor of shape (1,1,1)
            # need to un-standardize this value to get the actual value to trade
            
            value = prediction[0].numpy()
            value = (value*std) + mean
            value = value.item()
            """print('----------------')
            print(f'{pair_name} at next ticker update: ', value)
            print('----------------\n')"""

            # TODO - this monitoring of the model should happen on a different thread?
            # grab the last predicted value and compare it to the current ticker value
            if last_prediction is not None:
                # calculate error from previous prediction with current bid price
                current_pair_bid = pair_cache[pair_name][bid_index]
                errors.append(float(current_pair_bid) - last_prediction) # positive == underestimated
                print('----------------')
                print(f'predicted: {last_prediction}')
                print(f'actual: {current_pair_bid}')
                print(f'mean error: {np.mean(np.array(errors))}')
                print('----------------\n')
            last_prediction = value

            examples_processed = _examples_received # use local value in case global one was updated while this thread was sleeping


model_thread = threading.Thread(target=model_thread_func)
model_thread.start()

# Function to handle incoming messages
def on_message(ws, message):
    global examples_received
    global raw_ticker_stream
    # Parse the incoming message (assuming it's JSON)
    result = json.loads(message)
    # Process the data as needed
    data = result[1]
    pair = result[3]
    now = str(datetime.datetime.now())
    raw_ticker_stream.append({now: {'pair': pair, 'data': data}})
    examples_received = examples_received + 1
    print(examples_received)

# Function to handle WebSocket open event
def on_open(ws):
    print("WebSocket connection opened")
    'XBT/USD'
    # Subscribe to the desired channel(s) after the connection is open
    # Send the subscription message to the WebSocket server
    subscription_message = {
        "event": "subscribe",
        "pair": get_all_wsnames(),
        "subscription": {
            "name": "ticker"
        }
    }
    ws.send(json.dumps(subscription_message))

# Function to handle WebSocket close event
def on_close(ws):
    print("Closed websocket")

ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open, on_close=on_close)

def ws_thread_func():
    global ws
    # Create a WebSocket connection
    # Start the WebSocket connection (runs in a separate thread)
    ws.run_forever() # this hangs until websocket is stopped

    

ws_thread = threading.Thread(target=ws_thread_func)
ws_thread.start()



def signal_handle_func(sig, frame):
    global ws_thread
    global model_thread
    global model_thread_stop
    global ws

    # Close websocket gracefully
    ws.close()
    ws_thread.join()

    model_thread_stop = True
    model_thread.join()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handle_func)

while True:
    time.sleep(1)
    


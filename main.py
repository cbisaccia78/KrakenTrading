import websocket
import json
import datetime

from api import get_all_wsnames
from high_bid_model import create_model

PING = {
    "event": "ping",
    "reqid": 69420
}

PUBLIC_URL = "wss://ws.kraken.com/"
PRIVATE_URL = "wss://ws-auth.kraken.com/"

with open('./ticker-1-test', 'a') as ticker_fp:

    # Function to handle incoming messages
    def on_message(ws, message):
        # Parse the incoming message (assuming it's JSON)
        result = json.loads(message)
        # Process the data as needed
        data = result[1]
        pair = result[3]
        now = str(datetime.datetime.now())
        ticker_fp.write(json.dumps({now: {'pair': pair, 'data': data}}) + '\n')

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
        print("WebSocket connection closed")

    count = 0

    NUM_EXAMPLES = 4459

    # Create a WebSocket connection
    ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open, on_close=on_close)

    """
    Need a separate thread of execution here which, upon reaching NUM_EXAMPLES messages, will create a model from these examples.
    As the model is being created, more data will be streaming in. Call this extra data X_extra. 
    After the model is created, the main thread should go to the most recent X_extra examples and start prediction. 

    Question: Should there be "backfilling" of this unused X_extra data? Presumably this thread could online-train 
    this extra data in between attempts at predicting the most recent data, considering that we are throttled by 
    120 trades per minute. On first cut this data will just be lost. 

    THe main thread should also be checking the mse score for incoming data, and if the score deteriorates past a certain threshold,
    should tell the model thread to train a new model with more recent data. This implies that data that we run predictions on should
    be stored for future model training. 
    """

    # Start the WebSocket connection (runs in a separate thread)
    ws.run_forever() # this hangs until websocket is stopped

    # Close websocket gracefully
    ws.close()
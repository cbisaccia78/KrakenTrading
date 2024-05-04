import websocket
import json
import datetime

from api import get_all_wsnames

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

    # Create a WebSocket connection
    ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_message=on_message, on_open=on_open, on_close=on_close)

    # Start the WebSocket connection (runs in a separate thread)
    ws.run_forever() # this hangs until websocket is stopped

    # Close websocket gracefully
    ws.close()
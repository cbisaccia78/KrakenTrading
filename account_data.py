import time

from authentication import kraken_request

# Construct the request and print the result


def getAccountBalance():
    resp = kraken_request('/0/private/Balance', {
        "nonce": str(int(1000*time.time()))
    })
    return resp

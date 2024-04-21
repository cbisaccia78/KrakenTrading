from utilities import get_result

"""
--------------------------------------------------------------------------------------------------------------------------------

PUBLIC 

--------------------------------------------------------------------------------------------------------------------------------
"""

PUBLIC_URL_BASE = "https://api.kraken.com/0/public/"

def get_asset_info(assets=[]):
    url = PUBLIC_URL_BASE + "Assets"
    if assets:
        url = url + '?asset={}'.format(','.join(assets))
        
    return get_result(url)

def get_all_asset_altnames():
    assets = get_asset_info()
    return [assets[asset]['altname'] for asset in assets]

def get_currency_pairs(pairs=[]):
    url = PUBLIC_URL_BASE + "AssetPairs"
    if pairs:
        url = url + '?pair={}'.format(','.join(pairs))
        
    return get_result(url)
    
def get_ticker_info(pairs=None):
    url = PUBLIC_URL_BASE + "Ticker"
    if pairs:
        url = url + '?pair={}'.format(','.join(pairs))
    print(url)
    return get_result(url)
    
def get_all_wsnames():
    result = get_currency_pairs()

    if not result:
        return []
    
    return [coin['wsname'] for coin in result.values() if 'wsname' in coin]

#print(get_asset_info(['XBT']))
#print(get_ticker_info(['XBTUSD', 'ETHUSD', 'LTCUSD']))
#print(get_all_asset_altnames())
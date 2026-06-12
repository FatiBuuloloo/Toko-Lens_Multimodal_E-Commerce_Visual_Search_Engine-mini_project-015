import pandas as pd
import requests
import time

# Can be modified as needed, e.g., to increase or decrease the number of pages per letter
df = pd.read_csv('list_shop.csv')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
    'content-type': 'application/json',
    'accept': '*/*',
    'origin': 'https://www.tokopedia.com',
    'referer': 'https://www.tokopedia.com/',
    'x-version': 'cfd5462',
    'x-source': 'tokopedia-lite',
    'x-device': 'default_v3',
    'x-tkpd-lite-service': 'zeus',
}

def get_shop_sid(shop_idx, shop_url: str, retries: int = 5) -> str | None:
    domain = shop_url.rstrip('/').split('/')[-1]
    url = 'https://gql.tokopedia.com/graphql/ShopInfoByID'
    payload = [{
        "operationName": "ShopInfoByID",
        "variables": {"domain": domain, "id": 0},
        "query": """
        query ShopInfoByID($id: Int!, $domain: String) {
          shopInfoByID(input: {shopIDs: [$id], fields: ["core"], domain: $domain}) {
            result {
              shopCore { shopID name domain }
            }
            error { message }
          }
        }"""
    }]
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()[0]['data']['shopInfoByID']['result']
                if result:
                    shop_core = result[0]['shopCore']
                    sid = str(shop_core['shopID'])
                    return sid
                else:
                    print(f"[{shop_idx}] No Result: {domain}")
                    return None
            elif resp.status_code == 502:
                wait = attempt * 2
                print(f"[{shop_idx}] Status {resp.status_code}: {domain} - {shop_url}  Retrying")
                print()
                time.sleep(wait)
            elif resp.status_code == 429:
                print(f"[{shop_idx}] 429 Rate limit {domain}  - {shop_url}")
                print()
                time.sleep(10)
            else:
                print(f"[{shop_idx}] Status {resp.status_code} untuk: {domain}")
                print()
                return None
        except requests.exceptions.Timeout:
            wait = attempt * 2
            print(f"[{shop_idx}] Timeout: {domain} - {shop_url}")
            time.sleep(wait)
        except Exception as e:
            print(f"[{shop_idx}] Error {domain} - {shop_url}: {e}")
            return None
    return None

def get_all_sids(df: pd.DataFrame,
                 shop_name: str = 'Nama Toko',
                 shop_url:  str = 'URL') -> pd.DataFrame:
    hasil = []
    total = len(df)
    for i, row in df.iterrows():
        nama = row[shop_name]
        url  = row[shop_url]
        sid = get_shop_sid(i, url)
        if not sid:
            print(f"[{i}] SID not Found on {nama}")
            print()
        hasil.append({
            shop_name:  nama,
            shop_url:   url,
            'SID':     sid,
        })
        time.sleep(0.3)
    df_hasil = pd.DataFrame(hasil)
    berhasil = df_hasil['SID'].notna().sum()
    gagal    = df_hasil['SID'].isna().sum()
    print(f"Success : {berhasil}")
    print(f"Fail    : {gagal}")
    return df_hasil

df_sid = get_all_sids(
    df,
    shop_name = 'Nama Toko',
    shop_url  = 'URL'
)
df_sid.to_csv('shop_with_ID.csv', index=False)
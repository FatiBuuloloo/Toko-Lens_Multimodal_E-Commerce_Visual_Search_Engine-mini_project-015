import pandas as pd
import requests
import time
import sqlite3
import os

df = pd.read_csv('shop_with_ID.csv')

def scrape_all_product(df_shops, db_name="tokopedia_data.db"):
    folder_name = 'Picture_Product'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    url = 'https://gql.tokopedia.com/graphql/ShopProducts'
    headers = {
        'sec-ch-ua-platform': '"Windows"',
        'x-version': 'cfd5462', 
        'Referer': 'https://www.tokopedia.com/',
        'sec-ch-ua': '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
        'x-price-center': 'true',
        'sec-ch-ua-mobile': '?0',
        'bd-device-id': '7628337271406577160',
        'x-source': 'tokopedia-lite',
        'x-device': 'default_v3',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
        'accept': '*/*',
        'content-type': 'application/json',
        'x-tkpd-lite-service': 'zeus'
    }
    all_extracted_products = []
    error_list = []
    image_error_list = []
    global_id = 1
    
    try:
        for index, row in df_shops.iterrows():
            nama_toko = row['Nama Toko']
            
            try:
                sid = str(int(float(row['SID'])))
            except Exception:
                sid = str(row['SID']).strip()
                
            page = 1
            print(f"\n[PROCESS] Attempting to fetch data from Shop: {nama_toko} (SID: {sid})")
            
            while True:
                payload = [{
                    "operationName": "ShopProducts",
                    "variables": {
                        "source": "shop", "sid": sid, "page": page, "perPage": 80,
                        "etalaseId": "etalase", "sort": 1, "user_districtId": "2274",
                        "user_cityId": "176", "user_lat": "0", "user_long": "0",
                        "usecase": "ace_get_shop_product_v2"
                    },
                    "query": "query ShopProducts($sid: String!, $source: String, $page: Int, $perPage: Int, $keyword: String, $etalaseId: String, $sort: Int, $user_districtId: String, $user_cityId: String, $user_lat: String, $user_long: String, $usecase: String) {\n  GetShopProduct(shopID: $sid, source: $source, filter: {page: $page, perPage: $perPage, fkeyword: $keyword, fmenu: $etalaseId, sort: $sort, user_districtId: $user_districtId, user_cityId: $user_cityId, user_lat: $user_lat, user_long: $user_long, usecase: $usecase}) {\n    status\n    errors\n    links {\n      prev\n      next\n      __typename\n    }\n    data {\n      name\n      product_url\n      product_id\n      price {\n        text_idr\n        __typename\n      }\n      primary_image {\n        original\n        thumbnail\n        resize300\n        __typename\n      }\n      flags {\n        isSold\n        isPreorder\n        isWholesale\n        isWishlist\n        __typename\n      }\n      campaign {\n        discounted_percentage\n        original_price_fmt\n        start_date\n        end_date\n        __typename\n      }\n      label {\n        color_hex\n        content\n        __typename\n      }\n      label_groups {\n        position\n        title\n        type\n        url\n        styles {\n          key\n          value\n          __typename\n        }\n        __typename\n      }\n      badge {\n        title\n        image_url\n        __typename\n      }\n      stats {\n        reviewCount\n        rating\n        averageRating\n        __typename\n      }\n      category {\n        id\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n}\n"
                }]
                
                try:
                    print(f"  -> Sending Request to Page {page}...")
                    response = requests.post(url, headers=headers, json=payload, timeout=15)
                    
                    if response.status_code != 200:
                        err_msg = f"HTTP {response.status_code} (Possibly blocked/Forbidden)"
                        print(f"  [ERROR] {err_msg}")
                        error_list.append({'Index_DF': index, 'Nama Toko': nama_toko, 'SID': sid, 'Pesan_Error': err_msg})
                        break
                        
                    raw = response.json()
                    
                    if not raw or 'data' not in raw[0] or 'GetShopProduct' not in raw[0]['data'] or 'data' not in raw[0]['data']['GetShopProduct']:
                        err_msg = "Invalid JSON structure. Tokopedia might have detected a bot or requested a captcha."
                        print(f"  [ERROR] {err_msg}")
                        print(f"  [DEBUG RES] {str(raw)[:200]}")
                        error_list.append({'Index_DF': index, 'Nama Toko': nama_toko, 'SID': sid, 'Pesan_Error': err_msg})
                        break
                        
                    products = raw[0]['data']['GetShopProduct']['data']
                    if not products or len(products) == 0:
                        print("  [INFO] Finished / No more products in this shop.")
                        break
                        
                    print(f"  [SUCCESS] Successfully retrieved {len(products)} products.")
                    
                    for p in products:
                        rating = p.get('stats', {}).get('averageRating', "0")
                        all_extracted_products.append({
                            'Id': global_id,
                            'Shop_Name': nama_toko,
                            'Product_Name': p.get('name', ''),
                            'Price': p.get('price', {}).get('text_idr', ''),
                            'Rating': rating,
                            'URL_Product_Picture': p.get('primary_image', {}).get('original', ''),
                            'URL_Product': p.get('product_url', '')
                        })
                        
                        url_gambar = p.get('primary_image', {}).get('original', '')
                        if url_gambar:
                            try:
                                img_response = requests.get(url_gambar, headers={'User-Agent': headers['User-Agent']}, timeout=10)
                                if img_response.status_code == 200:
                                    save_path = os.path.join(folder_name, f"{global_id}.jpg")
                                    with open(save_path, 'wb') as f:
                                        f.write(img_response.content)
                                else:
                                    image_error_list.append({'Index_Product': global_id, 'Nama Toko': nama_toko, 'Product_Name': p.get('name', '')})
                            except Exception:
                                image_error_list.append({'Index_Product': global_id, 'Nama Toko': nama_toko, 'Product_Name': p.get('name', '')})
                        
                        global_id += 1
                        
                    page += 1
                    time.sleep(2)
                    
                except Exception as e:
                    err_msg = f"Request Exception: {str(e)}"
                    print(f"  [ERROR] {err_msg}")
                    error_list.append({'Index_DF': index, 'Nama Toko': nama_toko, 'SID': sid, 'Pesan_Error': err_msg})
                    break
                    
    except KeyboardInterrupt:
        print("\n[INFO] Process forcefully terminated by user.")
        
    if all_extracted_products:
        df_final = pd.DataFrame(all_extracted_products)
        conn = sqlite3.connect(db_name)
        df_final.to_sql('tokopedia_product', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n[SUCCESS] Successfully saved {len(df_final)} products to {db_name}.")
        print(f"Total system errors: {len(error_list)}")
        return df_final, error_list, image_error_list
    else:
        print("\n==========================================================")
        print("[FAILED] No product data was successfully retrieved.")
        print("Technical error details:")
        print("==========================================================")
        for err in error_list:
            print(f"- Shop: {err['Nama Toko']} (SID: {err['SID']}) | Error: {err['Pesan_Error']}")
        return None, error_list, image_error_list

df_products, list_error, list_image_error = scrape_all_product(df, db_name="database_products.db")
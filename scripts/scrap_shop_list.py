import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

def get_chrome_options():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    return options

def initiate_driver():
    try:
        options = get_chrome_options()
        driver = webdriver.Chrome(options=options)
        print("Driver Works")
        return driver
    except Exception as e:
        print(f"Fail: {e}")
        return None

def get_driver():
    options = get_chrome_options()
    return webdriver.Chrome(options=options)

# Adjust as needed to add or remove pages per letter
pages_dict = {
    'a': 125, 'b': 110, 'c': 42, 'd': 29, 'e': 22,
    'f': 47, 'g': 85, 'h': 40, 'i': 77, 'j': 26,
    'k': 35, 'l': 29, 'm': 64, 'n': 21, 'o': 59,
    'p': 45, 'q': 3, 'r': 23, 's': 138, 't': 39,
    'u': 9, 'v': 12, 'w': 15, 'x': 3, 'y': 7,
    'z': 7}

js_script = """
let dataToko = [];
let seenUrls = new Set();
let allLinks = document.querySelectorAll('a');
allLinks.forEach(el => {
    let text = el.innerText;
    let href = el.href;
    if (text.includes('Lihat Toko')) {
        let namaBersih = text.split('\\n')[0].trim();
        if (namaBersih && !seenUrls.has(href)) {
            dataToko.push({"Nama Toko": namaBersih, "URL": href});
            seenUrls.add(href);
        }
    }
});
return dataToko;
"""

list_shop = []

# Test driver 1st to ensure it works before running the full scraping process
driver_test = initiate_driver()
if driver_test:
    driver_test.quit()

for query, max_page in pages_dict.items():
    print(f"Start query '{query}'...")
    driver = get_driver()
    empty_pages = 0
    toko_per_huruf = []
    try:
        for page in range(1, max_page + 1):
            url = f"https://www.tokopedia.com/search?page={page}&q={query}&shop_tier=2&st=shop"
            try:
                driver.get(url)
                time.sleep(5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                res = driver.execute_script(js_script)
                if res:
                    toko_per_huruf.extend(res)
                else:
                    empty_pages += 1
            except Exception as e:
                empty_pages += 1
                continue
            time.sleep(3)
    finally:
        driver.quit()
        list_shop.extend(toko_per_huruf)
        if empty_pages > 0:
            print(f"Query '{query}' is finished. Note: {empty_pages} pages were empty or failed.")
        else:
            print(f"Query '{query}' is finished successfully.")
        time.sleep(10)

if list_shop:
    df = pd.DataFrame(list_shop).drop_duplicates(subset='URL')
    df.to_csv('list_shop.csv', index=False)
    print(f"Selesai! Berhasil menyimpan {len(df)} data toko ke 'list_shop.csv'")
else:
    print("Proses selesai, namun tidak ada data toko yang ditemukan.")
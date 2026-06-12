import io
import re
import sqlite3
import requests
import numpy as np
import pandas as pd
import torch
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="Toko Lens API Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home_health_check():
    return {
        "status"       : "healthy",
        "message"      : "Backend Toko Lens Retrieval berhasil dibangunkan dan berjalan lancar!",
        "documentation": "/docs"
    }

# LOAD RESOURCE─
REPO_ID    = "Viewww/tokopedia-search-indices"
MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

print("Memuat model AI dan basis data dari Hugging Face Hub...")

# Hanya satu FAISS index (unified CLIP space)
img_idx_path = hf_hub_download(
    repo_id   = REPO_ID,
    filename  = "tokopedia_img_hnsw.faiss",
    repo_type = "dataset"
)
db_path = hf_hub_download(
    repo_id   = REPO_ID,
    filename  = "metadata/metadata_tokopedia.db",
    repo_type = "dataset"
)

index_img = faiss.read_index(img_idx_path, faiss.IO_FLAG_MMAP)

device     = "cuda" if torch.cuda.is_available() else "cpu"
processor  = CLIPProcessor.from_pretrained(MODEL_NAME)
model_clip = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model_clip = model_clip.to(device)
model_clip.eval()

print("Semua resource berhasil dimuat!")


def fetch_tokopedia_image(sid, shop_name, product_name):
    try:
        sid_str = str(sid)
        shop    = str(shop_name).lower()
        key     = str(product_name).lower().replace(" ", "-")

        HEADERS = {
            'sec-ch-ua-platform' : '"Windows"',
            'Referer'            : f'https://www.tokopedia.com/{shop}/{key}',
            'sec-ch-ua'          : '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            'x-price-center'     : 'true',
            'sec-ch-ua-mobile'   : '?0',
            'bd-device-id'       : '7628337271406577160',
            'x-source'           : 'tokopedia-lite',
            'x-device'           : 'default_v3',
            'User-Agent'         : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36',
            'accept'             : '*/*',
            'content-type'       : 'application/json',
            'x-tkpd-lite-service': 'zeus',
        }

        payload = [{
            "operationName": "ShopProducts",
            "variables": {
                "source": "shop", "sid": sid_str, "page": 1, "perPage": 5,
                "keyword": key, "etalaseId": "etalase", "sort": 1,
                "user_districtId": "2274", "user_cityId": "176",
                "user_lat": "0", "user_long": "0",
                "usecase": "ace_get_shop_product_v2",
            },
            "query": (
                "query ShopProducts($sid: String!, $source: String, $page: Int, $perPage: Int, "
                "$keyword: String, $etalaseId: String, $sort: Int, $user_districtId: String, "
                "$user_cityId: String, $user_lat: String, $user_long: String, $usecase: String) {\n"
                "  GetShopProduct(shopID: $sid, source: $source, filter: {page: $page, perPage: $perPage, "
                "fkeyword: $keyword, fmenu: $etalaseId, sort: $sort, user_districtId: $user_districtId, "
                "user_cityId: $user_cityId, user_lat: $user_lat, user_long: $user_long, usecase: $usecase}) {\n"
                "    data {\n      product_url\n      primary_image { original }\n    }\n  }\n}\n"
            ),
        }]

        res = requests.post(
            'https://gql.tokopedia.com/graphql/ShopProducts',
            headers=HEADERS, json=payload, timeout=4,
        )
        if res.status_code == 200:
            data = res.json()[0]['data']['GetShopProduct']['data']
            if data:
                return data[0]['primary_image']['original']
    except Exception:
        pass
    return None


def embed_image(pil_img: Image.Image) -> np.ndarray:
    inputs       = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        vision_outputs = model_clip.vision_model(pixel_values=pixel_values)
        img_emb        = model_clip.visual_projection(vision_outputs.pooler_output)
        img_emb        = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.float().cpu().numpy().astype('float32')


def embed_text(text: str) -> np.ndarray:
    text_inputs = processor(
        text       = [text],
        padding    = True,
        truncation = True,
        max_length = 77,
        return_tensors = "pt"
    ).to(device)
    with torch.no_grad():
        text_outputs = model_clip.text_model(**text_inputs)
        txt_emb      = model_clip.text_projection(text_outputs.pooler_output)
        txt_emb      = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb.float().cpu().numpy().astype('float32')

@app.post("/search")
async def search_products(
    file       : UploadFile = File(...),
    search_text: str = Form(""),
    price_min  : int = Form(0),
    price_max  : int = Form(100_000_000),
):
    try:
        img_bytes = await file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        img_emb = embed_image(pil_img)

        _, I_img      = index_img.search(img_emb, 2000)
        candidate_ids = [int(cid) for cid in I_img[0].tolist() if cid != -1]

        if not candidate_ids:
            return {"results": []}

        if search_text.strip():
            txt_emb = embed_text(search_text)

            _, I_txt   = index_img.search(txt_emb, 5000)
            txt_id_set = set(int(i) for i in I_txt[0].tolist() if i != -1)

            candidate_ids = [cid for cid in candidate_ids if cid in txt_id_set]

        if not candidate_ids:
            return {"results": []}

        conn         = sqlite3.connect(db_path)
        placeholders = ",".join("?" for _ in candidate_ids)
        query        = f"""
            SELECT ID_Product, Product_Name, Price, Rating, URL_Product, SID, Shop_Name
            FROM products
            WHERE ID_Product IN ({placeholders})
        """
        df_meta = pd.read_sql_query(query, conn, params=candidate_ids)
        conn.close()

        if df_meta.empty:
            return {"results": []}

        def parse_price(p):
            try:
                return int("".join(re.findall(r"\d+", str(p))))
            except Exception:
                return 0

        df_meta['Price_int'] = df_meta['Price'].apply(parse_price)
        df_meta = df_meta[
            (df_meta['Price_int'] >= price_min) &
            (df_meta['Price_int'] <= price_max)
        ]

        if df_meta.empty:
            return {"results": []}

        rank_map        = {pid: idx for idx, pid in enumerate(candidate_ids)}
        df_meta['rank'] = df_meta['ID_Product'].map(rank_map)
        df_meta         = df_meta.sort_values('rank')

        results = []
        for _, row in df_meta.head(100).iterrows():
            raw_rating = row['Rating']
            try:
                rating_val = float(raw_rating) if pd.notna(raw_rating) else 0.0
            except (ValueError, TypeError):
                rating_val = 0.0

            results.append({
                "id"    : int(row['ID_Product']),
                "name"  : str(row['Product_Name']),
                "price" : int(row['Price_int']),
                "rating": rating_val,
                "url"   : str(row['URL_Product']),
                "sid"   : str(row['SID']),
                "shop"  : str(row['Shop_Name']),
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-image")
def get_image(sid: str, shop: str, name: str):
    """Mengembalikan URL gambar asli Tokopedia secara real-time."""
    url = fetch_tokopedia_image(sid, shop, name)
    if url:
        return {"image_url": url}
    return {"image_url": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
# Toko Lens: Multimodal E-Commerce Visual Search Engine

> Search Tokopedia by image. Upload a photo, crop the product, add an optional keyword and price range, and get visually ranked results from a catalogue of over 3.4 million real product listings.

---

## What This Is

Toko Lens is an end-to-end multimodal search system built on top of a large-scale dataset of Tokopedia Mall products. Instead of typing a product name, you upload a photo. The system encodes it with **CLIP (laion/CLIP-ViT-H-14-laion2B-s32B-b79K)**, searches a prebuilt FAISS index of 3.4 million CLIP image embeddings, optionally filters the visual results with a SQLite keyword search on product names, applies price filtering, and returns ranked results with live product images.

Text keyword input does not trigger a separate vector search. It filters the visual candidate set using SQLite LIKE matching against `Product_Name`. This means text can only narrow down what the image already found, never introduce irrelevant products from outside the visual results. This is the same principle used by Google Lens.

---

## Architecture Overview

```
User uploads image
       |
  [Optional crop]
       |
  [Optional text keyword + price range]
       |
  FastAPI backend
       |
       v
  CLIP vision encoder
       | 1024-dim L2-normalized vector
       v
  HNSW FAISS search across 3.4M products
       | top-5000 visual candidates
       v
  (if text keyword provided)
  SQLite LIKE filter on Product_Name
  within the top-5000 visual candidates only
       | filtered candidates (visual rank preserved)
       v
  (fallback: if no name match, keep all 5000)
       |
       v
  SQLite metadata lookup + price filter
       |
       v
  Sort by visual similarity rank
       |
       v
  Return top-100 product cards
       |
  Live image URLs fetched from Tokopedia GQL
```

---

## Repository Structure

```text
toko-lens-retrieval/
|
+-- README.md                # Main project documentation
+-- requirements.txt         # Core Python dependencies for local scripts
|
+-- notebooks/                                        # Development and experiment notebooks
|   +-- Scrapping_Tokopedia.ipynb                     # Initial playground for scraping logic and API testing
|   +-- embedding-images-phase1-clip.ipynb            # Batch 1 of CLIP image embedding extraction
|   +-- embedding-images-phase2-clip.ipynb            # Batch 2 of CLIP image embedding extraction
|   +-- embedding-images-phase3-clip.ipynb            # Batch 3 of CLIP image embedding extraction
|   +-- embedding-images-phase4-clip.ipynb            # Batch 4 of CLIP image embedding extraction
|   +-- embedding-images-phase5-clip.ipynb            # Batch 5 of CLIP image embedding extraction
|   +-- embedding-images-phase6-clip.ipynb            # Batch 6 of CLIP image embedding extraction
|   +-- embedding-images-phase7-clip.ipynb            # Batch 7 of CLIP image embedding extraction
|   +-- embedding-images-phase8-clip.ipynb            # Batch 8 of CLIP image embedding extraction
|   +-- embedding-images-phase9-clip.ipynb            # Batch 9 of CLIP image embedding extraction
|   +-- embedding-images-phase10-clip.ipynb           # Batch 10 of CLIP image embedding extraction
|   +-- embedding-images-phase11-clip.ipynb           # Batch 11 of CLIP image embedding extraction
|   +-- embedding-images-phase12-clip.ipynb           # Batch 12 of CLIP image embedding extraction
|   +-- embedding-images-phase13-clip.ipynb           # Batch 13 of CLIP image embedding extraction
|   +-- embedding-images-phase14-clip.ipynb           # Batch 14 of CLIP image embedding extraction
|   +-- creating-faiss-clip-versi-indexhnswflat.ipynb # Building the unified HNSW FAISS index
|   +-- push_images_zip_to_kaggle.ipynb               # Automation to package and upload image archives to Kaggle
|   +-- push-faiss-clip-to-hf.ipynb                   # Utility to push the FAISS index to Hugging Face Hub
|
+-- scripts/                 # Core Pipeline: Independent ETL scripts executed sequentially
|   +-- scrap_shop_list.py   # Scrapes shop list via Selenium (Outputs: list_shop.csv)
|   +-- scrap_shop_id.py     # Resolves shop domains to SIDs via GraphQL (Outputs: shop_with_ID.csv)
|   +-- scrap_product.py     # Extracts product metadata to SQLite and downloads images
|   +-- embedding_images.py  # Computes CLIP image embeddings and builds the FAISS index
|
+-- backend/                 # API Service (Hosted on Hugging Face Spaces)
|   |                        # Live Repository: https://huggingface.co/spaces/sole1l/tokolens-backend/tree/main
|   +-- app.py               # FastAPI application and CLIP-based vector search logic
|   +-- Dockerfile           # Containerization setup for the backend environment
|   +-- requirements.txt     # Specific dependencies for the backend server
|
+-- frontend/                # User Interface (Hosted on Hugging Face Spaces)
    |                        # Live Repository: https://huggingface.co/spaces/Viewww/TokoLens/tree/main
    +-- index.html           # Main user interface for multimodal search
    +-- style.css            # Visual styling and responsive layouts
    +-- assets/              # Local JavaScript modules for image cropping and API integration
```

---

## Data Pipeline

### Step 1: Scraping Shops (13,165 Shops)

All shops carrying the **Mall** label on Tokopedia were scraped to collect shop names and their corresponding **Shop IDs (SID)**. The Mall label filters for verified, high-quality merchants and keeps the product catalogue representative. The result was **13,165 unique shop entries**.

### Step 2: Scraping Products (3,614,752 Products, 7 Phases)

Using the SIDs, the product catalogue of every shop was queried through Tokopedia's GraphQL API to collect `Product_Name`, `Price`, `Rating`, and `URL_Product`. Due to Colab runtime limits and occasional session crashes, this stage was divided into **7 sequential phases**, each resuming from where the previous one stopped. Total products collected: **3,614,752 listings**.

Dataset published on Kaggle:
[tokopedia-products-with-images-dataset-v1](https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1)

### Step 3: Scraping Product Images (3,475,088 Images, 15 Phases)

The primary image for each product was downloaded by visiting its product URL. Image downloads are time-intensive and Colab sessions expire before completing the full catalogue, so this stage was run as **15 sequential phases**. Some products (primarily prescription medical items) have no scrapable image due to platform restrictions. After all 15 phases, **3,475,088 product images** were collected. Products without images were removed from the catalogue, reducing the final product count to match.

Dataset published on Kaggle:
[tokopedia-images-product](https://www.kaggle.com/datasets/fati22/tokopedia-images-product)

---

## Embedding Pipeline

### Model: CLIP ViT-H-14 (1024-Dimensional Image Space)

All embeddings are produced by **[laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)**, an open CLIP model trained by LAION on 2 billion curated image-text pairs.

| Property | Value |
|---|---|
| Vision backbone | ViT-H/14 (1280-dim internal, 32 encoder layers) |
| Output dimension | 1024-dim (after visual projection layer) |
| Training data | LAION-2B (2 billion image-text pairs) |
| Similarity metric | Cosine similarity via L2-normalized inner product |

The image encoder projects its output to **1024-dimensional** space through the `visual_projection` layer. After L2 normalization, all product image vectors can be compared directly using dot product inside a single FAISS index.

```python
with torch.no_grad():
    vision_outputs = model_clip.vision_model(pixel_values=pixel_values)
    img_emb        = model_clip.visual_projection(vision_outputs.pooler_output)
    img_emb        = img_emb / img_emb.norm(dim=-1, keepdim=True)
```

### Phased Execution (14 Parquet Files)

Embedding 3.4 million product images with a ViT-H/14 backbone was split into **14 sequential phases** on Kaggle GPU notebooks (Tesla T4) with batch size 32. Each phase produces one Parquet file storing `ID_Product`, `Judul`, and `Image_Embedding`. Only image embeddings are stored since text keyword filtering uses SQLite directly at query time.

---

## FAISS Index Construction

### Single Index: IndexHNSWFlat with Inner Product

```python
d      = 1024
M      = 32
ef     = 200

core_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
core_index.hnsw.efConstruction = ef
index  = faiss.IndexIDMap2(core_index)
```

HNSW provides very low query latency with high recall and requires no training step. All 14 Parquet files were added sequentially. `IndexIDMap2` stores `ID_Product` directly inside the index so product IDs are returned immediately in search results. After construction, the FAISS index and SQLite metadata database were pushed to Hugging Face:

```
Viewww/tokopedia-search-indices
  tokopedia_img_CLIP_hnsw.faiss
  metadata/metadata_tokopedia.db
```

---

## Backend: FastAPI

The backend is a Python **FastAPI** application. At startup it downloads the FAISS index and the SQLite database from Hugging Face Hub, then loads the CLIP model in **float16** precision to stay within the Space's RAM budget.

**`POST /search`** accepts a product image, optional text keyword, minimum price, and maximum price. The pipeline:

1. Decodes the uploaded image with Pillow.
2. Encodes it with the CLIP vision encoder to get a 1024-dim L2-normalized vector.
3. Searches the HNSW FAISS index for the top 5,000 nearest neighbours.
4. If a text keyword is provided, runs a SQLite LIKE query on `Product_Name` for each word in the keyword, filtering only within the top-5000 visual candidates. The original visual ranking order is preserved. If no product name matches the keyword, the system falls back to the full image-only results.
5. Queries the SQLite database for product metadata of all surviving candidates.
6. Filters by price range.
7. Re-ranks by the original FAISS visual similarity order.
8. Returns the top 100 results as JSON.

```python
if search_text.strip():
    keywords   = search_text.strip().split()
    conditions = " AND ".join(
        [f"LOWER(Product_Name) LIKE LOWER('%{kw}%')" for kw in keywords]
    )
    placeholders  = ",".join("?" for _ in candidate_ids)
    df_txt_filter = pd.read_sql_query(
        f"""
        SELECT ID_Product FROM products
        WHERE ID_Product IN ({placeholders})
        AND ({conditions})
        """,
        conn_tmp, params=candidate_ids
    )
    if not df_txt_filter.empty:
        txt_id_set    = set(df_txt_filter['ID_Product'].tolist())
        candidate_ids = [cid for cid in candidate_ids if cid in txt_id_set]
```

**`GET /get-image`** fetches the live primary image URL for a given product from Tokopedia's GraphQL API in real time.

---

## Frontend: HTML / CSS / JavaScript

The frontend is a plain single-page application built with **HTML, CSS, and JavaScript** (no framework dependencies). The user flow is:

1. Upload a product image from device storage or capture one with the device camera.
2. Use the in-browser crop tool (Cropper.js) to isolate the specific object of interest before searching.
3. Optionally enter one or more keywords to filter results by product name within the visual candidates.
4. Set a price range to filter by budget.
5. Submit. Results are rendered as product cards with name, price, rating, and live images loaded via the `/get-image` endpoint.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Scraping | Python `requests`, Tokopedia GraphQL API |
| Data Storage | SQLite, Parquet, Kaggle Datasets |
| Image Embedding | CLIP ViT-H/14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) via Hugging Face Transformers |
| Text Keyword Filtering | SQLite LIKE query on `Product_Name` |
| Vector Search | FAISS (IndexHNSWFlat, IndexIDMap2, METRIC_INNER_PRODUCT) |
| Backend | FastAPI, Uvicorn, Python |
| Frontend | HTML, CSS, JavaScript, Cropper.js |
| Deployment | Hugging Face Spaces (backend + frontend separately) |
| Model / Index Hosting | Hugging Face Hub |
| Embedding Environment | Kaggle Notebooks (GPU: Tesla T4) |
| Scraping Environment | Google Colab |

---

## Links

| Resource | URL |
|---|---|
| Live Demo | https://viewww-tokolens.static.hf.space |
| Frontend Space | https://huggingface.co/spaces/Viewww/TokoLens/tree/main |
| Backend Space | https://huggingface.co/spaces/sole1l/tokolens-backend/tree/main |
| Products Dataset | https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1 |
| Images Dataset | https://www.kaggle.com/datasets/fati22/tokopedia-images-product |

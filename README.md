# Toko Lens: Multimodal E-Commerce Visual Search Engine

> Search Tokopedia by image. Upload a photo, crop the product, add an optional keyword and price range, and get visually ranked results from a catalogue of over 3.4 million real product listings.

---

## What This Is

Toko Lens is an end-to-end multimodal search system built on top of a large-scale dataset of Tokopedia Mall products. Instead of typing a product name, you upload a photo. The system encodes it with a Vision Transformer, searches a prebuilt FAISS index of 3.4 million product image embeddings, optionally intersects with a BGE-M3 text search, applies price filtering, and returns ranked results with live product images.

---

## Architecture Overview

```
User uploads image
       |
  [Optional crop]
       |
  [Optional text query + price range]
       |
  FastAPI backend
  |-- ViT encodes image  -----------> HNSW index search --> top-5000 visual matches
  |-- BGE-M3 encodes text ----------> IVF-PQ index search --> top-5000 text matches
                                               |
                                    Intersection of both candidate sets
                                               |
                                    SQLite metadata lookup + price filter
                                               |
                                    Re-rank by visual similarity order
                                               |
                                    Return top-100 product cards
                                               |
                                  Live image URLs fetched from Tokopedia GQL
```
## Repository Structure

This repository is organized as a monorepo to separate research experiments, production data pipelines, and deployment code. You can directly access and view the live hosted repositories for both the frontend and backend via the links embedded below.

```text
toko-lens-retrieval/
│
├── README.md                # Main project documentation
├── requirements.txt         # Core Python dependencies for local scripts
│
├── notebooks/                                       # Development and experiment notebooks
│   ├── Scrapping_Tokopedia.ipynb                    # Initial playground for scraping logic and API testing
│   ├── embedding-images-phase1-clip.ipynb           # Batch 1 of image embedding extraction using CLIP
│   ├── embedding-images-phase2-clip.ipynb           # Batch 2 of image embedding extraction using CLIP
│   ├── embedding-images-phase3-clip.ipynb           # Batch 3 of image embedding extraction using CLIP
│   ├── embedding-images-phase4-clip.ipynb           # Batch 4 of image embedding extraction using CLIP
│   ├── embedding-images-phase5-clip.ipynb           # Batch 5 of image embedding extraction using CLIP
│   ├── embedding-images-phase6-clip.ipynb           # Batch 6 of image embedding extraction using CLIP
│   ├── embedding-images-phase7-clip.ipynb           # Batch 7 of image embedding extraction using CLIP
│   ├── embedding-images-phase8-clip.ipynb           # Batch 8 of image embedding extraction using CLIP
│   ├── embedding-images-phase9-clip.ipynb           # Batch 9 of image embedding extraction using CLIP
│   ├── embedding-images-phase10-clip.ipynb          # Batch 10 of image embedding extraction using CLIP
│   ├── embedding-images-phase11-clip.ipynb          # Batch 11 of image embedding extraction using CLIP
│   ├── embedding-images-phase12-clip.ipynb          # Batch 12 of image embedding extraction using CLIP
│   ├── embedding-images-phase13-clip.ipynb          # Batch 13 of image embedding extraction using CLIP
│   ├── embedding-images-phase14-clip.ipynb          # Batch 14 of image embedding extraction using CLIP
│   ├── creating-faiss-clip-versi-indexhnswflat.ipynb# Building and evaluating the IndexHNSWFlat vector index
│   ├── push_images_zip_to_kaggle.ipynb              # Automation script to package and upload image archives to Kaggle
│   └── push-faiss-clip-to-hf.ipynb                  # Utility to push completed FAISS indices to Hugging Face Hub
│
├── scripts/                 # Core Pipeline: Independent ETL scripts executed sequentially
│   ├── scrap_shop_list.py# Scrapes shop list via Selenium (Outputs: list_shop.csv)
│   ├── scrap_shop_id.py  # Resolves shop domains to SIDs via GraphQL (Outputs: shop_with_ID.csv)
│   ├── scrap_product.py  # Extracts product metadata to SQLite and downloads images
│   └── embedding_images.py# Computes ViT and BGE-M3 embeddings and builds FAISS indices
│
├── backend/                 # API Service (Hosted on Hugging Face Spaces)
│   │                        # Live Repository: [HuggingFace](https://huggingface.co/spaces/Viewww/image-retrieval-backend/tree/main)
│   ├── app.py               # FastAPI application and vector search logic
│   ├── Dockerfile           # Containerization setup for the backend environment
│   └── requirements.txt     # Specific dependencies for the backend server
│
└── frontend/                # User Interface (Hosted on Hugging Face Spaces)
    │                        # Live Repository: [HuggingFace](https://huggingface.co/spaces/Viewww/TokoLens/tree/main)
    ├── index.html           # Main user interface for multimodal search
    ├── style.css            # Visual styling and responsive layouts
    └── assets/              # Local JavaScript modules for image cropping and API integration

---

## Data Pipeline

### Step 1: Scraping Shops (13,165 Shops)

All shops carrying the **Mall** label on Tokopedia were scraped to collect shop names and their corresponding **Shop IDs (SID)**. The Mall label filters for verified, high-quality merchants and keeps the product catalogue representative. The result was **13,165 unique shop entries**.

### Step 2: Scraping Products (3,614,752 Products, 7 Phases)

Using the SIDs, the product catalogue of every shop was queried through Tokopedia's GraphQL API to collect `Product_Name`, `Price`, `Rating`, and `URL_Product`. Due to Colab runtime limits and occasional session crashes, this stage was divided into **7 sequential phases**, each resuming from where the previous one stopped. Total products collected: **3,614,752 listings**.

Dataset published on Kaggle:
[tokopedia-products-with-images-dataset-v1](https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1)

### Step 3: Scraping Product Images (3,475,088 Images, 15 Phases)

The primary image for each product was downloaded by visiting its product URL. Image downloads are time-intensive, and Colab sessions would expire before completing the full catalogue, so this stage was run as **15 sequential phases**. Some products (primarily prescription medical items) have no scrapable image due to platform restrictions. After all 15 phases, **3,475,088 product images** were collected. Products without images were removed from the catalogue, reducing the final product count to match.

Dataset published on Kaggle:
[tokopedia-images-product](https://www.kaggle.com/datasets/fati22/tokopedia-images-product)

---

## Embedding Pipeline

### Text Embeddings: BGE-M3 (1024 dimensions)

Product titles were encoded with **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)**, a multilingual dense retrieval model that handles Indonesian and English natively. Each title was encoded into a 1024-dimensional float32 vector.

### Image Embeddings: ViT Base Patch16-224 (768 dimensions)

Product images were encoded with **[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)**, a Vision Transformer pretrained on ImageNet-21k. Each image was resized to 224x224, processed through the ViT patch tokenizer, and the `[CLS]` token from the final hidden layer was used as the image vector (768 dimensions).

### Phased Execution (8 Parquet Files Total)

Encoding 3.4 million products requires splitting the work into manageable chunks. Both the text and image embedding runs were each divided into **4 phases**, producing **8 Parquet files in total** (4 for text embeddings, 4 for image embeddings). Each Parquet file stores product IDs alongside their corresponding vectors.

---

## FAISS Index Construction

### Image Index: IndexHNSWFlat

```python
M_hnsw = 32
ef     = 200

core_index = faiss.IndexHNSWFlat(dim, M_hnsw)   # dim = 768
core_index.hnsw.efConstruction = ef
final_index = faiss.IndexIDMap2(core_index)
```

HNSW was chosen for the image index because it provides very low query latency with high recall and requires no training step. All 7 Parquet chunks were added sequentially. The final index was saved as `tokopedia_img_hnsw.faiss`.

### Text Index: IndexIVFPQ

```python
nlist = 1000
m     = 64
nbits = 8

quantizer  = faiss.IndexFlatL2(dim)              # dim = 1024
core_index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
final_index = faiss.IndexIDMap2(core_index)
```

IVF-PQ was chosen for the text index because product quantization compresses the 1024-dimensional vectors significantly, reducing RAM usage at inference time while still allowing fast approximate search with a tunable `nprobe` value. The index was trained on the first 100,000 vectors and then populated with all 7 chunks. The final index was saved as `tokopedia_txt_ivfpq.faiss`.

Both indices use `IndexIDMap2` so that `ID_Product` values are stored directly inside the index and returned in search results without an additional lookup table.

After construction, both FAISS files and the SQLite metadata database were pushed to a Hugging Face dataset repository: **Viewww/tokopedia-search-indices**.

---

## Backend: FastAPI

The backend is a Python **FastAPI** application. At startup it downloads the FAISS indices and the SQLite database from Hugging Face Hub, loads the ViT and BGE-M3 models into memory, and exposes two endpoints.

**`POST /search`** accepts a product image, optional search text, minimum price, and maximum price. The pipeline:

1. Decodes the uploaded image with Pillow.
2. Encodes it with ViT to get a 768-d vector.
3. Searches the HNSW image index for the top 2,000 nearest neighbours.
4. If a text query is provided, encodes it with BGE-M3 to get a 1024-d vector, searches the IVF-PQ text index (nprobe=64) for top 5,000 candidates, then intersects with the image candidates.
5. Queries the SQLite database for product metadata of all surviving candidates.
6. Filters by price range.
7. Re-ranks by the original FAISS image similarity order.
8. Returns the top 100 results as JSON.

**`GET /get-image`** fetches the live primary image URL for a given product from Tokopedia's GraphQL API in real time.

---

## Frontend: HTML / CSS / JavaScript

The frontend is a plain single-page application built with **HTML, CSS, and JavaScript** (no framework dependencies). The user flow is:

1. Upload a product image.
2. Crop the region of interest using the in-browser crop tool.
3. Optionally enter a product name or keyword to refine results.
4. Set a price range.
5. Submit. Results are rendered as product cards with name, price, rating, and live images loaded via the `/get-image` endpoint.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Scraping | Python `requests`, Tokopedia GraphQL API |
| Data Storage | SQLite, Parquet, Kaggle Datasets |
| Image Embedding | ViT Base Patch16-224 (Hugging Face Transformers) |
| Text Embedding | BGE-M3 (Sentence Transformers) |
| Vector Search | FAISS (IndexHNSWFlat, IndexIVFPQ) |
| Backend | FastAPI, Uvicorn, Python |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Hugging Face Spaces (backend + frontend separately) |
| Model Hosting | Hugging Face Hub |
| Embedding Environment | Google Colab, Kaggle Notebooks (GPU) |

---

## Links

| Resource | URL |
|---|---|
| Live Demo | https://viewww-tokolens.static.hf.space |
| Frontend Space | https://huggingface.co/spaces/Viewww/TokoLens/tree/main |
| Backend Space | https://huggingface.co/spaces/Viewww/image-retrieval-backend/tree/main |
| Products Dataset | https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1 |
| Images Dataset | https://www.kaggle.com/datasets/fati22/tokopedia-images-product |

# Toko Lens: Multimodal E-Commerce Visual Search Engine

> Search Tokopedia by image. Upload a photo, crop the product, add an optional keyword and price range, and get visually ranked results from a catalogue of over 3.4 million real product listings.

---

## What This Is

Toko Lens is an end-to-end multimodal search system built on top of a large-scale dataset of Tokopedia Mall products. Instead of typing a product name, you upload a photo. The system encodes it with a Vision Transformer, searches a prebuilt FAISS index of 3.4 million product image embeddings, optionally intersects with a BGE-M3 text search, applies price filtering, and returns ranked results with live product images.

---

## How the Search Works

```
User uploads image (or captures via camera)
              │
              ▼
     [Optional: crop with Cropper.js]
              │
              ▼
     [Optional: add text keyword + price range]
              │
              ▼
         FastAPI backend
              │
              ├─ CLIP vision encoder
              │         │ 1024-dim vector
              │         ▼
              │   HNSW FAISS search → top-2000 visual candidates
              │
              └─ (if text provided)
                  CLIP text encoder
                        │ 1024-dim vector (same space)
                        ▼
                  HNSW FAISS search → top-5000 text candidates
                        │
                  Intersect with visual candidates
                        │
              SQLite metadata lookup
                        │
              Price range filter
                        │
              Sort by visual similarity rank
                        │
              Return top-100 product cards
                        │
              Live image URLs fetched from Tokopedia GQL
```

---

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

### Phase 1: Shop Scraping

The pipeline begins by collecting all shops labelled **Mall** on Tokopedia. Mall-labelled stores represent verified, high-quality merchants and provide a consistent product quality baseline. The scraper collected **13,165 unique shop names**.

For each shop name, a second scraping pass retrieved the corresponding **Shop ID (SID)**, the internal Tokopedia identifier required to query product listings from the platform's GraphQL API.

### Phase 2: Product Scraping (7 Phases)

Using the collected SIDs, the pipeline queried the product catalogue of every shop, extracting four fields per product: `Product_Name`, `Price`, `Rating`, and `URL_Product`. Because Colab free-tier runtime sessions have hard time limits and occasionally crash mid-run, the scraping was divided into **7 sequential phases**, each resuming from the last completed shop. The combined result was a raw product table of **3,614,752 listings**.

The full product dataset is publicly available on Kaggle:
**[tokopedia-products-with-images-dataset-v1](https://www.kaggle.com/datasets/fati22/tokopedia-products-with-images-dataset-v1)**

### Phase 3: Image Scraping (15 Phases)

Each product URL was visited to retrieve its primary product image. This was the most time-intensive stage because image downloads are bandwidth-bound and each Colab session could only process a fraction of the catalogue before timing out. The scraping was split into **15 phases**, each resuming from where the previous one stopped.

A subset of products could not be scraped for images. These were mostly prescription-grade medical products whose listing pages display a doctor-prescription notice instead of a product image, making automated image retrieval impossible.

Final image count after all 15 phases: **3,475,088 product images**, matching a cleaned product table of the same size (down from 3,614,752 after removing image-less entries).

The image dataset is also publicly available on Kaggle:
**[tokopedia-images-product](https://www.kaggle.com/datasets/fati22/tokopedia-images-product)**

---

## Embedding Pipeline

### Why CLIP

The previous version of this system used two separate models: ViT for image embeddings and BGE-M3 for text embeddings. Those two models produce vectors in incompatible spaces (768-dim vs 1024-dim), requiring two separate FAISS indices and an intersection operation at query time.

CLIP solves this at the architecture level. Because it was trained end-to-end using contrastive loss on 2 billion image-text pairs, its image encoder and text encoder are trained jointly to produce vectors that occupy the same semantic space. A photo of a laptop and the phrase "laptop gaming" land close together in this shared space even though they come from completely different input modalities. This makes CLIP the natural choice for a product search system where users may query by image, by text, or by both.

### Model: laion/CLIP-ViT-H-14-laion2B-s32B-b79K

The model used for all embeddings is **`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`**, an open CLIP model trained by LAION on 2 billion curated image-text pairs. Key specifications:

| Property | Value |
|---|---|
| Vision backbone | ViT-H/14 (1280-dim internal, 32 encoder layers) |
| Text backbone | Transformer (1024-dim internal, 24 encoder layers) |
| Output dimension | 1024-dim (after visual and text projection layers) |
| Training data | LAION-2B (2 billion image-text pairs) |
| Similarity metric | Cosine similarity (L2-normalized dot product) |

Both the image encoder and the text encoder project their outputs to the same **1024-dimensional** space through learned linear projection layers (`visual_projection` and `text_projection`). After L2 normalization, vectors from both modalities can be compared directly using inner product.

### Embedding Process (14 Phases)

Embedding 3.4 million product images with a ViT-H/14 backbone is computationally heavy. The embedding run was divided into **14 sequential phases** on Kaggle GPU notebooks (Tesla T4), each processing a slice of the dataset and saving one output Parquet file. The batch size was set to 32 to stay within T4 VRAM limits.

Each Parquet file stores:

```
ID_Product    (int64)     Product identifier matching the SQLite metadata
Judul         (str)       Product title, kept for traceability
Image_Embedding (list)    1024-dim float32 CLIP image vector, L2-normalized
```

Only image embeddings are stored in the Parquet files. Text embeddings for the product catalogue are not needed because the CLIP text encoder is applied at query time directly on the user's keyword, and its output searches the same image-based FAISS index.

```python
with torch.no_grad():
    vision_outputs = model.vision_model(pixel_values=imgs)
    image_features = model.visual_projection(vision_outputs.pooler_output)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

---

## FAISS Index Construction

### Single Index: IndexHNSWFlat with Inner Product

Because all vectors are already L2-normalized and cosine similarity equals inner product for unit vectors, the index uses `faiss.METRIC_INNER_PRODUCT` to align with CLIP's native similarity metric.

```python
d      = 1024
M      = 32
ef     = 200

core_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
core_index.hnsw.efConstruction = ef
index  = faiss.IndexIDMap2(core_index)
```

HNSW (Hierarchical Navigable Small World) was chosen because it delivers very low query latency with high recall and requires no training phase, which is important when indexing 3.4 million vectors in a single sequential pass across 14 Parquet files. `IndexIDMap2` wraps the core index so that `ID_Product` values are stored alongside vectors and returned directly in search results.

After construction, the index was saved and pushed to a Hugging Face dataset repository:

```
Viewww/tokopedia-search-indices
  tokopedia_img_CLIP_hnsw.faiss
  metadata/metadata_tokopedia.db
```

---

### Backend: FastAPI

The backend is a **Python FastAPI** application deployed as a Hugging Face Space. On startup it downloads the FAISS index and SQLite metadata database from the Hugging Face Hub, then loads the CLIP model in **float16** precision to fit within the Space's RAM budget.

```python
model_clip = CLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
```

Two helper functions handle embedding at inference time:

```python
def embed_image(pil_img):
    inputs       = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    with torch.no_grad():
        vision_outputs = model_clip.vision_model(pixel_values=pixel_values)
        img_emb        = model_clip.visual_projection(vision_outputs.pooler_output)
        img_emb        = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.float().cpu().numpy().astype('float32')

def embed_text(text):
    text_inputs = processor(
        text=text, padding=True, truncation=True,
        max_length=77, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_outputs = model_clip.text_model(**text_inputs)
        txt_emb      = model_clip.text_projection(text_outputs.pooler_output)
        txt_emb      = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb.float().cpu().numpy().astype('float32')
```

**`POST /search`** accepts a product image, an optional text keyword, a minimum price, and a maximum price. The search pipeline proceeds as follows:

```
1. Decode uploaded image with Pillow
2. Encode image with CLIP vision encoder + visual_projection
   → 1024-dim L2-normalized float32 vector
3. Search the single HNSW FAISS index → top-2000 candidate IDs

4. If user provided a text keyword:
     Encode keyword with CLIP text encoder + text_projection
     → 1024-dim L2-normalized float32 vector
     Search the SAME FAISS index → top-5000 IDs
     Keep only candidate IDs that appear in both result sets,
     preserving the original visual similarity ranking order

5. Query SQLite for metadata of surviving candidate IDs
6. Filter by price range (price_min to price_max)
7. Sort by original FAISS visual similarity rank
8. Return top-100 results as JSON
```

**`GET /get-image`** fetches the live primary image URL for a given product from Tokopedia's GraphQL API in real time, since image URLs are not stored statically in the metadata database.

---

## Frontend: HTML / CSS / JavaScript

The frontend is a plain single-page application built with **HTML, CSS, and JavaScript** (no framework dependencies). The user flow is:

1. Upload a product image.
2. Crop the region of interest using the in-browser crop tool.
3. Optionally enter a product name or keyword to refine results.
4. Set a price range.
5. Submit. Results are rendered as product cards with name, price, rating, and live images loaded via the `/get-image` endpoint.

---

## Tools and Libraries

| Category | Library / Tool |
|---|---|
| Data Scraping | Python `requests`, Tokopedia GraphQL API |
| Data Storage | SQLite, Parquet, Kaggle Datasets |
| Image and Text Embedding | CLIP ViT-H/14 (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`) via Hugging Face Transformers |
| Vector Search | FAISS (IndexHNSWFlat, IndexIDMap2, METRIC_INNER_PRODUCT) |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript, Cropper.js |
| Model / Index Hosting | Hugging Face Hub (Datasets + Spaces) |
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

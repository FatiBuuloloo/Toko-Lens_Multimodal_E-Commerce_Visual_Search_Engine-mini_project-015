import os
import gc
import sqlite3
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

DB_PATH = "database_products.db"
IMAGE_DIR = "Picture_Product"
OUTPUT_PARQUET = "tokopedia_embeddings.parquet"
MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
BATCH_SIZE = 32

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"[ERROR] Database file '{DB_PATH}' not found!")

print(f"[INFO] Reading data from database: {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
df_final = pd.read_sql_query("SELECT * FROM tokopedia_product", conn)
conn.close()

df_final = df_final.drop_duplicates(subset=['Id'], keep='first')
print(f"[INFO] Successfully loaded {len(df_final):,} unique products from database.")

try:
    total_images = len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
    print(f"[INFO] Total images detected in '{IMAGE_DIR}': {total_images:,}")
except FileNotFoundError:
    print(f"[ERROR] Path '{IMAGE_DIR}' not found. Please check your folder name.")
    raise

class TokopediaDataset(Dataset):
    def __init__(self, dataframe, img_dir, processor):
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_product = row['Id']
        judul = row['Product_Name']
        
        img_name = f"{int(id_product)}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 224, 224)
            
        return pixel_values, str(id_product), str(judul)

print(f"[INFO] Loading CLIP model: {MODEL_NAME}")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME).cuda()
model.eval()

dataset = TokopediaDataset(df_final, IMAGE_DIR, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_img_vectors = []
all_ids = []
all_juduls = []

total_batches = len(dataloader)
print(f"\n[PROCESS] Starting embedding extraction for {total_batches} batches...")

with torch.no_grad():
    for i, (imgs, ids, juduls) in enumerate(dataloader):
        imgs = imgs.cuda()
        
        vision_outputs = model.vision_model(pixel_values=imgs)
        image_features = model.visual_projection(vision_outputs.pooler_output)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        all_img_vectors.append(image_features.cpu().numpy())
        all_ids.extend(ids)
        all_juduls.extend(juduls)
        
        if i % 50 == 0 or i == total_batches - 1:
            print(f" -> Batch {i}/{total_batches - 1} completed", flush=True)


print("\n[PROCESS] Structuring data and converting matrix to list...")
df_output = pd.DataFrame({
    'ID_Product': all_ids,
    'Judul': all_juduls,
    'Image_Embedding': np.vstack(all_img_vectors).tolist(),
})

print(f"[INFO] Saving dataframe to parquet format: {OUTPUT_PARQUET}")
df_output.to_parquet(OUTPUT_PARQUET, engine='pyarrow', index=False)
print(f"[SUCCESS] File successfully saved as {OUTPUT_PARQUET}")

del df_final, dataset, dataloader, df_output, all_img_vectors, all_ids, all_juduls
gc.collect()
torch.cuda.empty_cache()

print("Done!")
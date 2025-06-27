import cv2
import faiss
from tqdm import tqdm
from inference.models import PerceptionEncoder
import os
import faulthandler; faulthandler.enable()

pe = PerceptionEncoder(model_id="perception_encoder/PE-Core-B16-224", device="mps")

prompt = "coffee"
text_embedding = pe.embed_text(prompt)

DESKTOP = "/Users/james/Desktop"
INDEX_PATH = os.path.join(DESKTOP, "image_embeddings.index")

def get_files():
    files = os.listdir(DESKTOP)
    files.sort()
    files = [file for file in files if file.endswith(".png")]
    return files

if not os.path.exists(os.path.join(DESKTOP, "image_embeddings.index")):
    index = faiss.IndexFlatIP(1024)
    files = get_files()
    for file in tqdm(files, desc="Embedding images"):
        image_path = os.path.join(DESKTOP, file)
        image = cv2.imread(image_path)
        if image is not None:
            embedding = pe.embed_image(image)
            index.add(embedding)

    faiss.write_index(index, INDEX_PATH)
else:
    index = faiss.read_index(INDEX_PATH)

k = 5
query_embedding = text_embedding[0].reshape(1, 1024)
D, I = index.search(query_embedding, k)
files = get_files()

for i in range(k):
    print(f"Image /Users/james/Desktop/{files[I[0][i]]} with distance {D[0][i]}")

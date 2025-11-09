import os
import time
import io
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio

load_dotenv()

# Set Tesseract path if Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud='aws', region=PINECONE_ENV)
existing_index = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_index:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

def extract_pdf_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = []

    for page in doc:
        text = page.get_text("text").strip()

        # If text layer is mostly empty → page is scanned → OCR whole page
        if len(text) < 50:# make detection more reliable for scanned pages
            pix = page.get_pixmap(dpi=350)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # ---- Strong OCR Preprocessing ----
            img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # Upscale to improve clarity for OCR
            img_arr = cv2.resize(img_arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # Remove noise
            img_arr = cv2.GaussianBlur(img_arr, (5, 5), 0)

            # Automatic binarization (best for scanned pages)
            _, img_arr = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Optional: sharpen edges to improve character boundaries
            kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])
            img_arr = cv2.filter2D(img_arr, -1, kernel)

            img = Image.fromarray(img_arr)
            # ----------------------------------

            ocr_text = pytesseract.image_to_string(img, lang="eng").strip()
            final_text.append(ocr_text)

        else:
            final_text.append(text)

    doc.close()
    return "\n\n".join(final_text)


async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

     # ✅ Clear old vectors ONCE (not inside loop)
    index.delete(delete_all=True)
    print("✅ Cleared old vectors")

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        print(f"Extracting text + OCR: {file.filename}")
        full_text = extract_pdf_text_with_ocr(str(save_path))

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(full_text)

        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file.filename,
                "doc_id": doc_id,
                "role": role,
                "text": chunk      # ✅ Store actual content here
            }
            for chunk in chunks
        ]
      
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = await asyncio.to_thread(embed_model.embed_documents, chunks)

        print("Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file.filename}")
from fastapi import APIRouter, Depends, UploadFile, File
from auth.routes import authenticate
from .vectorstore import load_vectorstore
import uuid

router = APIRouter()

@router.post("/upload_docs")
async def upload_docs(
    user=Depends(authenticate),
    file: UploadFile = File(...)
):
    doc_id = str(uuid.uuid4())
    await load_vectorstore([file], doc_id)

    return {
        "message": f"{file.filename} uploaded successfully",
        "doc_id": doc_id
    }

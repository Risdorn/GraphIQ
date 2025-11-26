from services import vectordb, graphdb
from agents import extract_entities_relationship

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from io import BytesIO
import docx
import pdfplumber

# Test, DO NOT UNCOMMENT
# chunk_metadata = {
#     "chunk_id": 0,
#     "text": "Gradient Descent is used to optimize neural networks...",
#     "summary": "Gradient Descent and Neural Networks"
# }
# chunk_metadata["chunk_id"] = vectordb.add(chunk_metadata["text"], chunk_metadata["summary"])

# extract_entities_relationship(chunk_metadata)

app = FastAPI(title="GraphIQ Backend")
run_app = True # Make this false if you don't want to run backend

# Allow all origins (development)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"], # GET, POST, PUT, etc.
    allow_headers=["*"], # Content-Type, Authorization, etc.
)

# Document Parser, convert to string then use ingestion agents
def parse_docs(files: List[UploadFile]) -> List[str]:
    output = []

    for file in files:
        filename = file.filename.lower()

        # --- TXT FILE ---
        if filename.endswith(".txt"):
            text = file.file.read().decode("utf-8", errors="ignore")
            output.append(text)

        # --- DOCX FILE ---
        elif filename.endswith(".docx"):
            doc_bytes = BytesIO(file.file.read())
            doc = docx.Document(doc_bytes)
            text = "\n".join([p.text for p in doc.paragraphs])
            output.append(text)
        
        # --- PDF ---
        elif filename.endswith(".pdf"):
            try:
                pdf_bytes = BytesIO(file.file.read())
                text = ""
                with pdfplumber.open(pdf_bytes) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to read PDF file: {filename}"
                )
            output.append(text)
        
        # Reset internal pointer for reuse if needed
        file.file.seek(0)

    return output

def retrieve_reason(text):
    return len(text)

@app.post("/chat")
async def chat_endpoint(
    text: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    
    # Ensure at least one input is provided
    if (not text or text.strip() == "") and (not files or len(files) == 0):
        raise HTTPException(
            status_code=400,
            detail="Provide at least text or one document."
        )

    # Handle case where files=None
    files = files or []
    
    # Validate file types
    allowed_ext = {".txt", ".pdf", ".docx"}

    for file in files:
        filename = file.filename.lower()

        if not any(filename.endswith(ext) for ext in allowed_ext):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file.filename}"
            )
    
    # Process documents, Ingestion + Entity Extraction
    if len(files) > 0:
        output = parse_docs(files)
    
    # If text provided, reason and answer
    if len(text) > 0:
        result = retrieve_reason(text)

    return {
        "status": "success",
        "document_response": "Documents processed" if len(files) > 0 else "Not Applicable",
        "document_parsed": output if len(files) > 0 else "Not Applicable",
        "text_received": text,
        "details": result if len(text) > 0 else "Not Applicable"
    }

@app.get("/graph")
async def get_graph():
    nodes, relations = graphdb.get_graph()
    return {
        "status": "success",
        "nodes": nodes, 
        "relations": relations
    }

@app.get("/node/{node_name}")
async def get_node(node_name: str):
    node_info = graphdb.get_node(node_name)
    chunks = []
    for chunk_id in node_info["chunk_ids"]:
        if len(vectordb) < chunk_id:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Chunk not found"}
            )
        chunks.append(vectordb.get(chunk_id))
    node_info["chunks"] = chunks
    return {
        "status": "success",
        "node_info": node_info
    }

@app.get("/status")
async def status():
    chunks = len(vectordb)
    nodes, relations = graphdb.get_graph()
    nodes = len(nodes)
    relations = len(relations)
    return {
        "status": "success",
        "vectorstore_chunks": chunks,
        "neo4j_nodes": nodes,
        "neo4j_relations": relations
    }


if __name__ == "__main__" and run_app:
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
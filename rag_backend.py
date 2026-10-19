import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import uuid
import asyncio
import httpx
import tempfile
import chromadb
import time
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

runtime_config = {
    "opencode_url": "http://127.0.0.1:4096",
    "provider_id": "opencode",
    "model_id": "big-pickle"
}

CHROMA_URL = "http://127.0.0.1:8000"

print("正在加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
print("正在连接 ChromaDB...")
chroma_client = chromadb.HttpClient(host="127.0.0.1", port=8000)
vectorstore = Chroma(client=chroma_client, collection_name="knowledge_base", embedding_function=embeddings)

preview_cache = {}

@app.get("/api/health")
async def health():
    return {"ok": True}

@app.post("/api/config/mcp/apply-model")
async def apply_model(payload: dict):
    model_str = payload.get("model", "opencode/big-pickle")
    runtime_config["opencode_url"] = "http://127.0.0.1:4096"
    parts = model_str.split("/")
    if len(parts) >= 2:
        runtime_config["provider_id"] = parts[0]
        runtime_config["model_id"] = "/".join(parts[1:])
    else:
        runtime_config["provider_id"] = "opencode"
        runtime_config["model_id"] = model_str
    print(f"\n✅ [配置更新] 模型: {runtime_config['provider_id']} / {runtime_config['model_id']}")
    return {"ok": True, "config": runtime_config}

@app.post("/api/config/vdb/test")
async def test_vdb(payload: dict = None):
    url = CHROMA_URL
    if payload and payload.get("url"):
        url = payload.get("url").rstrip("/")
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(url + "/api/v2/heartbeat")
            if r.status_code == 200: return {"ok": True, "endpoint": "/api/v2/heartbeat"}
            r = await c.get(url + "/api/v1/heartbeat")
            if r.status_code == 200: return {"ok": True, "endpoint": "/api/v1/heartbeat"}
        except: pass
    return {"ok": False, "message": "连接失败"}

async def yield_text_as_stream(base_url: str, prompt: str):
    async with httpx.AsyncClient(timeout=120) as client:
        # 每次请求创建一个新 Session
        r = await client.post(f"{base_url}/session", json={"title": "rag-chat"})
        r.raise_for_status()
        session_id = r.json()["id"]
        
        body = {
            "parts": [{"type": "text", "text": prompt}],
            "model": {"providerID": runtime_config["provider_id"], "modelID": runtime_config["model_id"]},
            "agent": "build"
        }
        
        res = await client.post(f"{base_url}/session/{session_id}/message", json=body)
        if res.status_code != 200:
            # 提取具体的错误原因并直接返回给前端
            error_text = ""
            try:
                err_json = res.json()
                error_text = err_json.get("error", {}).get("message", "") or err_json.get("message", "")
            except:
                error_text = res.text[:200]
            yield f"❌ OpenCode 服务返回异常 ({res.status_code}): {error_text}"
            return
            
        data = res.json()
        full_text = ""
        for p in data.get("parts", []):
            if p.get("type") == "text":
                full_text += p.get("text", "")
                
        if not full_text:
            yield "（模型没有生成任何文本内容）"
            return

        i = 0
        while i < len(full_text):
            step = 2 + (i % 3)
            chunk = full_text[i:i+step]
            i += step
            yield chunk
            await asyncio.sleep(0.01)

@app.post("/api/opencode/test-model")
async def test_model_api():
    base_url = runtime_config["opencode_url"]
    try:
        full_content = ""
        async for chunk in yield_text_as_stream(base_url, "请回复：pong"):
            full_content += chunk
        return {"ok": True, "message": f"模型可用，返回内容: {full_content}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/opencode/chat-test")
async def chat_test_api(request: Request):
    payload = await request.json()
    msg = payload.get("message", "你好")
    base_url = runtime_config["opencode_url"]

    async def stream():
        try:
            async for chunk in yield_text_as_stream(base_url, msg):
                yield f"event: token\ndata: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: token\ndata: {json.dumps({'token': f'[请求失败] {str(e)}'}, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/api/ingest/preview")
async def preview(source_type: str = Form(...), file: UploadFile = File(None), title: str = Form(None), content: str = Form(None), slice_config: str = Form("{}")):
    cfg = json.loads(slice_config)
    chunk_size = cfg.get("chunkSize", 800)
    chunk_overlap = cfg.get("chunkOverlap", 150)
    docs = []
    if source_type == "file" and file:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            suffix_lower = suffix.lower()
            if suffix_lower == ".pdf": docs = PyPDFLoader(tmp_path).load()
            elif suffix_lower in [".docx", ".doc"]: docs = Docx2txtLoader(tmp_path).load()
            else: docs = TextLoader(tmp_path, encoding="utf-8").load()
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
        for doc in docs: doc.metadata["source"] = file.filename
    elif source_type == "url": docs = WebBaseLoader(title).load()
    else: docs = [Document(page_content=content, metadata={"source": title})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", "。", "！", "？", "；", " "])
    chunks = splitter.split_documents(docs)
    doc_id = uuid.uuid4().hex
    preview_cache[doc_id] = {"chunks": chunks, "source": title if source_type != 'file' else file.filename}
    lengths = [len(c.page_content) for c in chunks]
    return {
        "ok": True, "doc_id": doc_id,
        "stats": {"total": len(chunks), "avg_len": sum(lengths)//len(lengths) if lengths else 0, "max_len": max(lengths) if lengths else 0, "min_len": min(lengths) if lengths else 0},
        "chunks": [{"text": c.page_content, "length": len(c.page_content)} for c in chunks[:5]]
    }

@app.post("/api/ingest/execute")
async def execute(doc_id: str = Form(...)):
    cache_data = preview_cache.get(doc_id)
    if not cache_data:
        return {"ok": False, "message": "未找到切片缓存"}
    
    chunks = cache_data["chunks"]
    
    async def stream():
        yield f"event: stage\ndata: {json.dumps({'stage':'parse','progress':25,'message':'解析完成'})}\n\n"
        await asyncio.sleep(0.2)
        yield f"event: stage\ndata: {json.dumps({'stage':'split','progress':50,'message':'切片完成'})}\n\n"
        await asyncio.sleep(0.2)
        yield f"event: stage\ndata: {json.dumps({'stage':'embed','progress':75,'message':'向量化中'})}\n\n"
        
        if chunks: 
            vectorstore.add_documents(chunks)
            
        await asyncio.sleep(0.2)
        yield f"event: stage\ndata: {json.dumps({'stage':'write','progress':100,'message':'写入完成'})}\n\n"
        yield f"event: done\ndata: {json.dumps({'chunks': len(chunks)})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/api/kb/files")
async def get_kb_files():
    try:
        results = vectorstore.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        source_counts = {}
        for m in metadatas:
            if m and "source" in m:
                src = m["source"]
                source_counts[src] = source_counts.get(src, 0) + 1
                
        files = [{"name": k, "chunks": v} for k, v in source_counts.items()]
        files.sort(key=lambda x: x["name"])
        return {"ok": True, "files": files}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/kb/delete")
async def delete_kb_file(payload: dict):
    source = payload.get("source")
    if not source: return {"ok": False, "message": "缺少文件名"}
    try:
        vectorstore.delete(where={"source": source})
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/kb/chunks")
async def get_file_chunks(payload: dict):
    source = payload.get("source")
    try:
        results = vectorstore.get(where={"source": source}, include=["documents"])
        chunks = results.get("documents", [])
        return {"ok": True, "chunks": chunks}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    q = payload.get("question")
    history = payload.get("history", [])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(q)
    
    if not docs: context = "（无相关资料）"
    else: context = "\n\n".join([d.page_content for d in docs])
    
    # 限制历史最多 2 轮（4条消息），大幅降低超限概率
    recent_history = history[-4:] if history else []
    history_str = ""
    if recent_history:
        history_str = "【历史对话】\n"
        for msg in recent_history:
            role = "用户" if msg.get("role") == "user" else "助手"
            history_str += f"{role}: {msg.get('content')}\n"
        history_str += "\n"
        
    prompt_text = f"""你是一个知识库助手。请基于以下资料和对话历史回答问题。如果资料中没有答案，请直接回答不知道。
{history_str}
【资料】
{context}

【当前问题】
{q}"""

    base_url = runtime_config["opencode_url"]

    async def stream():
        sources = [{"index": i+1, "source": d.metadata.get("source", "未知"), "score": 0.9, "content": d.page_content[:100] + "..."} for i, d in enumerate(docs)]
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"
        
        try:
            async for chunk in yield_text_as_stream(base_url, prompt_text):
                yield f"event: token\ndata: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: token\ndata: {json.dumps({'token': f'[请求失败] {str(e)}'}, ensure_ascii=False)}\n\n"
            
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("\nRAG 后端已启动: http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)
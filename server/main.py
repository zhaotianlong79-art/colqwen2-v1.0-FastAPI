from io import BytesIO

import torch
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from fastapi import FastAPI, UploadFile, File, Form
from transformers.utils.import_utils import is_flash_attn_2_available
#pip install colpali_engine
#pip install uvicorn
#pip install fastapi
#pip install transformers
#pip install torch
#pip install Pillow


app = FastAPI()

# 小黄脸上下载colqwen2-v1.0 + colqwen2-base模型
#在colqwen2-v1.0目录下修改adapter_config.json："base_model_name_or_path": "自己的colqwen2-base模型",
MODEL_NAME = "本地模型目录"

model = ColQwen2.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda:3",  # 先加载到 GPU 3
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

# 如果有多块 GPU，用 DataParallel 包装
if torch.cuda.device_count() >= 2:
    model = torch.nn.DataParallel(model, device_ids=[3, 4])  # 使用 GPU 3 和 GPU 4

processor = ColQwen2Processor.from_pretrained(MODEL_NAME, use_fast=True)


@app.post("/process")
async def process(
        images: list[UploadFile] = File(...),
        queries: list[str] = Form(...)
):
    """处理图像和文本输入并返回相似度分数"""
    # 读取上传的图像
    pil_images = []
    for img_file in images:
        content = await img_file.read()
        pil_image = Image.open(BytesIO(content)).convert("RGB")
        pil_images.append(pil_image)

    batch_images = processor.process_images(pil_images)  # 数据先放到主 GPU
    batch_queries = processor.process_queries(queries)

    # 推理
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # 计算分数
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)

    return {
        "image_embeddings": image_embeddings.float().cpu().numpy().tolist(),
        "query_embeddings": query_embeddings.float().cpu().numpy().tolist(),
        "scores": scores.float().cpu().numpy().tolist()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=11111,
        workers=2,
        reload=False  # 生产环境建议关闭 reload
    )

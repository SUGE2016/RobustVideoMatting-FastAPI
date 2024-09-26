from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import shutil
import os
from inference import Converter
import tempfile
from typing import Optional, Tuple
import requests
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

class Settings(BaseModel):
    variant: str = 'mobilenetv3'
    checkpoint: str = f"checkpoints/rvm_mobilenetv3.pth"
    device: str = "cuda"

    
    @validator('checkpoint', pre=True, always=True)
    def set_checkpoint(cls, v, values):
        variant = values.get('variant', 'mobilenetv3')
        return f"checkpoints/rvm_{variant}.pth"

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.settings = Settings()
    app.state.converter = Converter(variant=app.state.settings.variant, checkpoint=app.state.settings.checkpoint, device=app.state.settings.device)

class MattingRequest(BaseModel):
    input_source: str
    input_resize: Optional[Tuple[int, int]] = None
    downsample_ratio: Optional[float] = None
    output_type: str = "video"
    output_composition: Optional[str] = None
    output_alpha: Optional[str] = None
    output_foreground: Optional[str] = None
    output_video_mbps: Optional[int] = None
    seq_chunk: int = 1
    num_workers: int = 0

@app.post("/matting")
async def matting(request: MattingRequest):
    input_path = ''
    temp_dir = tempfile.mkdtemp()

    try:
        if request.input_source.startswith(('http://', 'https://')):
            response = requests.get(request.input_source)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download the file")
            input_path = os.path.join(temp_dir, "input_file")
            with open(input_path, "wb") as f:
                f.write(response.content)
        else:
            if not os.path.exists(request.input_source):
                raise HTTPException(status_code=400, detail="Input file not found")
            input_path = request.input_source
        
        if not input_path:
            raise HTTPException(status_code=400, detail="Input file not found")

        output_composition = request.output_composition or os.path.join(temp_dir, "composition.mp4" if request.output_type == "video" else "composition")
        output_alpha = request.output_alpha or os.path.join(temp_dir, "alpha.mp4" if request.output_type == "video" else "alpha")
        output_foreground = request.output_foreground or os.path.join(temp_dir, "foreground.mp4" if request.output_type == "video" else "foreground")

        # 调用converter进行转换
        app.state.converter.convert(
            input_source=input_path,
            input_resize=request.input_resize,
            downsample_ratio=request.downsample_ratio,
            output_type=request.output_type,
            output_composition=output_composition,
            output_alpha=output_alpha,
            output_foreground=output_foreground,
            output_video_mbps=request.output_video_mbps,
            seq_chunk=request.seq_chunk,
            num_workers=request.num_workers,
            progress=True
        )

        # 返回处理后的文件
        if request.output_type == "video":
            print(output_composition)
            return FileResponse(output_composition, media_type="video/mp4", filename="composition.mp4")
        else:
            shutil.make_archive(output_composition, 'zip', output_composition)
            return FileResponse(f"{output_composition}.zip", media_type="application/zip", filename="composition.zip")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail="处理视频时发生错误")
    finally:
        #shutil.rmtree(temp_dir)
        pass

@app.get("/")
async def root():
    return {"message": "视频背景移除API"}
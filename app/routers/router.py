from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import numpy as np
import soundfile as sf
from pydantic import BaseModel
import tempfile
from app.services.ddsp_service import DDSPService
from ..services.separator_service import AudioSeparatorService
import logging
from typing import Optional, Dict

router = APIRouter()
separator_service = AudioSeparatorService()
ddsp_service = DDSPService()

logger = logging.getLogger(__name__)

class ProcessConfig(BaseModel):
    speaker_id: int = 1
    key: float = 0  # 变调参数
    enhance: bool = True
    pitch_extractor: str = 'rmvpe'
    f0_min: float = 50
    f0_max: float = 1100
    threhold: float = -60
    enhancer_adaptive_key: float = 0
    spk_mix_dict: Optional[Dict[str, float]] = None

@router.post("/voice/convert")
async def convert_voice(
    file: UploadFile = File(...),
    config: ProcessConfig = Form(...)
):
    """音频转换主接口"""
    temp_files = []
    try:
        # 创建输入临时文件
        input_path = f"{tempfile.gettempdir()}/input_{uuid.uuid4()}.wav"
        output_path = f"{tempfile.gettempdir()}/output_{uuid.uuid4()}.wav"
        temp_files.extend([input_path, output_path])
        
        # 保存上传的文件
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
            
        logger.info(f"Processing audio file: {input_path}")
        
        # 调用infer方法进行处理
        result_path = ddsp_service.infer(
            input_path=input_path,
            output_path=output_path,
            spk_id=config.speaker_id,
            spk_mix_dict=config.spk_mix_dict,
            key=config.key,
            enhance=config.enhance,
            pitch_extractor=config.pitch_extractor,
            f0_min=config.f0_min,
            f0_max=config.f0_max,
            threhold=config.threhold,
            enhancer_adaptive_key=config.enhancer_adaptive_key
        )
        
        return FileResponse(
            result_path,
            media_type='audio/wav',
            filename=f'converted_{uuid.uuid4()}.wav'
        )
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理所有临时文件
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {str(e)}")

@router.get("/speakers")
async def get_speakers():
    """
    获取可用的说话人列表
    """
    try:
        speakers = ddsp_service.get_speakers()
        return JSONResponse(content={"speakers": speakers})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/load")
async def load_model(
    model_path: str = Form(...),
):
    """
    加载/切换模型
    :param model_path: 模型路径
    """
    try:
        ddsp_service.__init__(model_path)
        return JSONResponse(content={"message": "Model loaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def get_model_info():
    """
    获取当前加载的模型信息
    """
    try:
        global ddsp_service
        model_info = ddsp_service.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/separate")
async def separate_audio(audio_file: UploadFile = File(...)):
    """分离音频轨道"""
    temp_files = []
    
    try:
        # 检查文件类型
        content_type = audio_file.content_type
        if not content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {content_type}. Must be audio/*"
            )

        # 创建临时文件并保存上传的音频
        input_path = f"{tempfile.gettempdir()}/input_{uuid.uuid4()}.wav"
        temp_files.append(input_path)
        
        # 读取并保存上传的文件
        content = await audio_file.read()
        with open(input_path, "wb") as f:
            f.write(content)
            
        logger.info(f"Saved uploaded file to {input_path}")
        
        # 分离音轨
        vocals, instruments, sr = await separator_service.separate_tracks(input_path)
        
        # 保存分离后的音频
        vocals_path = f"{tempfile.gettempdir()}/vocals_{uuid.uuid4()}.wav"
        instruments_path = f"{tempfile.gettempdir()}/instruments_{uuid.uuid4()}.wav"
        temp_files.extend([vocals_path, instruments_path])
        
        # 修复音频格式并保存
        logger.info(f"Vocals shape: {vocals.shape}, dtype: {vocals.dtype}")
        logger.info(f"Instruments shape: {instruments.shape}, dtype: {instruments.dtype}")
        
        # 确保音频数据是正确的格式和维度
        # soundfile需要的格式是 (samples, channels) 或者 (samples,) 对于单声道
        if vocals.ndim == 3:  # 如果是 [batch, channels, samples]
            vocals = vocals[0].T  # 转为 [samples, channels]
            instruments = instruments[0].T
        elif vocals.ndim == 2 and vocals.shape[0] <= 2:  # 如果是 [channels, samples]
            vocals = vocals.T     # 转为 [samples, channels]
            instruments = instruments.T
            
        # 确保数据类型正确
        if not isinstance(vocals, np.ndarray):
            vocals = vocals.numpy() if hasattr(vocals, 'numpy') else np.array(vocals)
        if not isinstance(instruments, np.ndarray):
            instruments = instruments.numpy() if hasattr(instruments, 'numpy') else np.array(instruments)
            
        # 转换为float32以确保兼容性
        vocals = vocals.astype(np.float32)
        instruments = instruments.astype(np.float32)
        
        # 检查是否有无效值
        if np.isnan(vocals).any() or np.isinf(vocals).any():
            logger.warning("Found NaN or Inf in vocals, replacing with zeros")
            vocals = np.nan_to_num(vocals)
        if np.isnan(instruments).any() or np.isinf(instruments).any():
            logger.warning("Found NaN or Inf in instruments, replacing with zeros")
            instruments = np.nan_to_num(instruments)
        
        # 规范化音频范围到 [-1, 1]
        max_val = max(np.abs(vocals).max(), np.abs(instruments).max())
        if max_val > 1.0:
            vocals /= max_val
            instruments /= max_val
        
        logger.info(f"After processing - Vocals shape: {vocals.shape}, range: [{vocals.min()}, {vocals.max()}]")
        
        # 保存处理后的音频
        sf.write(vocals_path, vocals, sr)
        sf.write(instruments_path, instruments, sr)
        
        logger.info("Audio separation completed successfully")
        
        # 返回文件路径可能导致清理问题，改为直接返回文件内容
        return {
            "vocals_path": vocals_path,
            "instruments_path": instruments_path,
            "sample_rate": sr
        }
        
    except Exception as e:
        logger.error(f"Audio separation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 只清理输入文件，输出文件需要留给客户端使用
        if temp_files and len(temp_files) > 0:
            try:
                os.unlink(temp_files[0])  # 只清理输入文件
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")

def save_audio(audio_data, file_path, sample_rate):
    """Save audio data to a file using soundfile"""
    sf.write(file_path, audio_data, sample_rate)

def _cleanup_temp_files(file_paths):
    """安全清理临时文件"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {str(e)}")
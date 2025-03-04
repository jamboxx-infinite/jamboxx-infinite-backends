from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import numpy as np
import soundfile as sf
from pydantic import BaseModel
import tempfile
from app.services import ddsp_service
from ..services.separator_service import AudioSeparatorService
import logging

router = APIRouter()
separator_service = AudioSeparatorService()
ddsp_service = ddsp_service.DDSPService()

logger = logging.getLogger(__name__)

class ProcessConfig(BaseModel):
    speaker_id: int = 0
    pitch_adjust: float = 0
    f0_min: float = 50
    f0_max: float = 1100
    threhold: float = -60
    enhance: bool = True

@router.post("/voice/convert")
async def convert_voice(
    file: UploadFile = File(...),
    config: ProcessConfig = Form(...),
):
    """
    音频转换主接口
    :param file: 上传的音频文件
    :param config: 转换配置参数
    :return: 转换后的音频文件
    """
    try:
        # 创建临时文件保存上传的音频
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        await file.seek(0)
        contents = await file.read()
        temp_input.write(contents)
        temp_input.close()
        
        # 调用DDSP处理逻辑
        processed_audio, sr = ddsp_service.process_audio(
            temp_input.name,
            config.pitch_adjust,
            config.speaker_id,
            config.f0_min,
            config.f0_max,
            config.threhold,
            config.enhance
        )
        
        # 保存处理后的音频
        sf.write(temp_output.name, processed_audio, sr)
        
        return FileResponse(
            temp_output.name,
            media_type='audio/wav',
            filename=f'converted_{uuid.uuid4()}.wav'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 清理临时文件
        os.unlink(temp_input.name)
        os.unlink(temp_output.name)

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
        ddsp_service.load_model(model_path)
        return JSONResponse(content={"message": "Model loaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def get_model_info():
    """
    获取当前加载的模型信息
    """
    try:
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
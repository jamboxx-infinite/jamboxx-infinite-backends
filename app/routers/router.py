from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import soundfile as sf
from pydantic import BaseModel
import tempfile
from app.services import ddsp_service
from ..services.separator_service import AudioSeparatorService

router = APIRouter()
separator_service = AudioSeparatorService()

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

@router.post("/voice/analyze")
async def analyze_voice(
    file: UploadFile = File(...),
):
    """
    分析音频特征
    :param file: 上传的音频文件
    :return: 音频特征数据
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        await file.seek(0)
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()
        
        # 分析音频特征
        features = ddsp_service.analyze_audio(temp_file.name)
        
        return JSONResponse(content=features)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        os.unlink(temp_file.name)

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
async def separate_audio(
    file: UploadFile = File(...),
):
    """
    分离音频中的人声和伴奏
    """
    try:
        # 保存上传文件
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_vocals = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_instruments = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        await file.seek(0)
        contents = await file.read()
        temp_input.write(contents)
        temp_input.close()
        
        # 执行分离
        vocals, instruments, sr = await separator_service.separate_tracks(temp_input.name)
        
        # 保存结果
        sf.write(temp_vocals.name, vocals, sr)
        sf.write(temp_instruments.name, instruments, sr)
        
        # 返回结果
        return {
            "vocals": FileResponse(
                temp_vocals.name,
                media_type='audio/wav',
                filename='vocals.wav'
            ),
            "instruments": FileResponse(
                temp_instruments.name, 
                media_type='audio/wav',
                filename='instruments.wav'
            )
        }
    finally:
        # 清理临时文件
        os.unlink(temp_input.name)
        os.unlink(temp_vocals.name)
        os.unlink(temp_instruments.name)
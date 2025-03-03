import os
import logging
from typing import Optional, Tuple, List, Dict
from fastapi import FastAPI, UploadFile, HTTPException
from app.ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from app.enhancer import Enhancer
import torch
import librosa
import numpy as np
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class DDSPService:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.args = None
        self.units_encoder = None
        self.enhancer = None
        self.current_model_path = None
        self._speakers_cache: Optional[Dict] = None
    
    def load_model(self, model_path: str) -> None:
        """
        加载或切换模型
        
        Args:
            model_path (str): 模型文件路径
            
        Raises:
            FileNotFoundError: 当模型文件不存在时
            RuntimeError: 当模型加载失败时
        """
        try:
            # 验证模型文件是否存在
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # 如果是同一个模型，跳过加载
            if self.current_model_path == model_path and self.model is not None:
                logger.info("Model already loaded")
                return
                
            logger.info(f"Loading model from {model_path}")
            
            # 释放现有模型的GPU内存
            if self.model is not None:
                self.model.cpu()
                torch.cuda.empty_cache()
            
            # 加载新模型
            self.model, self.args = load_model(model_path, device=self.device)
            
            # 重新初始化编码器
            self.units_encoder = Units_Encoder(
                self.args.data.encoder,
                self.args.data.encoder_ckpt,
                self.args.data.encoder_sample_rate,
                self.args.data.encoder_hop_size,
                device=self.device
            )
            
            # 重新初始化增强器
            self.enhancer = Enhancer(
                self.args.enhancer.type,
                self.args.enhancer.ckpt,
                device=self.device
            )
            
            # 更新当前模型路径
            self.current_model_path = model_path
            self._speakers_cache = None  # 清除说话人缓存
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # 清理可能部分加载的资源
            self._cleanup_resources()
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _cleanup_resources(self):
        """清理资源"""
        if self.model is not None:
            self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.units_encoder = None
        self.enhancer = None
        self._speakers_cache = None  # 清除说话人缓存
    
    def get_model_info(self) -> dict:
        """获取当前加载的模型信息"""
        return {
            "model_path": self.current_model_path,
            "device": self.device,
            "is_loaded": self.model is not None,
            "speakers": self.args.data.speakers if self.args else None
        }

    def get_speakers(self) -> List[Dict[str, any]]:
        """
        获取当前模型支持的说话人列表
        
        Returns:
            List[Dict]: 说话人列表，每个说话人包含id和name
            
        Raises:
            RuntimeError: 当模型未加载时
            ValueError: 当无法获取说话人信息时
        """
        try:
            # 检查模型是否已加载
            if self.model is None:
                raise RuntimeError("No model loaded. Please load a model first.")
                
            # 如果缓存存在且模型未改变，直接返回缓存
            if self._speakers_cache is not None and self.model is not None:
                return self._speakers_cache
                
            # 从模型配置中获取说话人信息
            if not hasattr(self.args.data, 'speakers'):
                raise ValueError("No speakers information found in model configuration")
                
            speakers_info = []
            for idx, speaker in enumerate(self.args.data.speakers):
                speakers_info.append({
                    "id": idx,
                    "name": speaker,
                    "description": f"Speaker {speaker}",
                    "is_active": True
                })
            
            # 更新缓存
            self._speakers_cache = speakers_info
            return speakers_info
            
        except Exception as e:
            logger.error(f"Error getting speakers list: {str(e)}")
            raise

    async def process_audio(
            self,
            audio_file: UploadFile,
            pitch_adjust: float = 0,
            speaker_id: int = 1
        ) -> Tuple[np.ndarray, int]:
        # 读取音频
        audio, sr = librosa.load(io.BytesIO(await audio_file.read()), sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
            
        hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
        
        # 提取音高
        f0_extractor = F0_Extractor('rmvpe', sr, hop_size)
        f0 = f0_extractor.extract(audio, uv_interp=True, device=self.device)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # 变调
        f0 = f0 * 2 ** (pitch_adjust / 12)
        
        # 提取音量包络
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # 提取特征
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sr, hop_size)
        
        # 转换
        spk_id = torch.LongTensor(np.array([[speaker_id]])).to(self.device)
        with torch.no_grad():
            audio_output, _, _ = self.model(units, f0, volume, spk_id=spk_id)
            
            # 增强
            audio_output, output_sr = self.enhancer.enhance(
                audio_output,
                self.args.data.sampling_rate,
                f0,
                self.args.data.block_size
            )
            
        # 转换为numpy
        audio_output = audio_output.squeeze().cpu().numpy()
        
        return audio_output, output_sr
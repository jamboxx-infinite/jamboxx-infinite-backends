import hashlib
import os
import logging
from typing import Optional, Tuple, List, Dict
from types import SimpleNamespace 
from fastapi import FastAPI, UploadFile, HTTPException
import tqdm
import yaml
from app.ddsp.core import upsample
from app.ddsp.vocoder import F0_Extractor, Units_Encoder, Volume_Extractor
from app.ddsp.vocoder import load_model
from app.enhancer import Enhancer
import torch
import librosa
import numpy as np
import soundfile as sf
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class DDSPService:
    def __init__(self, model_path=None, device=None):
        # 初始化服务
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # 设置默认模型路径
        if model_path is None:
            # 使用os.path.join确保跨平台兼容性
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, 'app', 'models', 'ddsp', 'Neuro_22000.pt')
            
        # 确保模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading DDSP model from {model_path} on {self.device}...")
        self.model, self.args = load_model(model_path, device=self.device)
        logger.info("Model loaded successfully")
        
        # 缓存目录
        self.cache_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir_path, exist_ok=True)
    
    def infer(self, 
              input_path,
              output_path,
              spk_id=1,
              spk_mix_dict=None,
              key=0,
              enhance=True,
              pitch_extractor='rmvpe',
              f0_min=50,
              f0_max=1100,
              threhold=-60,
              enhancer_adaptive_key=0):
        """执行DDSP推理转换"""
        
        # 加载输入音频
        audio, sample_rate = librosa.load(input_path, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        
        # 获取MD5哈希用于缓存
        md5_hash = ""
        with open(input_path, 'rb') as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
        
        # 检查并加载/提取音高曲线
        cache_file_path = os.path.join(
            self.cache_dir_path, 
            f"{pitch_extractor}_{hop_size}_{f0_min}_{f0_max}_{md5_hash}.npy"
        )
        
        if os.path.exists(cache_file_path):
            # 从缓存加载f0
            print('Loading pitch curves from cache...')
            f0 = np.load(cache_file_path, allow_pickle=False)
        else:
            # 提取f0
            print(f'Extracting pitch using {pitch_extractor}...')
            pitch_extractor_obj = F0_Extractor(
                pitch_extractor, 
                sample_rate, 
                hop_size, 
                float(f0_min), 
                float(f0_max)
            )
            f0 = pitch_extractor_obj.extract(audio, uv_interp=True, device=self.device)
            
            # 保存f0缓存
            np.save(cache_file_path, f0, allow_pickle=False)
        
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # 变调
        f0 = f0 * 2 ** (float(key) / 12)
        
        # 提取音量包络
        print('Extracting volume envelope...')
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # 加载单元编码器
        if self.args.data.encoder == 'cnhubertsoftfish':
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
            
        units_encoder = Units_Encoder(
            self.args.data.encoder, 
            self.args.data.encoder_ckpt, 
            self.args.data.encoder_sample_rate, 
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=self.device
        )
        
        # 加载增强器
        enhancer = None
        if enhance:
            print(f'Using enhancer: {self.args.enhancer.type}')
            enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)
        
        # 处理说话人ID或混合字典
        if spk_mix_dict is not None:
            print('Using mix-speaker mode')
        else:
            print(f'Using speaker ID: {spk_id}')
            
        spk_id_tensor = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)
        
        # 执行推理
        result = np.zeros(0)
        current_length = 0
        segments = self._split_audio(audio, sample_rate, hop_size, db_thresh=threhold)
        print(f'Processing {len(segments)} audio segments...')
        
        with torch.no_grad():
            for segment in tqdm(segments):
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
                
                seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
                
                seg_output, _, (s_h, s_n) = self.model(
                    seg_units, 
                    seg_f0, 
                    seg_volume, 
                    spk_id=spk_id_tensor, 
                    spk_mix_dict=spk_mix_dict
                )
                seg_output *= mask[:, start_frame * self.args.data.block_size : 
                                  (start_frame + seg_units.size(1)) * self.args.data.block_size]
                
                if enhance and enhancer:
                    seg_output, output_sample_rate = enhancer.enhance(
                        seg_output, 
                        self.args.data.sampling_rate, 
                        seg_f0, 
                        self.args.data.block_size, 
                        adaptive_key=enhancer_adaptive_key
                    )
                else:
                    output_sample_rate = self.args.data.sampling_rate
                
                seg_output = seg_output.squeeze().cpu().numpy()
                
                # 处理静音部分和拼接
                silent_length = round(start_frame * self.args.data.block_size * 
                                     output_sample_rate / self.args.data.sampling_rate) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = self._cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
        
        # 保存输出音频
        sf.write(output_path, result, output_sample_rate)
        print(f"Output saved to {output_path}")
        return output_path

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
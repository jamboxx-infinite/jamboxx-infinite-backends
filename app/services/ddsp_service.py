from fastapi import FastAPI, UploadFile, File
from app.ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from app.enhancer import Enhancer
import torch
import librosa
import numpy as np
import io

app = FastAPI()

# 全局变量存储加载的模型
class DDSPService:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 加载DDSP模型
        self.model_path = "path/to/model.pt"
        self.model, self.args = load_model(self.model_path, device=self.device)
        
        # 加载编码器
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt, 
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            device=self.device
        )
        
        # 加载增强器
        self.enhancer = Enhancer(
            self.args.enhancer.type,
            self.args.enhancer.ckpt,
            device=self.device
        )

    async def process_audio(
            self,
            audio_file: UploadFile,
            pitch_adjust: float = 0,
            speaker_id: int = 1
        ):
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
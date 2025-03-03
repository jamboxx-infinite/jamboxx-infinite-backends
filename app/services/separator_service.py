from pathlib import Path
import torch
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio

class AudioSeparatorService:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # 加载预训练模型
        self.model = get_model("htdemucs").to(device)
        
    async def separate_tracks(self, audio_path: str):
        """分离音频轨道"""
        # 加载音频
        wav = AudioFile(audio_path).read()
        ref = wav.mean(0)
        wav = torch.tensor(wav, device=self.device)
        
        # 执行分离
        sources = self.model.separate(wav.unsqueeze(0))
        sources = sources.cpu().numpy()[0]
        
        # 提取vocals和instruments
        vocals = sources[self.model.sources.index('vocals')]
        instruments = sources[self.model.sources.index('drums')] + \
                     sources[self.model.sources.index('bass')] + \
                     sources[self.model.sources.index('other')]
                     
        return vocals, instruments
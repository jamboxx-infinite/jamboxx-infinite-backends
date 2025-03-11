import subprocess
import torchaudio
import os
from demucs.audio import convert_audio
from demucs.apply import apply_model
from demucs import pretrained
import torch
from demucs.audio import AudioFile, save_audio
import logging
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import tempfile
import uuid

class AudioSeparatorService:
    def __init__(self, model_name="htdemucs"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.get_model(model_name).to(self.device)
        self.logger = logging.getLogger(__name__)

    async def convert_audio_to_wav(self, input_path: str) -> str:
        """
        将任何音频格式转换为 WAV
        """
        output_path = os.path.splitext(input_path)[0] + "_converted.wav"
        
        try:
            # 使用 FFmpeg 进行更稳健的转换
            result = subprocess.run([
                "ffmpeg", "-y", 
                "-i", input_path,
                "-acodec", "pcm_s16le",  # 16位PCM编码
                "-ac", "2",              # 双声道
                "-ar", "44100",          # 44.1kHz采样率
                "-f", "wav",             # WAV格式
                "-loglevel", "error",    # 只显示错误信息
                output_path
            ], capture_output=True, text=True, check=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.logger.info(f"Successfully converted audio to {output_path}")
                return output_path
            else:
                raise ValueError("Converted file is empty or not created")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg conversion error: {e.stderr}")
            raise RuntimeError(f"Audio conversion failed: {e.stderr}")
            
        except Exception as e:
            self.logger.error(f"Conversion error: {str(e)}")
            raise

    def validate_audio_file(self, audio_path: str) -> bool:
        """
        验证音频文件是否可读
        """
        try:
            info = sf.info(audio_path)
            self.logger.info(f"Audio file info: {info}")
            return True
        except Exception as e:
            self.logger.error(f"Audio validation failed: {str(e)}")
            return False

    async def separate_tracks(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """使用 Demucs 进行音源分离"""
        try:
            # 转换为WAV并验证
            wav_path = await self.convert_audio_to_wav(audio_path)
            if not self.validate_audio_file(wav_path):
                raise ValueError(f"Invalid audio file after conversion: {wav_path}")

            # 使用 soundfile 读取音频（更可靠的方式）
            audio_data, sr = sf.read(wav_path)
            
            # 确保音频是双声道的
            if len(audio_data.shape) == 1:
                audio_data = np.stack([audio_data, audio_data])
            elif len(audio_data.shape) == 2:
                audio_data = audio_data.T  # 转置为 [channels, samples] 格式
                
            # 转换为 torch tensor 并添加 batch 维度
            wav_tensor = torch.tensor(audio_data).unsqueeze(0).float()
            
            # 确保在正确的设备上
            wav_tensor = wav_tensor.to(self.device)
            
            self.logger.info(f"Processing audio tensor of shape: {wav_tensor.shape}")
            
            # 进行分离
            sources = apply_model(self.model, wav_tensor, split=True, overlap=0.25)[0]
            
            # 提取人声和伴奏
            vocals = sources[self.model.sources.index('vocals')].cpu().numpy()
            
            # 合并其他轨道为伴奏
            instruments = np.zeros_like(vocals)
            for source in ['drums', 'bass', 'other']:
                if source in self.model.sources:
                    instruments += sources[self.model.sources.index(source)].cpu().numpy()
            
            self.logger.info(f"Raw output - Vocals shape: {vocals.shape}, Instruments shape: {instruments.shape}")
            
            # 清理临时文件
            if os.path.exists(wav_path):
                os.unlink(wav_path)
                
            return vocals, instruments, sr
            
        except Exception as e:
            self.logger.error(f"Separation error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio separation failed: {str(e)}")

    async def merge_tracks(self, vocals_path: str, instruments_path: str, output_path: Optional[str] = None, 
                           vocals_volume: float = 1.5, instruments_volume: float = 1.0) -> str:
        """
        Merge vocals and instruments tracks into a single audio file
        
        Args:
            vocals_path: Path to the vocals audio file
            instruments_path: Path to the instruments audio file
            output_path: Optional path for the output file. If None, a temporary file will be created
            vocals_volume: Volume multiplier for vocals track (default: 1.5)
            instruments_volume: Volume multiplier for instruments track (default: 1.0)
            
        Returns:
            str: Path to the merged audio file
            
        Raises:
            RuntimeError: When merging fails
        """
        try:
            self.logger.info(f"Merging vocals from {vocals_path} and instruments from {instruments_path}")
            self.logger.info(f"Vocals volume: {vocals_volume}, Instruments volume: {instruments_volume}")
            
            # Validate input files
            if not os.path.exists(vocals_path):
                raise FileNotFoundError(f"Vocals file not found: {vocals_path}")
            if not os.path.exists(instruments_path):
                raise FileNotFoundError(f"Instruments file not found: {instruments_path}")
            
            # Create output path if not provided
            if output_path is None:
                output_path = f"{tempfile.gettempdir()}/merged_{uuid.uuid4()}.wav"
                
            # Get audio information to ensure compatibility
            vocals_info = sf.info(vocals_path)
            instruments_info = sf.info(instruments_path)
            
            self.logger.info(f"Vocals: {vocals_info.samplerate}Hz, {vocals_info.channels} channels, {vocals_info.frames} frames")
            self.logger.info(f"Instruments: {instruments_info.samplerate}Hz, {instruments_info.channels} channels, {instruments_info.frames} frames")
            
            # Ensure the sample rates match
            target_sr = max(vocals_info.samplerate, instruments_info.samplerate)
            
            # Use FFmpeg to mix the audio files with custom volume levels
            # The filter_complex command applies volume adjustments before mixing
            cmd = [
                "ffmpeg", "-y",
                "-i", vocals_path,
                "-i", instruments_path,
                "-filter_complex", 
                f"[0:a]volume={vocals_volume}[v];[1:a]volume={instruments_volume}[i];[v][i]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0",
                "-ar", str(target_sr),
                "-ac", "2",  # Output in stereo
                "-b:a", "192k",  # High quality bitrate
                output_path
            ]
            
            self.logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg mixing failed: {result.stderr}")
                
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Merged file was not created or is empty")
                
            self.logger.info(f"Successfully merged tracks to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error merging tracks: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to merge audio tracks: {str(e)}")

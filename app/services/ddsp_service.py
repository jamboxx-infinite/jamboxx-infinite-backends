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
        # Initialize service
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Set default model path
        if model_path is None:
            # Use os.path.join for cross-platform compatibility
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir,'pretrain', 'ddsp', 'Neuro_22000.pt')
            
        # Ensure model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading DDSP model from {model_path} on {self.device}...")
        self.model, self.args = load_model(model_path, device=self.device)
        logger.info("Model loaded successfully")
        
        # Cache directory
        self.cache_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir_path, exist_ok=True)

        # Set contentvec model path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.contentvec_path = os.path.join(
            self.base_dir,
            'pretrain',
            'contentvec',
            'checkpoint_best_legacy_500.pt'
        )
        
        # Check contentvec model file
        if not os.path.exists(self.contentvec_path):
            raise FileNotFoundError(
                f"ContentVec model not found at: {self.contentvec_path}\n"
                "Please download the model file from the official repository "
                "and place it in the correct directory."
        )
    
    def infer(self, 
              input_path,
              output_path,
              spk_id=1,
              key=0,
              enhance=True,
              pitch_extractor='rmvpe',
              f0_min=50,
              f0_max=1100,
              threhold=-60,
              enhancer_adaptive_key=0):
        """
        Perform DDSP inference conversion - Improved version
        Replace the input pure vocal audio with the selected model's voice
        
        Parameters:
            input_path: Input audio file path
            output_path: Output audio file path
            spk_id: Target speaker ID
            key: Pitch adjustment parameter (semitones)
            enhance: Whether to use audio enhancement
            pitch_extractor: Pitch extraction algorithm
            f0_min: Minimum pitch (Hz)
            f0_max: Maximum pitch (Hz)
            threhold: Volume threshold (dB)
            enhancer_adaptive_key: Enhancer adaptive key value
        """
        import os
        import tempfile
        import uuid
        
        # Load input audio
        audio, sample_rate = librosa.load(input_path, sr=None)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        
        # Get MD5 hash for cache
        md5_hash = ""
        with open(input_path, 'rb') as f:
            data = f.read()
            md5_hash = hashlib.md5(data).hexdigest()
        
        # Check and load/extract pitch curves
        cache_file_path = os.path.join(
            self.cache_dir_path, 
            f"{pitch_extractor}_{hop_size}_{f0_min}_{f0_max}_{md5_hash}.npy"
        )
        
        if os.path.exists(cache_file_path):
            # Load f0 from cache
            print('Loading pitch curves from cache...')
            f0 = np.load(cache_file_path, allow_pickle=False)
        else:
            # Extract f0
            print(f'Extracting pitch using {pitch_extractor}...')
            pitch_extractor_obj = F0_Extractor(
                pitch_extractor, 
                sample_rate, 
                hop_size, 
                float(f0_min), 
                float(f0_max)
            )
            f0 = pitch_extractor_obj.extract(audio, uv_interp=True, device=self.device)
            
            # Save f0 cache
            np.save(cache_file_path, f0, allow_pickle=False)
        
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # Pitch adjustment
        f0 = f0 * 2 ** (float(key) / 12)
        
        # Extract volume envelope
        print('Extracting volume envelope...')
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        
        # Load unit encoder
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
            device=self.device,
        )
        
        # Load enhancer
        enhancer = None
        if enhance:
            print(f'Using enhancer: {self.args.enhancer.type}')
            enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)
        
        # Process speaker ID
        print(f'Using speaker ID: {spk_id}')
        spk_id_tensor = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)
        
        try:
            # Process input audio directly
            audio_t = torch.tensor(audio).float().unsqueeze(0).to(self.device)
            
            # Extract features
            units = units_encoder.encode(audio_t, sample_rate, hop_size)
            
            # Process long audio in segments (if needed)
            if units.size(1) > 800:  # If audio is too long, process in segments
                # Pre-estimate result length, allocate enough space
                estimated_length = int(units.size(1) * self.args.data.block_size * sample_rate / self.args.data.sampling_rate)
                result_segments = []  # Store processed segments
                segment_positions = []  # Store segment start positions in final result
                
                # Split audio
                segment_length = 500  # 500 frames per segment
                overlap = 50  # 50 frames overlap
                
                for start_idx in range(0, units.size(1), segment_length - overlap):
                    end_idx = min(start_idx + segment_length, units.size(1))
                    print(f"Processing segment {start_idx/units.size(1)*100:.1f}% - [{start_idx}:{end_idx}]/{units.size(1)}")
                    
                    # Extract features for current segment
                    segment_units = units[:, start_idx:end_idx]
                    segment_f0 = f0[:, start_idx:end_idx]
                    segment_volume = volume[:, start_idx:end_idx]
                    
                    # Perform model inference
                    with torch.no_grad():
                        segment_output, _, _ = self.model(
                            segment_units,
                            segment_f0,
                            segment_volume,
                            spk_id=spk_id_tensor
                        )
                        
                        # Apply volume mask
                        segment_mask = mask[:, start_idx * self.args.data.block_size:(end_idx) * self.args.data.block_size]
                        if segment_mask.size(1) < segment_output.size(1):
                            segment_mask = torch.cat([segment_mask, segment_mask[:, -1:].repeat(1, segment_output.size(1) - segment_mask.size(1))], dim=1)
                        segment_output = segment_output * segment_mask[:, :segment_output.size(1)]
                        
                        # Apply enhancement
                        if enhance and enhancer:
                            segment_output, output_sample_rate = enhancer.enhance(
                                segment_output,
                                self.args.data.sampling_rate,
                                segment_f0,
                                self.args.data.block_size,
                                adaptive_key=enhancer_adaptive_key
                            )
                        else:
                            output_sample_rate = self.args.data.sampling_rate
                        
                        segment_output = segment_output.squeeze().cpu().numpy()
                    
                    # Save processed segment and position
                    result_segments.append(segment_output)
                    segment_positions.append(start_idx)
                
                # Merge all segments using improved method
                result = self._merge_segments(
                    result_segments, 
                    segment_positions, 
                    segment_length, 
                    overlap, 
                    self.args.data.block_size, 
                    output_sample_rate, 
                    self.args.data.sampling_rate
                )
                
            else:
                # Process shorter audio directly
                with torch.no_grad():
                    output, _, _ = self.model(
                        units,
                        f0,
                        volume,
                        spk_id=spk_id_tensor
                    )
                    
                    # Apply volume mask
                    output = output * mask[:, :output.size(1)]
                    
                    # Apply enhancement
                    if enhance and enhancer:
                        output, output_sample_rate = enhancer.enhance(
                            output,
                            self.args.data.sampling_rate,
                            f0,
                            self.args.data.block_size,
                            adaptive_key=enhancer_adaptive_key
                        )
                    else:
                        output_sample_rate = self.args.data.sampling_rate
                    
                    result = output.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"Error during voice conversion: {str(e)}")
            raise
        
        # Save output audio
        sf.write(output_path, result, output_sample_rate)
        print(f"Output saved to {output_path}")
        return output_path

    def _merge_segments(self, segments, positions, segment_length, overlap, block_size, output_sample_rate, input_sample_rate):
        """
        Improved segment merging method
        
        Parameters:
            segments: List of processed audio segments
            positions: Start positions of each segment
            segment_length: Length of each segment (frames)
            overlap: Overlap length (frames)
            block_size: Block size
            output_sample_rate: Output sample rate
            input_sample_rate: Input sample rate
        Returns:
            Merged complete audio
        """
        if len(segments) == 0:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        # Calculate overlap samples after conversion
        overlap_samples = int(overlap * block_size * output_sample_rate / input_sample_rate)
        
        # Estimate final output length
        last_pos = positions[-1]
        last_segment = segments[-1]
        total_length = int((last_pos + segment_length) * block_size * output_sample_rate / input_sample_rate)
        
        # Create output array of sufficient size
        merged = np.zeros(total_length, dtype=np.float32)
        weights = np.zeros(total_length, dtype=np.float32)
        
        # Iterate and merge all segments
        for i, (segment, pos) in enumerate(zip(segments, positions)):
            # Calculate start and end positions of current segment in output
            start_sample = int(pos * block_size * output_sample_rate / input_sample_rate)
            end_sample = start_sample + len(segment)
            
            # Ensure we don't exceed boundaries
            if end_sample > len(merged):
                end_sample = len(merged)
                segment = segment[:end_sample - start_sample]
            
            # Create weights - fade in/out at edges, weight=1 in middle
            weight = np.ones(len(segment))
            
            # If not the first segment, fade in at beginning
            if i > 0 and overlap_samples > 0:
                fade_in_length = min(overlap_samples, len(weight))
                weight[:fade_in_length] = np.linspace(0, 1, fade_in_length)
            
            # If not the last segment, fade out at end
            if i < len(segments) - 1 and overlap_samples > 0:
                fade_out_length = min(overlap_samples, len(weight))
                weight[-fade_out_length:] = np.linspace(1, 0, fade_out_length)
            
            # Add weighted segment to result
            current_range = slice(start_sample, start_sample + len(segment))
            merged[current_range] += segment * weight
            weights[current_range] += weight
        
        # Avoid division by zero
        valid_indices = weights > 0
        merged[valid_indices] /= weights[valid_indices]
        
        return merged

    def _cross_fade(self, y1, y2, length):
        """
        Cross-fade between two audio segments
        
        Parameters:
            y1: First audio segment
            y2: Second audio segment
            length: Overlap position
            
        Returns:
            Mixed audio
        """
        if length < 1:
            return np.append(y1, y2)
        
        # Ensure length doesn't exceed y1 or y2 length
        length = min(length, len(y1), len(y2))
        
        # Create linear fade in/out curves, ensure length match
        fade_out = np.linspace(1, 0, length)
        fade_in = np.linspace(0, 1, length)
        
        # Apply cross-fade
        y1_end = y1[-length:]
        y2_start = y2[:length]
        
        # Output debug information
        print(f"Cross-fade - y1_end shape: {y1_end.shape}, y2_start shape: {y2_start.shape}, fade curves length: {len(fade_in)}")
        
        # Ensure all arrays have matching lengths
        min_len = min(len(y1_end), len(y2_start), len(fade_in))
        if min_len < length:
            print(f"Warning: Adjusting fade length from {length} to {min_len}")
            length = min_len
            fade_out = np.linspace(1, 0, length)
            fade_in = np.linspace(0, 1, length)
            y1_end = y1[-length:]
            y2_start = y2[:length]
        
        # Apply fades
        y1[-length:] = y1_end * fade_out
        y2_first = y2_start * fade_in
        
        # Merge results
        result = np.append(y1[:-length], y1[-length:] + y2_first)
        result = np.append(result, y2[length:])
        
        return result

    def get_speakers(self) -> List[Dict[str, any]]:
        """
        Get the list of speakers supported by the current model
        
        Returns:
            List[Dict]: Speaker list, each speaker contains id and name
            
        Raises:
            RuntimeError: When no model is loaded
            ValueError: When speaker information cannot be obtained
        """
        try:
            # Check if model is loaded
            if self.model is None:
                raise RuntimeError("No model loaded. Please load a model first.")
                
            # If cache exists and model hasn't changed, return from cache
            if self._speakers_cache is not None and self.model is not None:
                return self._speakers_cache
                
            # Get speaker information from model config
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
            
            # Update cache
            self._speakers_cache = speakers_info
            return speakers_info
            
        except Exception as e:
            logger.error(f"Error getting speakers list: {str(e)}")
            raise
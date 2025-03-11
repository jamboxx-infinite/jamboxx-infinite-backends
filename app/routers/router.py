import json
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

class VoiceConvertConfig(BaseModel):
    speaker_id: int = 1
    key: int = 0
    enhance: bool = True
    pitch_extractor: str = "rmvpe"
    f0_min: int = 50
    f0_max: int = 1100
    threhold: int = -60
    enhancer_adaptive_key: int = 0

@router.post("/voice/convert")
async def convert_voice(
    file: UploadFile = File(...),
    speaker_id: int = Form(1),
    key: int = Form(0),
    enhance: bool = Form(True),
    pitch_extractor: str = Form("rmvpe"),
    f0_min: int = Form(50),
    f0_max: int = Form(1100),
    threhold: int = Form(-60),
    enhancer_adaptive_key: int = Form(0)
):
    """Voice conversion main interface"""
    temp_files = []
    input_path = None
    output_path = None
    
    try:

        # Add debug logs
        logger.debug("=== Request Parameters ===")
        logger.debug(f"Filename: {file.filename}")
        logger.debug(f"Content type: {file.content_type}")
        logger.debug(f"Parameters:")
        logger.debug(f"- speaker_id: {speaker_id}")
        logger.debug(f"- key: {key}")
        logger.debug(f"- enhance: {enhance}")
        logger.debug(f"- pitch_extractor: {pitch_extractor}")
        logger.debug(f"- f0_min: {f0_min}")
        logger.debug(f"- f0_max: {f0_max}")
        logger.debug(f"- threhold: {threhold}")
        logger.debug(f"- enhancer_adaptive_key: {enhancer_adaptive_key}")
        
        # Create input temp file
        input_path = f"{tempfile.gettempdir()}/input_{uuid.uuid4()}.wav"
        output_path = f"{tempfile.gettempdir()}/output_{uuid.uuid4()}.wav"
        # Only record input file, output file will be handled separately
        temp_files.append(input_path)
        
        # Log file save path
        print(f"Saving input file to: {input_path}")
        
        # Save uploaded file
        content = await file.read()
        print(f"File content size: {len(content)} bytes")
        
        with open(input_path, "wb") as f:
            f.write(content)
            
        print(f"Processing audio file: {input_path}")
        
        config = VoiceConvertConfig(
            speaker_id=speaker_id,
            key=key,
            enhance=enhance,
            pitch_extractor=pitch_extractor,
            f0_min=f0_min,
            f0_max=f0_max,
            threhold=threhold,
            enhancer_adaptive_key=enhancer_adaptive_key
        )
        
        # Call infer method for processing
        result_path = ddsp_service.infer(
            input_path=input_path,
            output_path=output_path,
            spk_id=config.speaker_id,
            key=config.key,
            enhance=config.enhance,
            pitch_extractor=config.pitch_extractor,
            f0_min=config.f0_min,
            f0_max=config.f0_max,
            threhold=config.threhold,
            enhancer_adaptive_key=config.enhancer_adaptive_key
        )
        
        # Verify output file exists and is valid
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Converted file does not exist: {result_path}")
            
        logger.info(f"Preparing to return file: {result_path}, size: {os.path.getsize(result_path)} bytes")
        
        # Use safer method to return file
        # 1. First read file content into memory
        with open(result_path, 'rb') as f:
            file_content = f.read()
            
        # 2. Specify content type and filename when returning response
        filename = f'converted_{uuid.uuid4()}.wav'
        
        # 3. Use JSONResponse to return file link instead of direct file
        # This may bypass some file handling issues
        if len(file_content) > 0:
            # Save to new location using relative path
            static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
            # Create subdirectory for this endpoint
            convert_dir = os.path.join(static_dir, "convert")
            os.makedirs(convert_dir, exist_ok=True)
            
            final_output_path = os.path.join(convert_dir, f"output_{uuid.uuid4()}.wav")
            
            with open(final_output_path, 'wb') as f:
                f.write(file_content)
                
            # Build URL path
            url_path = f"/static/convert/{os.path.basename(final_output_path)}"
            
            # Return file URL instead of direct file
            return JSONResponse({
                "status": "success", 
                "message": "Conversion complete", 
                "file_url": url_path,
                "file_size": len(file_content)
            })
        else:
            raise ValueError("Generated audio file is empty")
        
    except Exception as e:
        logger.error(f"Voice conversion failed: {str(e)}", exc_info=True)
        # If processing fails, ensure output file is cleaned up
        if 'output_path' in locals() and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception as clean_err:
                logger.warning(f"Failed to clean output file: {str(clean_err)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up all temp files
        for path in temp_files:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {str(e)}")
                
        # Also clean up output file since we've copied its content
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Failed to delete output file {output_path}: {str(e)}")

@router.get("/speakers")
async def get_speakers():
    """
    Get available speakers list
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
    Load/switch models
    :param model_path: Model path
    """
    try:
        ddsp_service.__init__(model_path)
        return JSONResponse(content={"message": "Model loaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
async def get_model_info():
    """
    Get information about the currently loaded model
    """
    try:
        global ddsp_service
        model_info = ddsp_service.get_model_info()
        return JSONResponse(content=model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/separate")
async def separate_audio(audio_file: UploadFile = File(...)):
    """Separate audio tracks"""
    temp_files = []
    
    try:
        # Check file type
        content_type = audio_file.content_type
        if not content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {content_type}. Must be audio/*"
            )

        # Create temp file and save uploaded audio
        input_path = f"{tempfile.gettempdir()}/input_{uuid.uuid4()}.wav"
        temp_files.append(input_path)
        
        # Read and save uploaded file
        content = await audio_file.read()
        with open(input_path, "wb") as f:
            f.write(content)
            
        logger.info(f"Saved uploaded file to {input_path}")
        
        # Separate tracks
        vocals, instruments, sr = await separator_service.separate_tracks(input_path)
        
        # Save separated audio
        vocals_path = f"{tempfile.gettempdir()}/vocals_{uuid.uuid4()}.wav"
        instruments_path = f"{tempfile.gettempdir()}/instruments_{uuid.uuid4()}.wav"
        temp_files.extend([vocals_path, instruments_path])
        
        # Fix audio format and save
        logger.info(f"Vocals shape: {vocals.shape}, dtype: {vocals.dtype}")
        logger.info(f"Instruments shape: {instruments.shape}, dtype: {instruments.dtype}")
        
        # Ensure audio data has correct format and dimensions
        # soundfile requires format (samples, channels) or (samples,) for mono
        if vocals.ndim == 3:  # If [batch, channels, samples]
            vocals = vocals[0].T  # Convert to [samples, channels]
            instruments = instruments[0].T
        elif vocals.ndim == 2 and vocals.shape[0] <= 2:  # If [channels, samples]
            vocals = vocals.T     # Convert to [samples, channels]
            instruments = instruments.T
            
        # Ensure correct data type
        if not isinstance(vocals, np.ndarray):
            vocals = vocals.numpy() if hasattr(vocals, 'numpy') else np.array(vocals)
        if not isinstance(instruments, np.ndarray):
            instruments = instruments.numpy() if hasattr(instruments, 'numpy') else np.array(instruments)
            
        # Convert to float32 for compatibility
        vocals = vocals.astype(np.float32)
        instruments = instruments.astype(np.float32)
        
        # Check for invalid values
        if np.isnan(vocals).any() or np.isinf(vocals).any():
            logger.warning("Found NaN or Inf in vocals, replacing with zeros")
            vocals = np.nan_to_num(vocals)
        if np.isnan(instruments).any() or np.isinf(instruments).any():
            logger.warning("Found NaN or Inf in instruments, replacing with zeros")
            instruments = np.nan_to_num(instruments)
        
        # Normalize audio range to [-1, 1]
        max_val = max(np.abs(vocals).max(), np.abs(instruments).max())
        if max_val > 1.0:
            vocals /= max_val
            instruments /= max_val
        
        logger.info(f"After processing - Vocals shape: {vocals.shape}, range: [{vocals.min()}, {vocals.max()}]")
        
        # Save processed audio to temp files
        sf.write(vocals_path, vocals, sr)
        sf.write(instruments_path, instruments, sr)
        
        logger.info("Audio separation completed successfully")
        
        # Save to static directory under separator subfolder
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
        separator_dir = os.path.join(static_dir, "separator")
        os.makedirs(separator_dir, exist_ok=True)
        
        # Copy files to static directory
        static_vocals_path = os.path.join(separator_dir, f"vocals_{uuid.uuid4()}.wav")
        static_instruments_path = os.path.join(separator_dir, f"instruments_{uuid.uuid4()}.wav")
        
        with open(vocals_path, 'rb') as f_in:
            with open(static_vocals_path, 'wb') as f_out:
                f_out.write(f_in.read())
                
        with open(instruments_path, 'rb') as f_in:
            with open(static_instruments_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Build URL paths
        vocals_url = f"/static/separator/{os.path.basename(static_vocals_path)}"
        instruments_url = f"/static/separator/{os.path.basename(static_instruments_path)}"
        
        # Return paths to the static files
        return {
            "vocals_path": vocals_path,
            "instruments_path": instruments_path,
            "vocals_url": vocals_url,
            "instruments_url": instruments_url,
            "sample_rate": sr
        }
        
    except Exception as e:
        logger.error(f"Audio separation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Only clean up input file, output files need to be kept for client
        if temp_files and len(temp_files) > 0:
            try:
                os.unlink(temp_files[0])  # Only clean up input file
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {str(e)}")

def save_audio(audio_data, file_path, sample_rate):
    """Save audio data to a file using soundfile"""
    sf.write(file_path, audio_data, sample_rate)

def _cleanup_temp_files(file_paths):
    """Safely clean up temporary files"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {str(e)}")

@router.post("/audio/merge")
async def merge_audio_tracks(
    vocals_path: str = Form(...),
    instruments_path: str = Form(...),
    vocals_volume: float = Form(1.5),
    instruments_volume: float = Form(1.0),
    output_filename: Optional[str] = Form(None)
):
    """
    Merge vocals and instruments tracks into a single audio file
    
    Args:
        vocals_path: Path to the vocals audio file
        instruments_path: Path to the instruments audio file
        vocals_volume: Volume multiplier for vocals track (default: 1.5)
        instruments_volume: Volume multiplier for instruments track (default: 1.0)
        output_filename: Optional custom filename for the output file
        
    Returns:
        JSON with the merged file URL and information
    """
    try:
        # Validate input paths
        if not os.path.exists(vocals_path):
            raise HTTPException(status_code=400, detail=f"Vocals file not found: {vocals_path}")
        if not os.path.exists(instruments_path):
            raise HTTPException(status_code=400, detail=f"Instruments file not found: {instruments_path}")
            
        logger.info(f"Merging audio tracks - Vocals: {vocals_path}, Instruments: {instruments_path}")
        
        # Create output path in temp directory
        output_path = f"{tempfile.gettempdir()}/merged_{uuid.uuid4()}.wav"
        
        # Call the merge_tracks function with volume parameters
        merged_path = await separator_service.merge_tracks(
            vocals_path=vocals_path,
            instruments_path=instruments_path,
            output_path=output_path,
            vocals_volume=vocals_volume,
            instruments_volume=instruments_volume
        )
        
        # Verify the merged file exists
        if not os.path.exists(merged_path):
            raise HTTPException(status_code=500, detail="Merged file was not created")
            
        logger.info(f"Audio tracks merged successfully to {merged_path}")
        
        # Read file content
        with open(merged_path, 'rb') as f:
            file_content = f.read()
            
        # Create final output path in static directory for serving
        filename = output_filename or f"merged_{uuid.uuid4()}.wav"
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
        merge_dir = os.path.join(static_dir, "merge")
        os.makedirs(merge_dir, exist_ok=True)
        
        final_output_path = os.path.join(merge_dir, filename)
        
        # Save content to static directory
        with open(final_output_path, 'wb') as f:
            f.write(file_content)
            
        # Build URL path
        url_path = f"/static/merge/{os.path.basename(final_output_path)}"
        
        # Clean up the temporary file
        try:
            os.unlink(merged_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary merged file: {str(e)}")
            
        # Return response with file URL and information
        return JSONResponse({
            "status": "success",
            "message": "Audio tracks merged successfully",
            "file_url": url_path,
            "file_size": len(file_content)
        })
        
    except Exception as e:
        logger.error(f"Audio merging failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Also add a more convenient endpoint that handles file uploads directly
@router.post("/audio/merge-uploads")
async def merge_uploaded_audio(
    vocals_file: UploadFile = File(...),
    instruments_file: UploadFile = File(...),
    vocals_volume: float = Form(1.5),
    instruments_volume: float = Form(1.0)
):
    """
    Merge uploaded vocals and instruments audio files
    
    Args:
        vocals_file: Uploaded vocals audio file
        instruments_file: Uploaded instruments audio file
        vocals_volume: Volume multiplier for vocals track (default: 1.5)
        instruments_volume: Volume multiplier for instruments track (default: 1.0)
        
    Returns:
        JSON with the merged file URL and information
    """
    temp_files = []
    
    try:
        # Create temporary files for uploads
        vocals_path = f"{tempfile.gettempdir()}/vocals_{uuid.uuid4()}.wav"
        instruments_path = f"{tempfile.gettempdir()}/instruments_{uuid.uuid4()}.wav"
        temp_files.extend([vocals_path, instruments_path])
        
        # Save uploaded files
        await vocals_file.seek(0)
        vocals_content = await vocals_file.read()
        with open(vocals_path, "wb") as f:
            f.write(vocals_content)
            
        await instruments_file.seek(0)
        instruments_content = await instruments_file.read()
        with open(instruments_path, "wb") as f:
            f.write(instruments_content)
            
        logger.info(f"Uploaded files saved to: {vocals_path}, {instruments_path}")
        
        # Create output path
        output_path = f"{tempfile.gettempdir()}/merged_{uuid.uuid4()}.wav"
        temp_files.append(output_path)
        
        # Merge the tracks with volume parameters
        merged_path = await separator_service.merge_tracks(
            vocals_path=vocals_path,
            instruments_path=instruments_path,
            output_path=output_path,
            vocals_volume=vocals_volume,
            instruments_volume=instruments_volume
        )
        
        # Read the merged file
        with open(merged_path, 'rb') as f:
            file_content = f.read()
            
        # Save to static directory for serving
        filename = f"merged_{uuid.uuid4()}.wav"
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
        merge_uploads_dir = os.path.join(static_dir, "merge-uploads")
        os.makedirs(merge_uploads_dir, exist_ok=True)
        final_output_path = os.path.join(merge_uploads_dir, filename)
        
        with open(final_output_path, 'wb') as f:
            f.write(file_content)
            
        # Build URL path
        url_path = f"/static/merge-uploads/{os.path.basename(final_output_path)}"
        
        return JSONResponse({
            "status": "success",
            "message": "Audio tracks merged successfully",
            "file_url": url_path,
            "file_size": len(file_content),
            "content_type": "audio/wav"
        })
        
    except Exception as e:
        logger.error(f"Audio merging failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files
        for path in temp_files:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {path}: {str(e)}")
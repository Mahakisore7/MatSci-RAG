# verify_gpu.py
import torch

# pyrefly: ignore [missing-import]
from src.logging_config import get_logger

log = get_logger(__name__)
log.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log.info(f"Device: {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM (GB): {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}")

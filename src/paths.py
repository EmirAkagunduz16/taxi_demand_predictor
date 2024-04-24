from pathlib import Path
import os

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR =  PARENT_DIR / 'data' / 'raw'
TRANSFORMED_DATA_DIR = PARENT_DIR / 'data' / 'transformed'

MODELS_DIR = PARENT_DIR / 'models'

if not Path(DATA_DIR).exists():
  os.nkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
  os.nkdir(RAW_DATA_DIR)

if not Path(TRANSFORMED_DATA_DIR).exists():
  os.nkdir(TRANSFORMED_DATA_DIR)

if not Path(MODELS_DIR).exists():
  os.makedirs(MODELS_DIR)
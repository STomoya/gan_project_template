from pathlib import Path

from training.utils import import_all_modules

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'training.train_loop')

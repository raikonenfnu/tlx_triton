"""Read-only foreign-allocation check; does not initialize the GPU runtime."""
import os
from pathlib import Path

def ownership():
    result = []
    for path in Path('/sys/class/kfd/kfd/proc').glob('*/vram_*'):
        try:
            size = int(path.read_text().strip())
            if size > 1048576 and int(path.parent.name) != os.getpid():
                result.append({'pid': int(path.parent.name), 'kfd_id': int(path.name.split('_')[1]), 'bytes': size})
        except (ValueError, PermissionError, FileNotFoundError):
            continue
    return result

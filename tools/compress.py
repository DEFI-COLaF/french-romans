import json
import gzip
from typing import Any

def dump(obj: Any, filename: str) -> None:
    """
    Serialize a Python object to JSON, compress it using gzip, and write it to a file.

    Args:
        obj: The Python object to serialize.
        filename: Path to the output .json.gz file.
    """
    with gzip.open(filename, 'wt', encoding='utf-8') as f:
        json.dump(obj, f)

def load(filename: str) -> Any:
    """
    Load a compressed JSON file and deserialize it into a Python object.

    Args:
        filename: Path to the input .json.gz file.

    Returns:
        The deserialized Python object.
    """
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        return json.load(f)

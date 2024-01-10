import os


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def get_data_dir() -> str:
    return _DATA_DIR

def resolve_data_file_path(file_name: str) -> str:
    return os.path.join(_DATA_DIR, file_name)
from pathlib import Path
from appdirs import user_cache_dir


cache_dir = Path(user_cache_dir("mitel_uniud"))
cache_dir.mkdir(exist_ok=True, parents=True)

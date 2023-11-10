import pickle
from os import PathLike
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar, Union

from .logger import print_log

K = TypeVar("K")
T = TypeVar("T")

# TODO: add lru
class Cache(Generic[K, T]):
    def __init__(
        self,
        func: Callable[[K], T],
        cache: bool,
        cache_root: Union[None, str, PathLike],
        key_func: Callable[[K], Hashable] = lambda x: x,
    ):
        """Create cache

        :param func: Function this cache evaluates
        :param cache: If true, do in memory caching.
        :param cache_root: If not None, cache to files at the provided path.
        :param key_func: Convert the key into a hashable object if needed
        """
        self.func = func
        self.key_func = key_func
        self.cache = cache
        if cache is True:
            print_log("Caching is activated", logger="current")
        self.cache_file = cache_root is not None
        self.cache_dict = dict()
        self.first = func
        if self.cache_file:
            self.cache_root = Path(cache_root)
            self.first = self.get_from_file
        if self.cache:
            self.first = self.get_from_dict

    def __getitem__(self, item: K, *args) -> T:
        return self.first(item, *args)

    def invalidate(self):
        """Invalidate entire cache."""
        self.cache_dict.clear()
        if self.cache_file:
            for p in self.cache_root.iterdir():
                p.unlink()

    def get(self, item: K, *args) -> T:
        return self.first(item, *args)

    def get_from_dict(self, item: K, *args) -> T:
        """Implements dict based cache."""
        cache_key = (self.key_func(item), *args)
        result = self.cache_dict.get(cache_key)
        if result is None:
            if self.cache_file:
                result = self.get_from_file(item, *args, cache_key=cache_key)
            else:
                result = self.func(item, *args)
            self.cache_dict[cache_key] = result
        return result

    def get_from_file(self, item: K, *args, cache_key=None) -> T:
        """Implements file based cache."""
        if cache_key is None:
            cache_key = (self.key_func(item), *args)
        filepath = self.cache_root / str(hash(cache_key))
        result = None
        if filepath.exists():
            with open(filepath, "rb") as f:
                (key, result) = pickle.load(f)
            if key == cache_key:
                return result
            else:
                # Hash collision! Handle this by overwriting the cache with the new query
                result = None
        if result is None:
            result = self.func(item, *args)
            with open(filepath, "wb") as f:
                pickle.dump((cache_key, result), f)
        return result

import pickle
from os import PathLike
from typing import Callable, Generic, Hashable, TypeVar, Union

from .logger import print_log

K = TypeVar("K")
T = TypeVar("T")
PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields


class Cache(Generic[K, T]):
    def __init__(
        self,
        func: Callable[[K], T],
        cache: bool = True,
        cache_file: Union[None, str, PathLike] = None,
        key_func: Callable[[K], Hashable] = lambda x: x,
        max_size: int = 4096,
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
        if cache is True or cache_file is not None:
            print_log("Caching is activated", logger="current")
            self._init_cache(cache_file, max_size)
            self.first = self.get_from_dict
        else:
            self.first = self.func

    def __getitem__(self, item: K, *args) -> T:
        return self.first(item, *args)

    def invalidate(self):
        """Invalidate entire cache."""
        self.cache_dict.clear()
        if self.cache_file:
            for p in self.cache_root.iterdir():
                p.unlink()

    def _init_cache(self, cache_file, max_size):
        self.cache = True
        self.cache_dict = dict()

        self.hits, self.misses, self.maxsize = 0, 0, max_size
        self.full = False
        self.root = []  # root of the circular doubly linked list
        self.root[:] = [self.root, self.root, None, None]

        if cache_file is not None:
            with open(cache_file, "rb") as f:
                cache_dict_from_file = pickle.load(f)
                self.maxsize += len(cache_dict_from_file)
                print_log(
                    f"Max size of the cache has been enlarged to {self.maxsize}.", logger="current"
                )
                for cache_key, result in cache_dict_from_file.items():
                    last = self.root[PREV]
                    link = [last, self.root, cache_key, result]
                    last[NEXT] = self.root[PREV] = self.cache_dict[cache_key] = link

    def get(self, obj, item: K, *args) -> T:
        return self.first(obj, item, *args)

    def get_from_dict(self, obj, item: K, *args) -> T:
        """Implements dict based cache."""
        # result = self.func(obj, item, *args)
        cache_key = (self.key_func(item), *args)
        link = self.cache_dict.get(cache_key)
        if link is not None:
            # Move the link to the front of the circular queue
            link_prev, link_next, _key, result = link
            link_prev[NEXT] = link_next
            link_next[PREV] = link_prev
            last = self.root[PREV]
            last[NEXT] = self.root[PREV] = link
            link[PREV] = last
            link[NEXT] = self.root
            self.hits += 1
            return result
        self.misses += 1

        result = self.func(obj, item, *args)

        if self.full:
            # Use the old root to store the new key and result.
            oldroot = self.root
            oldroot[KEY] = cache_key
            oldroot[RESULT] = result
            # Empty the oldest link and make it the new root.
            self.root = oldroot[NEXT]
            oldkey = self.root[KEY]
            oldresult = self.root[RESULT]
            self.root[KEY] = self.root[RESULT] = None
            # Now update the cache dictionary.
            del self.cache_dict[oldkey]
            self.cache_dict[cache_key] = oldroot
        else:
            # Put result in a new link at the front of the queue.
            last = self.root[PREV]
            link = [last, self.root, cache_key, result]
            last[NEXT] = self.root[PREV] = self.cache_dict[cache_key] = link
            if isinstance(self.maxsize, int):
                self.full = len(self.cache_dict) >= self.maxsize
        return result


def abl_cache(
    cache: bool = True,
    cache_file: Union[None, str, PathLike] = None,
    key_func: Callable[[K], Hashable] = lambda x: x,
    max_size: int = 4096,
):
    def decorator(func):
        cache_instance = Cache(func, cache, cache_file, key_func, max_size)

        def wrapper(self, *args, **kwargs):
            return cache_instance.get(self, *args, **kwargs)

        return wrapper

    return decorator

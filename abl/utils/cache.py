from typing import Callable, Generic, TypeVar

K = TypeVar("K")
T = TypeVar("T")
PREV, NEXT, KEY, RESULT = 0, 1, 2, 3  # names for the link fields


class Cache(Generic[K, T]):
    def __init__(self, func: Callable[[K], T]):
        """Create cache

        :param func: Function this cache evaluates
        :param cache: If true, do in memory caching.
        :param cache_root: If not None, cache to files at the provided path.
        :param key_func: Convert the key into a hashable object if needed
        """
        self.func = func
        self.has_init = False

    def __getitem__(self, obj, *args) -> T:
        return self.get_from_dict(obj, *args)

    def clear_cache(self):
        """Invalidate entire cache."""
        self.cache_dict.clear()

    def _init_cache(self, obj):
        if self.has_init:
            return

        self.cache = True
        self.cache_dict = dict()
        self.key_func = obj.key_func
        self.max_size = obj.cache_size

        self.hits, self.misses = 0, 0
        self.full = False
        self.root = []  # root of the circular doubly linked list
        self.root[:] = [self.root, self.root, None, None]

        self.has_init = True

    def get_from_dict(self, obj, *args) -> T:
        """Implements dict based cache."""
        # x is not used in cache key
        pred_pseudo_label, y, x, *res_args = args 
        cache_key = (self.key_func(pred_pseudo_label), self.key_func(y), *res_args)
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

        result = self.func(obj, *args)

        if self.full:
            # Use the old root to store the new key and result.
            oldroot = self.root
            oldroot[KEY] = cache_key
            oldroot[RESULT] = result
            # Empty the oldest link and make it the new root.
            self.root = oldroot[NEXT]
            oldkey = self.root[KEY]
            self.root[KEY] = self.root[RESULT] = None
            # Now update the cache dictionary.
            del self.cache_dict[oldkey]
            self.cache_dict[cache_key] = oldroot
        else:
            # Put result in a new link at the front of the queue.
            last = self.root[PREV]
            link = [last, self.root, cache_key, result]
            last[NEXT] = self.root[PREV] = self.cache_dict[cache_key] = link
            if isinstance(self.max_size, int):
                self.full = len(self.cache_dict) >= self.max_size
        return result


def abl_cache():
    def decorator(func):
        cache_instance = Cache(func)

        def wrapper(obj, *args):
            if obj.use_cache:
                cache_instance._init_cache(obj)
                return cache_instance.get_from_dict(obj, *args)
            else:
                return func(obj, *args)

        return wrapper

    return decorator

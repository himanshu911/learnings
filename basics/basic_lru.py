from collections import OrderedDict


class LRUCache[K, V]:
    """
    LRU (Least Recently Used) Cache implementation with O(1) operations.

    Design Considerations
    ---------------------

    Time complexity requirements:
    - get(key) → O(1)
    - put(key, value) → O(1)
    Anything slower than O(1) isn't acceptable for a real cache.

    Eviction policy: Least Recently Used (LRU)
    - If cache is full, evict the entry that hasn't been accessed for the longest time.
    - "Accessed" means either get or put.

    Data structure choices:
    - Hash map (dict in Python) → O(1) lookup of nodes by key.
    - Doubly Linked List (DLL) → O(1) move-to-front and remove-last operations.
    - Head = most recently used.
    - Tail = least recently used.
    - Together: HashMap + DLL = the classic LRU design.

    Capacity:
    - Fixed max size (capacity).
    - Once exceeded → evict the tail (least recently used).

    Implementation Note:
    - Uses OrderedDict which internally maintains insertion order using a hash table + DLL.
    - This gives us O(1) operations without manually implementing the linked list.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value  # Reinsert to mark as recently used
            return value
        return None

    def put(self, key: K, value: V) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value


if __name__ == "__main__":
    # Example with string keys and values
    cache_str = LRUCache[str, str](max_size=10)
    cache_str.put("name", "Alice")
    print(cache_str.get("name"))  # Alice

    # Example with int keys and dict values
    cache_int = LRUCache[int, dict[str, int]](max_size=5)
    cache_int.put(1, {"user_id": 100, "age": 30})
    print(cache_int.get(1))  # {'user_id': 100, 'age': 30}

    # Example with tuple keys and list values
    cache_tuple = LRUCache[tuple[int, int], list[int]](max_size=3)
    cache_tuple.put((1, 2), [10, 20, 30])
    print(cache_tuple.get((1, 2)))  # [10, 20, 30]

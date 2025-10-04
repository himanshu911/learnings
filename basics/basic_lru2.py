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
- Manually implements a doubly linked list to maintain order of usage.
"""


class Node[K, V]:
    def __init__(self, key: K | None, value: V | None):
        self.key = key
        self.value = value
        self.prev: Node[K, V] | None = None
        self.next: Node[K, V] | None = None


class LRUCache[K, V]:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: dict[K, Node[K, V]] = {}
        self.head: Node[K, V] = Node(None, None)  # Dummy head
        self.tail: Node[K, V] = Node(None, None)  # Dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node[K, V]) -> None:
        if node.prev is not None and node.next is not None:
            node.prev.next = node.next
            node.next.prev = node.prev

    def _add_to_front(self, node: Node[K, V]) -> None:
        node.prev = self.head
        node.next = self.head.next
        if self.head.next is not None:
            self.head.next.prev = node
        self.head.next = node

    def get(self, key: K) -> V | None:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_front(node)
            return node.value
        return None

    def put(self, key: K, value: V) -> None:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            node.value = value
            self._add_to_front(node)
        else:
            if len(self.cache) >= self.max_size:
                lru_node = self.tail.prev
                if lru_node is not None and lru_node.key is not None:
                    self._remove(lru_node)
                    del self.cache[lru_node.key]
            new_node = Node(key, value)
            self._add_to_front(new_node)
            self.cache[key] = new_node


if __name__ == "__main__":
    # Example with string keys and values
    cache_str = LRUCache[str, str](max_size=10)
    cache_str.put("name", "Alice")
    print(cache_str.get("name"))  # Alice

    # Example with int keys and dict values
    cache_int = LRUCache[int, dict[str, str]](max_size=5)
    cache_int.put(1, {"data": "value1"})
    print(cache_int.get(1))  # {'data': 'value1'}
    cache_int.put(2, {"data": "value2"})
    print(cache_int.get(2))  # {'data': 'value2'}

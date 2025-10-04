"""
__new__ vs __init__ — minimal, focused demo
"""


def line(title: str):
    print("\n" + "-" * 12, title)


# 1) __new__ creates; __init__ initializes
line("1) __new__ creates; __init__ initializes")


class Simple:
    def __init__(self, param: str):
        self.param = param


# Allocate without running __init__
obj = object.__new__(Simple)  # created but not initialized
print(type(obj).__name__, "created")
print("has 'param'?", hasattr(obj, "param"))  # False

# Now initialize manually
Simple.__init__(obj, "hello")
print("after __init__, param =", obj.param)  # "hello"


# 2) Use __new__ to control creation (e.g., singleton-ish)
line("2) Using __new__ to control creation")


class OneShot:
    _instance: "OneShot | None" = None

    def __new__(cls) -> "OneShot":
        if cls._instance is None:  # decide creation here
            cls._instance = super().__new__(cls)
        return cls._instance

    # no __init__ needed unless you have extra setup


a = OneShot()
b = OneShot()
print("a is b?", a is b)  # True


# 3) MRO: super().__new__ vs object.__new__
line("3) MRO: super().__new__ vs object.__new__")


class Base:
    def __new__(cls):
        print("Base.__new__ for", cls.__name__)
        return super().__new__(cls)


class Derived(Base):
    pass


# Path A: respects MRO → calls Base.__new__
d1 = super(Derived, Derived).__new__(Derived)
print("d1 created")

# Path B: bypass Base.__new__
d2 = object.__new__(Derived)
print("d2 created")

print("Derived MRO:", Derived.__mro__)

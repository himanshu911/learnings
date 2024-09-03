import fastcore.all as fc


class MyClass:
    def __init__(self, a, b, c=3):
        fc.store_attr()  # Automatically assigns a, b, c as attributes


obj = MyClass(1, 2)
print(obj.a)  # Output: 1
print(obj.b)  # Output: 2
print(obj.c)  # Output: 3


class Animal:
    def __init__(self, name):
        fc.store_attr()

    def make_sound(self, sound):
        print(f"{self.name} says {sound}")


# Use fastcore's patch to add new methods to Animal
@fc.patch
def sit(self: Animal):
    print(f"{self.name} sits down.")


@fc.patch
def roll_over(self: Animal):
    print(f"{self.name} rolls over.")


dog = Animal("Buddy")
dog.make_sound("Woof!")
dog.sit()
dog.roll_over()


def base_function(a, b=2, c=3):
    print(f"a: {a}, b: {b}, c: {c}")


@fc.delegates(base_function)  # This will import parameters from base_function
def derived_function(
    d, **kwargs
):  # Additional parameter `d` and kwargs to capture base function params
    print(f"d: {d}")
    base_function(**kwargs)


derived_function(10, a=1, b=4, c=5)

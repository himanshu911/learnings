import sys


def multiply(x, y):
    result = x * y
    return result


def add(x, y):
    result = x + y
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python example.py num1 num2")
        return

    num1 = int(sys.argv[1])
    num2 = int(sys.argv[2])

    # Put a breakpoint here to start debugging
    sum_result = add(num1, num2)
    print("Sum:", sum_result)

    # Step over this function call to see the result without entering the function
    product_result = multiply(num1, num2)
    print("Product:", product_result)

    diff_result = num1 - num2
    print(f"The difference of {num1} and {num2} is {diff_result}")


if __name__ == "__main__":
    main()

import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: python example.py num1 num2")
        return

    num1 = int(sys.argv[1])
    num2 = int(sys.argv[2])
    result = num1 + num2
    print(f"The sum of {num1} and {num2} is {result}")


if __name__ == "__main__":
    main()

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()

    # required positional argument
    parser.add_argument("name")

    # optional argument
    parser.add_argument("--occupation")

    # optional argument with default value
    parser.add_argument("--age", type=int, default=10)

    # boolean flag
    parser.add_argument("--verbose", action="store_true")

    # short and long versions of an optional argument
    parser.add_argument("-n", "--num", type=int)

    args = parser.parse_args()

    print("Hello,", args.name)
    if args.occupation:
        print("You are a", args.occupation)
    print("You are", args.age, "years old")
    if args.verbose:
        print("Verbose mode is on")
    if args.num:
        print("Number provided:", args.num)


if __name__ == "__main__":
    print(sys.argv)
    main()

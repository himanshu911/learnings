import asyncio
import time
from concurrent.futures import ProcessPoolExecutor


def fetch_data(params: int) -> str:
    print(f"Fetching data with params: {params}")
    time.sleep(params)  # Simulate a blocking I/O operation
    return f"Data fetched for params: {params}"


async def main():
    print("Starting Main Coroutine Function...")

    # Run in Threads
    print("*" * 20)
    task1 = asyncio.create_task(asyncio.to_thread(fetch_data, 1))
    task2 = asyncio.create_task(asyncio.to_thread(fetch_data, 2))
    result1 = await task1
    print("Result 1:", result1)
    result2 = await task2
    print("Result 2:", result2)

    # Run in Processes
    print("*" * 20)
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        task1 = loop.run_in_executor(executor, fetch_data, 1)
        task2 = loop.run_in_executor(executor, fetch_data, 2)
        result1 = await task1
        print("Result 1:", result1)
        result2 = await task2
        print("Result 2:", result2)

    print("Main coroutine Function Completed.")


if __name__ == "__main__":
    t1 = time.perf_counter()
    asyncio.run(main())
    t2 = time.perf_counter()
    print(f"Finished in {t2 - t1:.2f} seconds")

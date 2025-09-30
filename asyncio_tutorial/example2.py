import asyncio
import time


async def fetch_data(param: int) -> str:
    print(f"Fetching data with param: {param}")
    await asyncio.sleep(param)  # Simulate an async I/O operation
    return f"Data fetched for param: {param}"


async def main():
    print("Starting Main Coroutine Function...")

    # Method 1: create_task() - Use when you need individual task control/cancellation
    print("*" * 20)
    task1 = asyncio.create_task(fetch_data(1))
    task2 = asyncio.create_task(fetch_data(2))
    result1 = await task1
    print("Result 1:", result1)
    result2 = await task2
    print("Result 2:", result2)

    # Method 2: gather(coroutines) - Simple syntax, but lose individual task references
    print("*" * 20)
    coroutines = [fetch_data(i) for i in range(1, 3)]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    print(results)

    # Method 3: gather(tasks) - Best for concurrent execution with error handling
    print("*" * 20)
    tasks = [asyncio.create_task(fetch_data(i)) for i in range(1, 3)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(results)

    # Method 4: TaskGroup - Recommended for structured concurrency (Python 3.11+)
    # Guarantees all tasks complete or all are cancelled on first exception
    print("*" * 20)
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_data(i)) for i in range(1, 3)]
        # All tasks are awaited when the context manager exits.
    print(f"Task Group Results: {[task.result() for task in tasks]}")

    print("Main coroutine Function Completed.")


if __name__ == "__main__":
    t1 = time.perf_counter()
    asyncio.run(main())
    t2 = time.perf_counter()
    print(f"Finished in {t2 - t1:.2f} seconds")

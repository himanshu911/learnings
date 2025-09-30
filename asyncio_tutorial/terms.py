import asyncio
import time


def synchronous_function(param: str) -> str:
    print(f"Hello from a synchronous function! Received: {param}")
    time.sleep(0.2)
    return "Synchronous Result"


# ALSO KNOWN AS COROUTINE FUNCTION
async def asynchronous_function(param: str) -> str:
    print(f"Hello from an coroutine function! Received: {param}")
    await asyncio.sleep(0.2)
    return "Asynchronous Result"


async def main():
    print("Starting main function...")

    print("*" * 20)
    sync_result = synchronous_function("Test")
    print(sync_result)

    print("*" * 20)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    print(f"Empty Future: {future}")

    future.set_result("Future Result: Test")
    future_result = await future
    print(future_result)

    print("*" * 20)
    coroutine_obj = asynchronous_function("Test")
    print(f"Coroutine Object: {coroutine_obj}")
    coroutine_result = await coroutine_obj
    print(coroutine_result)

    print("*" * 20)
    task = asyncio.create_task(asynchronous_function("Test"))
    print(f"Task: {task}")
    task_result = await task
    print(task_result)

    print("Main function completed.")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio


async def set_future_result(future: asyncio.Future[str], value: str) -> str:
    await asyncio.sleep(0.2)
    # Set the result of the future
    future.set_result(value)
    print(f"Set the future's result to: {value}")
    return "Result set"


async def main():
    # Create a future object
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    # Schedule setting the future's result
    asyncio.create_task(set_future_result(future, "Future result is ready"))

    # Wait for the future's result
    result: str = await future
    print(f"Received the future's result: {result}")

    # Create another future object
    future: asyncio.Future[str] = loop.create_future()

    # Schedule setting the future's result
    task = asyncio.create_task(set_future_result(future, "Future result is ready"))

    # Wait for the task's result
    result: str = await task
    print(f"Received the task's result: {result}")


if __name__ == "__main__":
    asyncio.run(main())

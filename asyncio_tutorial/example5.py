import asyncio


async def waiter(event: asyncio.Event):
    print("Waiting for the event to be set")
    await event.wait()
    print("Event has been set, continuing execution")


async def setter(event: asyncio.Event):
    await asyncio.sleep(2)  # Simulate doing some work
    event.set()
    print("Event has been set!")


async def main():
    event: asyncio.Event = asyncio.Event()
    await asyncio.gather(waiter(event), setter(event))


asyncio.run(main())

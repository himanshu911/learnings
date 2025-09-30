import asyncio

shared_resource: int = 0  # A shared resource

# An asyncio Lock
lock = asyncio.Lock()


async def modify_shared_resource(i: int):
    global shared_resource
    print(f"Task {i} is waiting to modify the shared resource")
    async with lock:
        # Critical section starts
        print(f"Task {i}: Resource before modification: {shared_resource}")
        await asyncio.sleep(1)  # Simulate an IO operation
        shared_resource += 1  # Modify the shared resource
        print(f"Task {i}: Resource after modification: {shared_resource}")
        # Critical section ends


async def main():
    await asyncio.gather(*(modify_shared_resource(i) for i in range(5)))


asyncio.run(main())

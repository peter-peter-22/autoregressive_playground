import asyncio
import os

from dotenv import load_dotenv
from mega.client import MegaNzClient

load_dotenv()
email = os.getenv("MEGA_EMAIL")
password = os.getenv("MEGA_PASSWORD")
print(email, password)


async def mega_upload(local_path: str):
    async with MegaNzClient() as mega:
        await mega.login(email, password)

        with mega.progress_bar:
            result = await mega.upload(local_path)

        return result


if __name__ == '__main__':
    asyncio.run(mega_upload("output.zip"))  # The progress bar doesn't works when started with the run button

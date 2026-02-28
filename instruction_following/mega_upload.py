import asyncio
import os

from dotenv import load_dotenv
from mega.client import MegaNzClient

email = os.getenv("MEGA_EMAIL")
password = os.getenv("MEGA_PASSWORD")


async def mega_upload(local_path: str):
    async with MegaNzClient() as mega:
        await mega.login(email, password)

        with mega.progress_bar:
            result = await mega.upload(local_path)

        return result


if __name__ == '__main__':
    load_dotenv()
    asyncio.run(mega_upload("output.zip"))  # The progress bar doesn't work with asnycio.run

import asyncio

from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from aiogram import Bot

from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pathlib import Path
from PIL import Image

import os

from ocrmac import ocrmac

from app.database.requests import save_face_to_db, generate_code
from app.database.models import Code, async_session

MEDIA_FOLDER = Path("media")

user = Router()

@user.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"""
Hello, {message.from_user.full_name}!
Welcome to ApplyBot!
You will need to use /register to get a code, which you will use, and then send the picture that follows the rules!
Good luck!
"""
    )

@user.message(Command('register'))
async def register(message: Message):
    code = await generate_code()

    if code is not None:
        await message.answer(
            f"""
Your registration code is: {code}.
Remember! Your photo mustn't contain anything except for a code on a white sheet near your face!        
"""
        )
    else:
        await message.answer("Failed to generate a registration code. Please try again later.")

async def scan_photo(photo_path: Path) -> str:
    try:
        # Проверка, что файл существует и является изображением
        if not photo_path.exists() or not photo_path.is_file():
            return "Error: File does not exist."

        try:
            img = Image.open(photo_path)
            img.verify()  # Проверка, что файл действительно изображение
            img.close()
        except (IOError, SyntaxError) as e:
            return f"Error: Invalid image format. {e}"

        # Извлечение кода с помощью ocrmac
        ocr = ocrmac.OCR(str(photo_path))
        annotations = ocr.recognize()

        # Извлечение первого символа из каждой аннотации
        recognized_code = ''.join([i[0] for i in annotations]).strip()
        print("Recognized code:", recognized_code)
        return recognized_code if recognized_code else "No code found."
    except ValueError as e:
        return f"Error during OCR processing: {e}"


@user.message(F.photo)
async def download_photo(message: Message, bot: Bot):
    destination = MEDIA_FOLDER / f"{message.photo[-1].file_id}.jpg"

    await bot.download(
        message.photo[-1],
        destination=destination
    )

    try:
        img = Image.open(destination)
        img.verify()
    except (IOError, SyntaxError) as e:
        await message.reply(f"Downloaded file is not a valid image: {e}")
        return

    try:
        scanned_code = await scan_photo(destination)
        print(f"Scanned code: '{scanned_code}'")  # Логируем сканированный код

        async with async_session() as session:
            try:
                code_query = select(Code).filter_by(code=scanned_code)
                code = (await session.execute(code_query)).scalars().first()

                if code is None:
                    print(f"Code '{scanned_code}' not found in the database.")  # Логируем результат поиска
                    await message.reply("Code not found in the database.")
                    return

                code.activated = True
                code.picture = str(destination)
                await session.commit()
                await message.reply("Code found and activated! Picture saved.")
            except NoResultFound:
                await message.reply("Code not found in the database.")
    finally:
        pass
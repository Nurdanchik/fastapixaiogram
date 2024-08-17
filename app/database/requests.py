import random

from app.database.models import async_session, Face, Code
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_face_landmarks(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    return landmarks_array

async def save_face_to_db(name, image_path):
    landmarks_array = extract_face_landmarks(image_path)

    if landmarks_array is None:
        print("No face detected in the image.")
        return

    # Сериализация landmarks
    landmarks_binary = landmarks_array.tobytes()

    async with async_session() as session:
        # Проверка существования лица в базе данных
        existing_face = await session.execute(select(Face).filter_by(landmarks=landmarks_binary))
        existing_face = existing_face.scalars().first()

        if existing_face:
            print(f"Face with landmarks from '{image_path}' already exists in the database.")
            return

        new_face = Face(name=name, landmarks=landmarks_binary)
        session.add(new_face)

        try:
            await session.commit()
            print(f"Face '{name}' has been added to the database.")
        except IntegrityError:
            await session.rollback()
            print(f"Failed to add face '{name}' due to a unique constraint violation.")


async def generate_code():
    code_generated = random.randint(100, 999)

    async with async_session() as session:
        async with session.begin():
            # Ensure the code is unique
            while True:
                existing_code = await session.execute(select(Code).filter_by(code=code_generated))
                existing_code = existing_code.scalars().first()

                if existing_code is None:
                    break

                code_generated = random.randint(100, 999)

            new_code = Code(code=code_generated)
            session.add(new_code)

            try:
                await session.commit()
                return code_generated  # Return the generated code
            except IntegrityError:
                await session.rollback()
                print(f"Failed to add code {code_generated} due to a unique constraint violation.")
                return None

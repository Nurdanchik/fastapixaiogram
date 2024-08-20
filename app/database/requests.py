import random
import asyncio
import mediapipe as mp
import cv2
import numpy as np
from app.database.models import async_session, Face, Code
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
import pickle

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
    print("Extracted landmarks:", landmarks_array)
    return landmarks_array

async def save_face_to_db(name, image_path, code):
    landmarks_array = extract_face_landmarks(image_path)

    if landmarks_array is None:
        print("No face detected in the image.")
        return False, "No face detected in the image."

    # Serialize landmarks
    landmarks_binary = pickle.dumps(landmarks_array)

    async with async_session() as session:
        async with session.begin():
            # Check if face with the same name exists
            result = await session.execute(select(Face).filter(Face.name == name))
            existing_face = result.scalars().first()

            if existing_face:
                print(f"Face with name '{name}' already exists in the database.")
                return False, "This face is already registered."

            # Create new Face object
            new_face = Face(name=name, landmarks=landmarks_binary, picture=f'{image_path}', code=code)

            session.add(new_face)

            try:
                await session.commit()
                print(f"Face '{name}' has been added to the database.")
                return True, "Face has been registered successfully."
            except IntegrityError:
                await session.rollback()
                print(f"Failed to add face '{name}' due to a unique constraint violation.")
                return False, "Failed to add face due to a unique constraint violation."

async def generate_code():
    code_generated = random.randint(100, 999)

    async with async_session() as session:
        async with session.begin():
            # Ensure the code is unique
            while True:
                result = await session.execute(select(Code).filter(Code.code == code_generated))
                existing_code = result.scalars().first()

                if existing_code is None:
                    break

                code_generated = random.randint(100, 999)

            new_code = Code(code=code_generated)
            session.add(new_code)

            try:
                await session.commit()
                return code_generated
            except IntegrityError:
                await session.rollback()
                print(f"Failed to add code {code_generated} due to a unique constraint violation.")
                return None
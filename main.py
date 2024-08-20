from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Импортируем engine и Base
from app.database.models import async_session, engine  # из файла database
from app.database.models import Base, Face  # из файла database.models

import cv2
import numpy as np
import mediapipe as mp

# Создаем объект FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

app = FastAPI()

# Функция для извлечения лендмарков с использованием Mediapipe
def extract_face_landmarks(image_data):
    # Обработка изображения с помощью OpenCV и Mediapipe
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    landmarks_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    return landmarks_array

# Функция для получения асинхронной сессии базы данных
async def get_async_db() -> AsyncSession:
    async with async_session() as session:
        yield session

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...), db: AsyncSession = Depends(get_async_db)):
    image_data = await image.read()

    try:
        # Извлекаем лендмарки из загруженного изображения
        new_landmarks = extract_face_landmarks(image_data)
        if new_landmarks is None:
            raise HTTPException(status_code=400, detail="No face found in the image.")

        # Поиск совпадений в базе данных
        async with db.begin():
            result = await db.execute(select(Face))
            faces = result.scalars().all()

            for face in faces:
                # Преобразуем байты из базы обратно в numpy-массив
                landmarks_from_db = np.frombuffer(face.landmarks, dtype=np.float32).reshape(-1, 3)

                # Сравниваем массивы лендмарков
                if np.array_equal(landmarks_from_db, new_landmarks):
                    return {
                        "id": face.id,
                        "name": face.name,
                        "code": face.code,
                        "picture": face.picture,
                        "message": "Face matched!"
                    }

        # Если совпадение не найдено
        raise HTTPException(status_code=404, detail="No matching face found in the database.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Убедитесь, что таблицы базы данных созданы при запуске приложения
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
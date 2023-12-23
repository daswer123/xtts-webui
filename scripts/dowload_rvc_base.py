import os
import requests

# Путь к папке, где должны находиться файлы
folder = './rvc/base_model'

# Проверяем, существует ли папка. Если нет, создаем ее.
if not os.path.exists(folder):
    os.makedirs(folder)

# Список файлов и их URL для загрузки
files = {
    "hubert_base.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/hubert_base.pt",
    "rmvpe.pt": "https://huggingface.co/Daswer123/RVC_Base/resolve/main/rmvpe.pt"
}

# Проверяем каждый файл
for filename, url in files.items():
    file_path = os.path.join(folder, filename)

    # Если файл не существует, загружаем его
    if not os.path.exists(file_path):
        print(f'Файл {filename} не найден, начинается загрузка...')

        # Отправляем запрос на скачивание файла
        response = requests.get(url)

        # Проверяем, что запрос был успешным
        if response.status_code == 200:
            # Записываем содержимое ответа в файл
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'Файл {filename} успешно загружен.')
        else:
            print(f'Не удалось загрузить файл {filename}.')
    else:
        print(f'Файл {filename} уже существует.')
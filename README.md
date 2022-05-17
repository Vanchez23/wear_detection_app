## Описание
Приложение для определения спецодежды на человеке. 
На вход приложение принимает видеопоток и с помощью индикаторов указывает
присутствие спецодежды и ее комплектность.

### Используемое оборудование
- Ubuntu 20.04 / Windows 8.1 
- python==3.8.5

## Установка

1. Клонировать репозиторий
```bash
git clone https://gitlab.sch.ocrv.com.rzd/cv-research-group/wear_detection_app.git
cd wear_detection_app
```

2. Установить виртуальное окружение

   ### Ubuntu

    При помощи **virtualenv**:
   
    Установить **virtualenv**
    ```bash
    sudo python3 -m pip install virtualenv
    ````
   
    Создать виртуальное окружение
    ```bash
    virtualenv venv
    source venv/bin/activate
    ```

    При помощи **pyenv**:
   
    Установка [pyenv](https://khashtamov.com/ru/pyenv-python/)
   
    ```bash
    #install python version
    pyenv install 3.8.5
    pyenv virtualenv 3.8.5 wear_detection_app
    pyenv activate wear_detection_app
    ```

   ### Windows
    При помощи **virtualenv**:
   
    Установить **virtualenv**
    ```bash
    python -m pip install virtualenv
    ```

    Создать виртуальное окружение
    ```bash
    virtualenv venv
    venv\Scripts\activate
    ```
   
3. Устновить зависимости

```bash
python setup.py build develop
```

**CPU**
```bash
pip3 install -r requirements_cpu.txt
```

**GPU**
```bash
pip3 install -r requirements_gpu.txt
```

## Загрузить веса
### Ubuntu
Установить curl
```bash
sudo apt-get install curl
```
Скачать веса с помощью скрипта
```bash
python3 wearing_detector/utils/download_weights.py [--config_name=wearing_detector/configs/download_config.yaml][--overwrite=False]
```

### Windows
Для загрузки с помощью скрипта нужно 
1. [Скачать curl](https://curl.se/windows/)
2. Распаковать архив
3. Добавить путь к bin/curl.exe в path
4. Запустить скрипт
```bash
python wearing_detector\utils\download_weights.py [--config_name=wearing_detector\configs\download_config.yaml][--overwrite=False]
```

Или загрузить модели по ссылкам в папку weights:
 - [pose_hrnet_w32_256x192.pth](https://drive.google.com/file/d/1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38/view?usp=sharing)
 - [yolov5m_clothes_16_03_21_2v89f2b8.pt](https://drive.google.com/file/d/1CU82TBacloFUbvetKiZt8itaNT4Z85mA/view?usp=sharing)
 - [yolov5s.pt](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt)

## Запуск

1. Настроить конфигурацию в файле `wearing_detector/configs/wearing_detector_config.yaml`
    
**Основные настройки**

VIDEOSTREAM.PIPE - видеопоток (номер веб-камеры, путь к видео или строка подключения к ip-камере)

VIDEOSTREAM.SKIP_FRAMES_COUNT - количество пропускаемых кадров

[PERSON_MODEL, CLOTHES_MODEL, KEYPOINTS_MODEL].DEVICE - устройство на котором будут происходить вычисления

[PERSON_MODEL, CLOTHES_MODEL, KEYPOINTS_MODEL].WEIGHTS - путь до весов модели

2. Запустить приложение
```bash
python app/app.py
```

## (Опцианально) Установить **QtDesigner**
Для редактирования интерфейса
### Ubuntu
```bash
sudo apt install qttools5-dev-tools
```

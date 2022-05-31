# IDU_metrics

## Запуск приложения
1. УСтановить все зависимости в папке `/metrics`
```shell
pip install -r metrics/requirements.txt
```

2. Приложение FastAPI запускается через **uvicorn** в папке `/metrics`(!!!)
```shell
cd metrics
uvicorn app.main:app --host 0.0.0.0 --port 5000
```
По умолчанию режим DEBUG активен для Flask и FastAPI приложений.  
Чтобы отключить DEBUG можно настроить две переменные окружения: `FASTAPI_DEBUG` и `FLASK_DEBUG`

## Запуск контейнера
Dockerfile для сборки приложения находится в папке `/metrics`

## URLS
1. По url `/docs` доступна документация FastAPI.
1. По старым url доступны API из Flask.
1. Новые url от FastAPI доступны по старым url с префиксом `/api/v2`.
 Например, `/api/v2/mobility_analysis` вместо `/mobility_analysis`.

## Тестирование
1. Тесты находятся в папке `Tests/`.  
   По умолчанию тесты проверяют сервер указанный в файле `Tests/conf.py`.
2. Чтобы указать отличный сервер, нужно установить переменную окружения `APP_ADDRESS_FOR_TESTING`.
   - `APP_ADDRESS_FOR_TESTING = "127.0.0.1:5000/api/v2"` - Чтобы тестировать FastAPI (по умолчанию)
   - `APP_ADDRESS_FOR_TESTING = "127.0.0.1:5000"` - Чтобы тестировать Flask
3. Для запуска тестов необходимо в терминале выполнить команду
```shell
pytest
```
Более подробные инструкции с запуском отдельных тестовых методов, классов или модулей 
[здесь](https://docs.pytest.org/en/7.1.x/how-to/usage.html#how-to-invoke-pytest)

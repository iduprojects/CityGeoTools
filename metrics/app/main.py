from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from flask_app import app as flask_app
from app.routers import router
from app.core.config import settings


app = FastAPI(debug=settings.FASTAPI_DEBUG)
app.include_router(router, prefix="/api/v2")

app.mount("", WSGIMiddleware(flask_app))

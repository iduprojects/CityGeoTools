from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.middleware.cors import CORSMiddleware

from flask_app import app as flask_app
from app.routers import router
from app.core.config import settings


app = FastAPI(debug=settings.FASTAPI_DEBUG)
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='http://.*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v2")

app.mount("", WSGIMiddleware(flask_app))

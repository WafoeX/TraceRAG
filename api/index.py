"""Vercel serverless entrypoint.

Vercel's Python runtime serves the ASGI ``app`` exposed here. All routes are
rewritten to this function via vercel.json, and FastAPI handles routing.
"""
from app.main import app  # noqa: F401

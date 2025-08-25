import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from database import engine, Base, SessionLocal
import os


async def init_db():
    """Initialize the database by creating all tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Database tables created.")


if __name__ == "__main__":
    print("Initializing database...")
    asyncio.run(init_db())
    print("Database initialization complete.")

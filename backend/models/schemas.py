from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
import datetime


class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class MemoryBase(BaseModel):
    key: str
    value: str


class MemoryCreate(MemoryBase):
    pass


class Memory(MemoryBase):
    id: int
    user_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        orm_mode = True


# For bulk operations on memory
class MemoryBatch(BaseModel):
    memories: List[MemoryCreate]

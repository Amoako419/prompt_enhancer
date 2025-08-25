from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from datetime import datetime

from database import get_db
from auth.security import SECRET_KEY, ALGORITHM
from models.schemas import TokenData
from models.models import User
from sqlalchemy.future import select

# Define the token URL (must match the login endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


async def get_user_by_username(db: AsyncSession, username: str):
    """Get a user from the database by username"""
    result = await db.execute(select(User).filter(User.username == username))
    return result.scalars().first()


async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: AsyncSession = Depends(get_db)
):
    """Validate token and return current user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
            
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    # Get user from database
    user = await get_user_by_username(db, token_data.username)
    
    if user is None:
        raise credentials_exception
        
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if the user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

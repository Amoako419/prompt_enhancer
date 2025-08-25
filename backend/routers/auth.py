from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import timedelta

from database import get_db
from models.models import User
from models.schemas import Token, UserCreate, User as UserSchema
from auth.security import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from auth.dependencies import get_current_active_user

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
    responses={401: {"description": "Unauthorized"}},
)


@router.post("/register", response_model=UserSchema)
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    result = await db.execute(select(User).filter(User.username == user.username))
    db_user = result.scalars().first()
    
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
        
    # Check if email exists
    result = await db.execute(select(User).filter(User.email == user.email))
    db_user = result.scalars().first()
    
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
        
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    return db_user


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and return JWT token"""
    # Get user from database
    result = await db.execute(select(User).filter(User.username == form_data.username))
    user = result.scalars().first()
    
    # Verify user and password
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Generate token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires,
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserSchema)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
):
    """Get information about the currently authenticated user"""
    return current_user

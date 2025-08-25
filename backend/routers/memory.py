from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any
import json

from database import get_db
from models.models import Memory, User
from models.schemas import MemoryCreate, Memory as MemorySchema, MemoryBatch
from auth.dependencies import get_current_active_user

router = APIRouter(
    prefix="/api/memory",
    tags=["memory"],
    responses={401: {"description": "Unauthorized"}},
)


@router.post("/", response_model=MemorySchema)
async def create_memory(
    memory: MemoryCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create or update a memory for the current user"""
    # Check if memory with this key already exists
    result = await db.execute(
        select(Memory).filter(
            Memory.user_id == current_user.id,
            Memory.key == memory.key
        )
    )
    db_memory = result.scalars().first()
    
    if db_memory:
        # Update existing memory
        db_memory.value = memory.value
        await db.commit()
        await db.refresh(db_memory)
        return db_memory
    else:
        # Create new memory
        db_memory = Memory(
            user_id=current_user.id,
            key=memory.key,
            value=memory.value
        )
        db.add(db_memory)
        await db.commit()
        await db.refresh(db_memory)
        return db_memory


@router.post("/batch", response_model=List[MemorySchema])
async def create_memories_batch(
    memory_batch: MemoryBatch,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create or update multiple memories for the current user"""
    results = []
    
    for memory in memory_batch.memories:
        # Check if memory with this key already exists
        result = await db.execute(
            select(Memory).filter(
                Memory.user_id == current_user.id,
                Memory.key == memory.key
            )
        )
        db_memory = result.scalars().first()
        
        if db_memory:
            # Update existing memory
            db_memory.value = memory.value
            await db.commit()
            await db.refresh(db_memory)
            results.append(db_memory)
        else:
            # Create new memory
            db_memory = Memory(
                user_id=current_user.id,
                key=memory.key,
                value=memory.value
            )
            db.add(db_memory)
            await db.commit()
            await db.refresh(db_memory)
            results.append(db_memory)
    
    return results


@router.get("/{key}", response_model=MemorySchema)
async def get_memory(
    key: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a memory by key for the current user"""
    result = await db.execute(
        select(Memory).filter(
            Memory.user_id == current_user.id,
            Memory.key == key
        )
    )
    db_memory = result.scalars().first()
    
    if db_memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with key {key} not found"
        )
        
    return db_memory


@router.get("/", response_model=List[MemorySchema])
async def get_all_memories(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get all memories for the current user"""
    result = await db.execute(
        select(Memory).filter(Memory.user_id == current_user.id)
    )
    memories = result.scalars().all()
    return memories


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    key: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a memory by key"""
    result = await db.execute(
        select(Memory).filter(
            Memory.user_id == current_user.id,
            Memory.key == key
        )
    )
    db_memory = result.scalars().first()
    
    if db_memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory with key {key} not found"
        )
        
    await db.delete(db_memory)
    await db.commit()
    
    return None

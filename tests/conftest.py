"""
Shared test fixtures for Local Context Engine tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_context_engine.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    EngineConfig,
    MetadataStoreConfig,
    PIIMaskingConfig,
    SecurityConfig,
    VectorStoreConfig,
)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def minimal_config(tmp_path: Path) -> EngineConfig:
    """Minimal engine config pointing to temporary paths."""
    return EngineConfig(
        embedding=EmbeddingConfig(
            model="BAAI/bge-small-en-v1.5",
            device="cpu",
            batch_size=4,
            max_seq_length=128,
        ),
        vector_store=VectorStoreConfig(
            backend="faiss",
            dimension=384,
            index_type="flat",
            storage_path=tmp_path / "vectors",
        ),
        metadata_store=MetadataStoreConfig(
            db_path=tmp_path / "test.db",
        ),
        chunking=ChunkingConfig(
            min_tokens=10,
            target_tokens=100,
            max_tokens=300,
            overlap_tokens=10,
        ),
        security=SecurityConfig(
            enable_pii_masking=True,
            never_index_patterns=[],
        ),
        pii_masking=PIIMaskingConfig(),
    )


@pytest.fixture
def sample_php_source() -> str:
    return '''\
<?php

namespace App\\Http\\Controllers;

use App\\Services\\UserService;
use App\\Http\\Requests\\StoreUserRequest;

/**
 * Handles user management endpoints.
 */
class UserController extends Controller
{
    public function __construct(
        private readonly UserService $userService
    ) {}

    /**
     * List all users.
     */
    public function index()
    {
        return $this->userService->getAllUsers();
    }

    public function store(StoreUserRequest $request)
    {
        return $this->userService->createUser($request->validated());
    }
}
'''


@pytest.fixture
def sample_typescript_source() -> str:
    return '''\
import { useQuery, useMutation } from "@tanstack/react-query";
import { fetchUsers, createUser } from "../api/users";

interface User {
    id: number;
    name: string;
    email: string;
}

export function useUsers() {
    return useQuery({
        queryKey: ["users"],
        queryFn: fetchUsers,
    });
}

export function useCreateUser() {
    return useMutation({
        mutationFn: createUser,
    });
}

export function UserList() {
    const { data: users, isLoading } = useUsers();

    if (isLoading) return <div>Loading...</div>;

    return (
        <ul>
            {users?.map(user => (
                <li key={user.id}>{user.name}</li>
            ))}
        </ul>
    );
}
'''


@pytest.fixture
def sample_python_source() -> str:
    return '''\
"""User service module."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    id: int
    name: str
    email: str


class UserRepository:
    """Data access layer for users."""

    def find_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by their ID."""
        pass

    def find_all(self) -> list[User]:
        """Return all users."""
        return []


class UserService:
    """Business logic for user management."""

    def __init__(self, repository: UserRepository) -> None:
        self._repository = repository

    def get_user(self, user_id: int) -> Optional[User]:
        return self._repository.find_by_id(user_id)
'''

"""Small stdlib-only password/token primitives; no config, DB or model imports."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets


def normalize_username(username: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username):
        raise ValueError("Username must be 3-64 ASCII letters, digits, dot, dash or underscore")
    return username.lower()


def hash_password(password: str) -> str:
    if not 12 <= len(password) <= 128:
        raise ValueError("Password must contain 12-128 characters")
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(
        password.encode(), salt=salt, n=32768, r=8, p=3, maxmem=128 * 1024 * 1024, dklen=32
    )
    return "scrypt-32768-8-3$" + salt.hex() + "$" + key.hex()


def verify_password(password: str, encoded: str) -> bool:
    if len(password) > 128:
        return False
    try:
        algorithm, salt, expected = encoded.split("$")
        if algorithm != "scrypt-32768-8-3" or len(salt) != 32 or len(expected) != 64:
            return False
        actual = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt),
            n=32768,
            r=8,
            p=3,
            maxmem=128 * 1024 * 1024,
            dklen=32,
        )
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

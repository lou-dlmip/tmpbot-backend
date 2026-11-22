# Postgres access for the backend.
#
# Like NINOBOT's db.py (origin/main), each operation opens a short-lived
# synchronous psycopg connection and closes it; the calls run in a worker
# thread (asyncio.to_thread) so the event loop keeps streaming other
# visitors' answers meanwhile. Supabase's pooler sits in front, so the
# connection cost is a TCP handshake to it, not a new Postgres backend.
#
# `as_app_role` makes a transaction act as the visitor's database role
# (anon, or authenticated with the JWT claims), the way PostgREST does,
# so row level security on corpus_chunks backs up the backend's own
# access-tier filter (ADR-0007, Phase 4 decision 1).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Short-lived Postgres connections and role switching."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg

from .identity import Identity

log = logging.getLogger("tmpbot.db")

CONNECT_TIMEOUT_S = 10


class Database:
    """Opens connections to one Postgres database.

    Args:
        url: Connection URL (``DATABASE_URL``).
    """

    def __init__(self, url: str) -> None:
        """Create the helper.

        Args:
            url: Connection URL.
        """
        self.url = url

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        """Open a connection; commit on success, roll back on error.

        Yields:
            psycopg.Connection: The open connection.
        """
        conn = psycopg.connect(self.url, connect_timeout=CONNECT_TIMEOUT_S)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def jwt_claims(identity: Identity) -> dict[str, Any]:
    """Build the ``request.jwt.claims`` an app role would carry.

    Args:
        identity: The caller.

    Returns:
        dict[str, Any]: Claims; anonymous callers get only the role.
    """
    if not identity.signed_in:
        return {"role": "anon"}
    claims: dict[str, Any] = {"role": "authenticated",
                              "sub": identity.user_id}
    if identity.plan == "access":
        claims["plan"] = "access"
    return claims


def as_app_role(conn: psycopg.Connection, identity: Identity) -> None:
    """Make the current transaction act as the caller's app role.

    ``SET LOCAL`` and ``set_config(..., true)`` last until the end of
    the transaction, so the next transaction on the connection is back
    to the login role.

    Args:
        conn: Open connection, inside a transaction.
        identity: The caller.
    """
    role = "authenticated" if identity.signed_in else "anon"
    conn.execute(f"SET LOCAL ROLE {role}")
    conn.execute(
        "SELECT set_config('request.jwt.claims', %s, true)",
        (json.dumps(jwt_claims(identity)),),
    )

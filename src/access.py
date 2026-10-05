"""Who is allowed to see a chunk.

Every chunk carries a tenant_id and a list of allowed_roles. "all" in that
list means any role in the tenant. Chunks stored before this existed have
neither field, so they are read as the defaults (tenant "default", roles
["all"]) — that's what they effectively were before.

There is no real login here. UserContext is whatever the caller says it is
(the Streamlit sidebar, a CLI flag, a test).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TENANT = "default"
ALL_ROLES = "all"
DEFAULT_ROLES = [ALL_ROLES]


@dataclass(frozen=True)
class UserContext:
    tenant_id: str
    role: str
    user: str = "anonymous"


def normalize_roles(roles) -> list[str]:
    if roles is None:
        return list(DEFAULT_ROLES)
    if isinstance(roles, str):
        roles = roles.split(",")
    out = sorted({r.strip().lower() for r in roles if r and r.strip()})
    # an empty list would mean "nobody" to us but "missing field" to the
    # qdrant is_empty check used for old chunks, so refuse it outright
    if not out:
        raise ValueError("allowed_roles must name at least one role (or 'all')")
    return out


def chunk_tenant(payload: dict) -> str:
    return payload.get("tenant_id") or DEFAULT_TENANT


def chunk_roles(payload: dict) -> list[str]:
    return payload.get("allowed_roles") or list(DEFAULT_ROLES)


def can_see(payload: dict, user: UserContext | None) -> bool:
    # no user context = the old, unfiltered behaviour (CLI / admin use)
    if user is None:
        return True
    if chunk_tenant(payload) != user.tenant_id:
        return False
    roles = chunk_roles(payload)
    return ALL_ROLES in roles or user.role.lower() in roles

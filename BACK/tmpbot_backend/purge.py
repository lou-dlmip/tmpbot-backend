# The daily data clean-up (PLAN M5.8), run as a Render cron job.
#
# Deletes what TMPBOT has promised not to keep, by calling
# public.tmpbot_purge_expired() (migration 0010) once:
#
#   * anonymous visitors and their unclaimed conversations, 30 days after
#     the visitor was last seen (Lou, 2026-11-15);
#   * rate-limit counters, 14 days after their window started;
#   * contact submissions, TMP Access applications and usage_log only when
#     a period is configured. Until Yumiko and the lawyer answer, the
#     default is "keep" (proposed: 2 years, 2 years, 1 year).
#
# Every period is an environment variable (INFR/env_inventory.md), so a
# change needs no code. The job is idempotent: a second run the same day
# deletes nothing. It exits non-zero on any failure, so Render marks the
# run failed and emails the workspace.
#
# Usage (Render cron job, from the repository root):
#     cd BACK && python -m tmpbot_backend.purge
#     cd BACK && python -m tmpbot_backend.purge --dry-run
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Daily retention purge for the TMPBOT database."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import psycopg

from .logging_setup import configure_logging

log = logging.getLogger("tmpbot.purge")

CONNECT_TIMEOUT_S = 10
# Bound the run so a lock never leaves a cron job hanging for hours.
STATEMENT_TIMEOUT_MS = 120_000

KEEP = "keep"

# Variable names and defaults. The first two follow Lou's decision of
# 2026-11-15 and cannot be "keep"; the other three wait for the lawyer.
VISITOR_DAYS_ENV = "TMPBOT_RETENTION_VISITOR_DAYS"
RATE_LIMIT_DAYS_ENV = "TMPBOT_RETENTION_RATE_LIMIT_DAYS"
CONTACT_DAYS_ENV = "TMPBOT_RETENTION_CONTACT_DAYS"
APPLICATION_DAYS_ENV = "TMPBOT_RETENTION_APPLICATION_DAYS"
USAGE_LOG_DAYS_ENV = "TMPBOT_RETENTION_USAGE_LOG_DAYS"

DEFAULT_VISITOR_DAYS = 30
DEFAULT_RATE_LIMIT_DAYS = 14

# Longest period accepted (about 20 years): a typo such as 7300000 is a
# mistake, not a policy.
MAX_DAYS = 7300

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2


class ConfigError(ValueError):
    """A retention variable holds a value the job cannot use."""


@dataclass(frozen=True)
class RetentionPolicy:
    """How long each kind of record is kept, in days.

    Attributes:
        visitor_days: Anonymous visitors and unclaimed conversations,
            counted from the visitor's last visit.
        rate_limit_days: Rate-limit counters, from the window start.
        contact_days: Contact submissions; None keeps them.
        application_days: TMP Access applications; None keeps them.
        usage_log_days: usage_log rows; None keeps them.
    """

    visitor_days: int = DEFAULT_VISITOR_DAYS
    rate_limit_days: int = DEFAULT_RATE_LIMIT_DAYS
    contact_days: Optional[int] = None
    application_days: Optional[int] = None
    usage_log_days: Optional[int] = None


def parse_days(name: str, raw: Optional[str], default: Optional[int],
               allow_keep: bool) -> Optional[int]:
    """Read one retention period.

    Args:
        name: Variable name, for error messages.
        raw: The variable's value, or None when unset.
        default: Value when unset or blank (None means keep).
        allow_keep: Whether "keep" is accepted.

    Returns:
        Optional[int]: Days, or None for keep.

    Raises:
        ConfigError: The value is not a whole number from 1 to MAX_DAYS,
            or is "keep" where keeping is not allowed.
    """
    value = (raw or "").strip().lower()
    if not value:
        return default
    if value == KEEP:
        if not allow_keep:
            raise ConfigError(f"{name} cannot be '{KEEP}'")
        return None
    try:
        days = int(value)
    except ValueError:
        raise ConfigError(
            f"{name} must be a number of days or '{KEEP}', got {raw!r}"
        ) from None
    if not 1 <= days <= MAX_DAYS:
        raise ConfigError(f"{name} must be from 1 to {MAX_DAYS}, got {days}")
    return days


def policy_from_env(env: Mapping[str, str]) -> RetentionPolicy:
    """Build the retention policy from environment variables.

    Args:
        env: The environment (os.environ in production).

    Returns:
        RetentionPolicy: The periods to apply.

    Raises:
        ConfigError: When any variable is invalid.
    """
    return RetentionPolicy(
        visitor_days=parse_days(VISITOR_DAYS_ENV, env.get(VISITOR_DAYS_ENV),
                                DEFAULT_VISITOR_DAYS, allow_keep=False),
        rate_limit_days=parse_days(
            RATE_LIMIT_DAYS_ENV, env.get(RATE_LIMIT_DAYS_ENV),
            DEFAULT_RATE_LIMIT_DAYS, allow_keep=False),
        contact_days=parse_days(CONTACT_DAYS_ENV, env.get(CONTACT_DAYS_ENV),
                                None, allow_keep=True),
        application_days=parse_days(
            APPLICATION_DAYS_ENV, env.get(APPLICATION_DAYS_ENV), None,
            allow_keep=True),
        usage_log_days=parse_days(
            USAGE_LOG_DAYS_ENV, env.get(USAGE_LOG_DAYS_ENV), None,
            allow_keep=True),
    )


@dataclass(frozen=True)
class PurgeResult:
    """One line of the purge report.

    Attributes:
        target: What was purged, e.g. "anonymous_visitors".
        cutoff: ISO timestamp; older rows were removed.
        affected: Rows removed (or that would be, on a dry run).
    """

    target: str
    cutoff: str
    affected: int


def run_purge(conn: psycopg.Connection, policy: RetentionPolicy,
              dry_run: bool = False) -> list[PurgeResult]:
    """Call the purge function in one transaction.

    The caller commits. On a dry run the function only counts.

    Args:
        conn: An open connection (autocommit off).
        policy: Retention periods.
        dry_run: Count without deleting.

    Returns:
        list[PurgeResult]: One entry per target, in purge order.

    Raises:
        psycopg.Error: When the database refuses or fails.
    """
    conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
    rows = conn.execute(
        "SELECT target, cutoff, affected "
        "FROM public.tmpbot_purge_expired(%s, %s, %s, %s, %s, %s)",
        (policy.visitor_days, policy.rate_limit_days, policy.contact_days,
         policy.application_days, policy.usage_log_days, dry_run),
    ).fetchall()
    return [PurgeResult(target=r[0], cutoff=r[1].isoformat(),
                        affected=int(r[2])) for r in rows]


def purge_database(url: str, policy: RetentionPolicy,
                   dry_run: bool = False) -> list[PurgeResult]:
    """Connect, purge, commit (or roll back on a dry run), disconnect.

    Args:
        url: Postgres connection URL.
        policy: Retention periods.
        dry_run: Count without deleting; nothing is committed.

    Returns:
        list[PurgeResult]: The report.

    Raises:
        psycopg.Error: On a connection or database failure.
    """
    with psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S) as conn:
        try:
            results = run_purge(conn, policy, dry_run=dry_run)
        except Exception:
            conn.rollback()
            raise
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    return results


def main(argv: Optional[Sequence[str]] = None,
         env: Optional[Mapping[str, str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments; sys.argv[1:] when None.
        env: Environment; os.environ when None.

    Returns:
        int: 0 on success, 1 when the database step failed, 2 on a
        configuration error.
    """
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(
        prog="python -m tmpbot_backend.purge",
        description="Delete expired visitors, conversations and counters.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="count what would be deleted; delete nothing")
    args = parser.parse_args(argv)
    configure_logging(env.get("TMPBOT_LOG_LEVEL", "INFO"))

    url = (env.get("DATABASE_URL") or "").strip()
    if not url:
        log.error("purge: DATABASE_URL is not set")
        return EXIT_CONFIG
    try:
        policy = policy_from_env(env)
    except ConfigError as exc:
        log.error("purge: %s", exc)
        return EXIT_CONFIG

    log.info("purge: starting", extra={
        "dry_run": args.dry_run,
        "policy": {k: (v if v is not None else KEEP)
                   for k, v in vars(policy).items()},
    })
    started = time.monotonic()
    try:
        results = purge_database(url, policy, dry_run=args.dry_run)
    except psycopg.Error as exc:
        log.error("purge: database step failed: %s", exc)
        return EXIT_FAILED
    for result in results:
        log.info("purge: %s", result.target, extra={
            "target": result.target, "cutoff": result.cutoff,
            "affected": result.affected, "dry_run": args.dry_run,
        })
    log.info("purge: done", extra={
        "dry_run": args.dry_run,
        "elapsed_s": round(time.monotonic() - started, 3),
    })
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

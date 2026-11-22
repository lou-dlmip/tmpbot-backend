# The daily data clean-up (M5.8): configuration, the command line, and
# tmpbot_purge_expired() (migrations 0010 and 0012) on a real Postgres.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Retention purge: policy parsing, exit codes and database effects."""

from __future__ import annotations

import uuid

import psycopg
import pytest

from tmpbot_backend import purge
from tmpbot_backend.purge import (
    ConfigError,
    RetentionPolicy,
    parse_days,
    policy_from_env,
)

HASH = "a" * 64


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch):
    """Keep main() from replacing pytest's log handlers with stdout JSON."""
    monkeypatch.setattr(purge, "configure_logging", lambda level: None)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

def test_defaults_follow_lous_decision():
    """Unset variables give 30 and 14 days and keep the rest."""
    policy = policy_from_env({})
    assert policy == RetentionPolicy(visitor_days=30, rate_limit_days=14,
                                     contact_days=None,
                                     application_days=None,
                                     usage_log_days=None)


def test_proposed_lawyer_periods_parse():
    """The proposed two years and one year are accepted as day counts."""
    policy = policy_from_env({
        "TMPBOT_RETENTION_CONTACT_DAYS": "730",
        "TMPBOT_RETENTION_APPLICATION_DAYS": " 730 ",
        "TMPBOT_RETENTION_USAGE_LOG_DAYS": "365",
    })
    assert (policy.contact_days, policy.application_days,
            policy.usage_log_days) == (730, 730, 365)


def test_problem_report_period_parses_and_defaults_to_keep():
    """The new variable follows the same rules as the contact period."""
    assert policy_from_env({}).problem_report_days is None
    env = {"TMPBOT_RETENTION_PROBLEM_REPORT_DAYS": " 1460 "}
    assert policy_from_env(env).problem_report_days == 1460
    with pytest.raises(ConfigError):
        policy_from_env({"TMPBOT_RETENTION_PROBLEM_REPORT_DAYS": "0"})


@pytest.mark.parametrize("raw", ["keep", "KEEP", " Keep ", "", None])
def test_keep_and_blank_mean_keep_for_undecided_tables(raw):
    """'keep', blank and unset all keep contact submissions."""
    assert parse_days("X", raw, None, allow_keep=True) is None


def test_keep_is_refused_for_visitors_and_rate_limits():
    """Visitors and counters always have a period."""
    with pytest.raises(ConfigError):
        policy_from_env({"TMPBOT_RETENTION_VISITOR_DAYS": "keep"})
    with pytest.raises(ConfigError):
        policy_from_env({"TMPBOT_RETENTION_RATE_LIMIT_DAYS": "keep"})


@pytest.mark.parametrize("raw", ["0", "-3", "30d", "1.5", "7301", "two"])
def test_bad_periods_are_refused(raw):
    """Anything but a whole number from 1 to 7300 is an error."""
    with pytest.raises(ConfigError):
        parse_days("X", raw, 30, allow_keep=True)


def test_main_without_database_url_is_a_config_error():
    """No DATABASE_URL: exit 2 before touching anything."""
    assert purge.main([], env={}) == purge.EXIT_CONFIG


def test_main_with_bad_policy_is_a_config_error(monkeypatch):
    """A bad period: exit 2, and the database is never called."""
    called = []
    monkeypatch.setattr(purge, "purge_database",
                        lambda *a, **k: called.append(a) or [])
    env = {"DATABASE_URL": "postgresql://x/y",
           "TMPBOT_RETENTION_VISITOR_DAYS": "0"}
    assert purge.main([], env=env) == purge.EXIT_CONFIG
    assert called == []


def test_main_reports_database_failure(monkeypatch):
    """A database error exits 1 so Render marks the run failed."""
    def boom(*_args, **_kwargs):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(purge, "purge_database", boom)
    env = {"DATABASE_URL": "postgresql://x/y"}
    assert purge.main([], env=env) == purge.EXIT_FAILED


def test_main_passes_policy_and_dry_run(monkeypatch):
    """The command line hands the parsed policy and --dry-run on."""
    seen = {}

    def fake(url, policy, dry_run=False):
        seen.update(url=url, policy=policy, dry_run=dry_run)
        return [purge.PurgeResult("anonymous_visitors", "2026-01-01", 3)]

    monkeypatch.setattr(purge, "purge_database", fake)
    env = {"DATABASE_URL": "postgresql://x/y",
           "TMPBOT_RETENTION_USAGE_LOG_DAYS": "365"}
    assert purge.main(["--dry-run"], env=env) == purge.EXIT_OK
    assert seen["dry_run"] is True
    assert seen["policy"].usage_log_days == 365
    assert seen["policy"].visitor_days == 30


# ----------------------------------------------------------------------
# Database (migration 0010)
# ----------------------------------------------------------------------

TABLES = ("messages", "conversations", "anonymous_visitors",
          "request_rate_limits", "contact_submissions",
          "access_applications", "usage_log", "problem_reports")


@pytest.fixture
def db(migrated_db):
    """The migrated test database, emptied of purge targets.

    Yields:
        str: Superuser URL.
    """
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        conn.execute("TRUNCATE " + ", ".join(
            f"public.{t}" for t in TABLES) + " CASCADE")
    yield migrated_db


def _visitor(conn, days_ago, claimed_by=None):
    """Insert a visitor last seen `days_ago` days ago.

    Returns:
        str: The visitor id.
    """
    return str(conn.execute(
        "INSERT INTO public.anonymous_visitors "
        "(ip_hash, last_seen_at, first_seen_at, claimed_by) "
        "VALUES (%(h)s, now() - make_interval(days => %(d)s), "
        "        now() - make_interval(days => %(d)s), %(u)s) "
        "RETURNING visitor_id",
        {"h": HASH, "d": days_ago, "u": claimed_by},
    ).fetchone()[0])


def _conversation(conn, visitor_id=None, user_id=None, messages=2):
    """Insert a conversation with some messages.

    Returns:
        str: The conversation id.
    """
    cid = str(conn.execute(
        "INSERT INTO public.conversations (visitor_id, user_id, language) "
        "VALUES (%(v)s, %(u)s, 'en') RETURNING conversation_id",
        {"v": visitor_id, "u": user_id},
    ).fetchone()[0])
    for seq in range(messages):
        conn.execute(
            "INSERT INTO public.messages (conversation_id, seq, role, "
            "content) VALUES (%(c)s, %(s)s, %(r)s, 'hello')",
            {"c": cid, "s": seq, "r": "user" if seq % 2 == 0 else
             "assistant"},
        )
    return cid


def _user(conn):
    """Insert an auth.users row.

    Returns:
        str: The user id.
    """
    return str(conn.execute(
        "INSERT INTO auth.users (email) VALUES (%s) RETURNING id",
        (f"{uuid.uuid4().hex[:8]}@example.com",),
    ).fetchone()[0])


def _count(conn, table):
    """Rows in a table.

    Returns:
        int: Row count.
    """
    return conn.execute(f"SELECT count(*) FROM public.{table}").fetchone()[0]


def _report(results):
    """Turn purge results into {target: affected}.

    Returns:
        dict[str, int]: Affected rows by target.
    """
    return {r.target: r.affected for r in results}


def test_old_visitor_and_unclaimed_chat_go_recent_and_claimed_stay(db):
    """30 days after last seen, a visitor and its chats are deleted.

    A recent visitor stays, and so does a conversation claimed at
    sign-in, even when its old visitor row goes.
    """
    with psycopg.connect(db, autocommit=True) as conn:
        old = _visitor(conn, 31)
        recent = _visitor(conn, 29)
        user = _user(conn)
        claimed_visitor = _visitor(conn, 40, claimed_by=user)
        _conversation(conn, visitor_id=old, messages=3)
        _conversation(conn, visitor_id=recent, messages=2)
        claimed_chat = _conversation(conn, user_id=user, messages=2)

    results = purge.purge_database(db, RetentionPolicy())
    report = _report(results)
    assert report["anonymous_visitors"] == 2   # old + claimed_visitor
    assert report["unclaimed_conversations"] == 1
    assert report["messages_of_unclaimed_conversations"] == 3

    with psycopg.connect(db) as conn:
        ids = {r[0] for r in conn.execute(
            "SELECT visitor_id::TEXT FROM public.anonymous_visitors")}
        assert ids == {recent}
        assert old not in ids and claimed_visitor not in ids
        chats = {r[0] for r in conn.execute(
            "SELECT conversation_id::TEXT FROM public.conversations")}
        assert claimed_chat in chats
        assert _count(conn, "conversations") == 2
        assert _count(conn, "messages") == 4


def test_rate_limit_rows_go_after_fourteen_days(db):
    """Counters whose window started over 14 days ago are deleted."""
    with psycopg.connect(db, autocommit=True) as conn:
        for scope, days in (("ip", 15), ("user_daily", 14.5),
                            ("visitor", 13), ("auth_ip", 0)):
            conn.execute(
                "INSERT INTO public.request_rate_limits "
                "(scope, key, window_start_at) VALUES "
                "(%s, 'k', now() - make_interval(secs => %s))",
                (scope, days * 86400),
            )
    report = _report(purge.purge_database(db, RetentionPolicy()))
    assert report["request_rate_limits"] == 2
    with psycopg.connect(db) as conn:
        left = sorted(r[0] for r in conn.execute(
            "SELECT scope FROM public.request_rate_limits"))
    assert left == ["auth_ip", "visitor"]


def _old_records(conn, days_ago):
    """Insert one contact message, application, usage and report row."""
    conn.execute(
        "INSERT INTO public.contact_submissions "
        "(created_at, name, email, language) VALUES "
        "(now() - make_interval(days => %s), 'A', 'a@example.com', 'en')",
        (days_ago,))
    conn.execute(
        "INSERT INTO public.access_applications "
        "(created_at, name, email, country, plan_interest, language) "
        "VALUES (now() - make_interval(days => %s), 'A', 'a@example.com', "
        "'JP', 'yearly', 'en')",
        (days_ago,))
    conn.execute(
        "INSERT INTO public.usage_log (created_at, model, plan) VALUES "
        "(now() - make_interval(days => %s), 'm', 'anonymous')",
        (days_ago,))
    conn.execute(
        "INSERT INTO public.problem_reports "
        "(created_at, category, language) VALUES "
        "(now() - make_interval(days => %s), 'other', 'en')",
        (days_ago,))


def test_undecided_tables_are_kept_by_default(db):
    """Submissions, applications and usage rows survive the default."""
    with psycopg.connect(db, autocommit=True) as conn:
        _old_records(conn, 4000)
    report = _report(purge.purge_database(db, RetentionPolicy()))
    assert "contact_submissions" not in report
    assert "access_applications" not in report
    assert "usage_log" not in report
    assert "problem_reports" not in report
    with psycopg.connect(db) as conn:
        for table in ("contact_submissions", "access_applications",
                      "usage_log", "problem_reports"):
            assert _count(conn, table) == 1


def test_configured_periods_purge_undecided_tables(db):
    """With the proposed periods, only older rows go."""
    with psycopg.connect(db, autocommit=True) as conn:
        _old_records(conn, 800)   # older than two years
        _old_records(conn, 400)   # between one and two years
        _old_records(conn, 10)
    policy = RetentionPolicy(contact_days=730, application_days=730,
                             usage_log_days=365, problem_report_days=730)
    report = _report(purge.purge_database(db, policy))
    assert report["problem_reports"] == 1
    assert report["contact_submissions"] == 1
    assert report["access_applications"] == 1
    assert report["usage_log"] == 2
    with psycopg.connect(db) as conn:
        assert _count(conn, "contact_submissions") == 2
        assert _count(conn, "access_applications") == 2
        assert _count(conn, "usage_log") == 1
        assert _count(conn, "problem_reports") == 2


def test_problem_reports_purge_on_their_own_period(db):
    """With 1460 days, a 1500-day-old report goes and a 1400-day one stays."""
    with psycopg.connect(db, autocommit=True) as conn:
        _old_records(conn, 1500)
        _old_records(conn, 1400)
    policy = RetentionPolicy(problem_report_days=1460)
    report = _report(purge.purge_database(db, policy))
    assert report["problem_reports"] == 1
    assert "contact_submissions" not in report
    with psycopg.connect(db) as conn:
        assert _count(conn, "problem_reports") == 1
        assert _count(conn, "contact_submissions") == 2


def test_dry_run_counts_and_deletes_nothing(db):
    """--dry-run reports the same numbers and leaves every row."""
    with psycopg.connect(db, autocommit=True) as conn:
        old = _visitor(conn, 60)
        _conversation(conn, visitor_id=old, messages=2)
        _old_records(conn, 800)
    policy = RetentionPolicy(contact_days=730, problem_report_days=730)
    report = _report(purge.purge_database(db, policy, dry_run=True))
    assert report["problem_reports"] == 1
    assert report["anonymous_visitors"] == 1
    assert report["unclaimed_conversations"] == 1
    assert report["contact_submissions"] == 1
    with psycopg.connect(db) as conn:
        assert _count(conn, "anonymous_visitors") == 1
        assert _count(conn, "conversations") == 1
        assert _count(conn, "messages") == 2
        assert _count(conn, "contact_submissions") == 1
        assert _count(conn, "problem_reports") == 1


def test_second_run_deletes_nothing(db):
    """The purge is idempotent."""
    with psycopg.connect(db, autocommit=True) as conn:
        old = _visitor(conn, 31)
        _conversation(conn, visitor_id=old)
    first = _report(purge.purge_database(db, RetentionPolicy()))
    second = _report(purge.purge_database(db, RetentionPolicy()))
    assert first["anonymous_visitors"] == 1
    assert all(v == 0 for v in second.values())


def test_function_refuses_bad_periods(db):
    """Zero, negative or missing required periods raise 22023."""
    with psycopg.connect(db) as conn:
        for args in ((0, 14), (30, None), (30, 14, -1),
                     (30, 14, None, None, None, False, 0)):
            with pytest.raises(psycopg.errors.InvalidParameterValue):
                with conn.transaction():
                    placeholders = ", ".join(["%s"] * len(args))
                    conn.execute(
                        "SELECT * FROM public.tmpbot_purge_expired("
                        f"{placeholders})", args)


def test_six_argument_call_still_works(db):
    """A caller from before 0012 (six arguments) is not ambiguous."""
    with psycopg.connect(db) as conn:
        rows = conn.execute(
            "SELECT target FROM public.tmpbot_purge_expired("
            "30, 14, NULL, NULL, NULL, true)").fetchall()
        count = conn.execute(
            "SELECT count(*) FROM pg_proc WHERE proname = "
            "'tmpbot_purge_expired'").fetchone()[0]
    assert len(rows) == 4
    assert count == 1


@pytest.mark.parametrize("role", ["anon", "authenticated"])
def test_app_roles_cannot_run_the_purge(db, as_role, role):
    """Only the service role (and the owner) may call the function."""
    with psycopg.connect(db) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with as_role(conn, role):
                conn.execute(
                    "SELECT * FROM public.tmpbot_purge_expired(30, 14)")


def test_service_role_can_run_the_purge(db, as_role):
    """The service role may call it (dry run inside a rolled-back tx)."""
    with psycopg.connect(db) as conn:
        with as_role(conn, "service_role"):
            rows = conn.execute(
                "SELECT target FROM public.tmpbot_purge_expired("
                "30, 14, NULL, NULL, NULL, true)").fetchall()
    assert [r[0] for r in rows] == [
        "messages_of_unclaimed_conversations", "unclaimed_conversations",
        "anonymous_visitors", "request_rate_limits"]


def test_main_end_to_end_on_postgres(db):
    """The command line purges a real database and exits 0."""
    with psycopg.connect(db, autocommit=True) as conn:
        _visitor(conn, 45)
    assert purge.main([], env={"DATABASE_URL": db}) == purge.EXIT_OK
    with psycopg.connect(db) as conn:
        assert _count(conn, "anonymous_visitors") == 0

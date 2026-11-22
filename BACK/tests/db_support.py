# Re-exports the database fixtures of INFR/tests/conftest.py (Phase 4)
# for the backend's database tests: pg_admin_url (pgserver locally, the
# CI Postgres service in CI), fresh_db, migrated_db and as_role. The
# file is loaded by path under a unique module name, because every test
# folder has its own conftest.py.
#
# Without a server the database tests skip, unless
# TMPBOT_REQUIRE_DB_TESTS=1 (CI), in which case they fail.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Database fixtures shared with INFR/tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[2] / "INFR" / "tests"
         / "conftest.py")
_SPEC = importlib.util.spec_from_file_location("tmpbot_infr_db_fixtures",
                                               _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

pg_admin_url = _MODULE.pg_admin_url
fresh_db = _MODULE.fresh_db
migrated_db = _MODULE.migrated_db
as_role = _MODULE.as_role

# Sentry error reporting for the backend.
#
# Sentry starts only when SENTRY_DSN_BACKEND is set, so local runs, CI
# and tests never send anything. NINOBOT pinned sentry-sdk but never
# called init (BOOTSTRAP section 1); TMPBOT does from Phase 2.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Optional Sentry initialisation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from . import __version__
from .config import Settings

log = logging.getLogger("tmpbot.observability")


def _default_init(**kwargs: Any) -> None:
    """Call ``sentry_sdk.init``; imported lazily to keep startup light.

    Args:
        **kwargs: Passed straight to ``sentry_sdk.init``.
    """
    import sentry_sdk

    sentry_sdk.init(**kwargs)


def init_sentry(
    settings: Settings,
    init_fn: Optional[Callable[..., None]] = None,
) -> bool:
    """Start Sentry when a DSN is configured.

    Personal data stays out of Sentry (``send_default_pii=False``): chat
    text, IP addresses and cookies are personal data under GDPR and APPI,
    and the privacy page (Phase 6) does not list Sentry as a recipient.

    Args:
        settings: Backend settings; ``settings.sentry_dsn`` decides.
        init_fn: Replacement for ``sentry_sdk.init``, for tests.

    Returns:
        bool: True if Sentry was initialised, False if it stayed off or
        failed to start. A failure is logged, never raised: error
        reporting must not stop the service from starting.
    """
    if not settings.sentry_dsn:
        log.info("Sentry off: SENTRY_DSN_BACKEND not set")
        return False

    init = init_fn or _default_init
    try:
        init(
            dsn=settings.sentry_dsn,
            environment=settings.env,
            release=f"tmpbot-backend@{__version__}",
            send_default_pii=False,
            traces_sample_rate=0.0,
        )
    except Exception:  # noqa: BLE001 - any SDK failure must not block boot
        log.exception("Sentry failed to initialise; continuing without it")
        return False

    log.info("Sentry on", extra={"sentry_env": settings.env})
    return True

# TMPBOT backend package: the FastAPI service behind the asktmp.com chat.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""TMPBOT backend.

Phase 2 ships a hello-world service: a health probe and a canned,
streamed chat endpoint that speaks the wire format in
``DOCS/STND/streaming_wire_format.md``. Retrieval and the real model
call arrive in Phase 5.
"""

__version__ = "0.1.0"

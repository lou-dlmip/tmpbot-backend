# The published-content catalog: which case studies and publications
# exist, so every id the model names can be checked before a card is
# drawn.
#
# Spec section 3b: "The backend checks every id against published
# content ... A made-up id is dropped, never rendered." The catalog is
# read from COPS/ at startup, the same files the site's pages and the
# corpus index are built from (ADR-0006). A document counts as published
# when at least one of its language files has `source_lifecycle:
# active`; drafts never count.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Published case studies and publications, read from COPS/."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

log = logging.getLogger("tmpbot.catalog")


class CatalogError(ValueError):
    """Raised when COPS/ cannot be read."""


@dataclass(frozen=True)
class Publication:
    """A published, subscriber-only publication.

    Attributes:
        publication_id: The shared id of its language files.
        year: Publication year.
        titles: Language code to title, active files only.
    """

    publication_id: str
    year: int
    titles: Mapping[str, str]

    def title_in(self, language: str) -> str:
        """Return the title in a language, else English, else any.

        Args:
            language: Locale code.

        Returns:
            str: A title.
        """
        return (self.titles.get(language) or self.titles.get("en")
                or next(iter(self.titles.values())))


@dataclass(frozen=True)
class Catalog:
    """Ids of published content.

    Attributes:
        case_studies: Case-study id to year.
        publications: Publication id to Publication.
    """

    case_studies: Mapping[str, int] = field(default_factory=dict)
    publications: Mapping[str, Publication] = field(default_factory=dict)

    def is_case_study(self, case_id: Any) -> bool:
        """Whether ``case_id`` names a published case study.

        Args:
            case_id: The id the model sent.

        Returns:
            bool: True when published.
        """
        return isinstance(case_id, str) and case_id in self.case_studies

    def publication(self, publication_id: Any) -> Optional[Publication]:
        """Return a published publication by id.

        Args:
            publication_id: The id the model sent.

        Returns:
            Optional[Publication]: The publication, or None.
        """
        if not isinstance(publication_id, str):
            return None
        return self.publications.get(publication_id)


def read_frontmatter(path: Path) -> dict[str, Any]:
    """Read a Markdown file's YAML frontmatter.

    Args:
        path: The file.

    Returns:
        dict[str, Any]: The frontmatter, or an empty dict when absent.

    Raises:
        CatalogError: When the file cannot be read or the YAML is bad.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read {path}: {exc}") from exc
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError as exc:
        raise CatalogError(f"bad frontmatter in {path}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def load_catalog(cops_dir: Path) -> Catalog:
    """Read the published case studies and publications.

    Args:
        cops_dir: The ``COPS/`` folder.

    Returns:
        Catalog: Published ids. Missing folders give empty sections.

    Raises:
        CatalogError: When a file cannot be read or parsed.
    """
    cases: dict[str, int] = {}
    for path in sorted((cops_dir / "CASE").glob("*.md")):
        meta = read_frontmatter(path)
        if (meta.get("type") == "case_study"
                and meta.get("source_lifecycle") == "active"
                and meta.get("access_tier", "public") == "public"
                and isinstance(meta.get("id"), str)):
            cases[meta["id"]] = int(meta.get("year") or 0)

    titles: dict[str, dict[str, str]] = {}
    years: dict[str, int] = {}
    for path in sorted((cops_dir / "PUBL").glob("*.md")):
        meta = read_frontmatter(path)
        if (meta.get("type") == "publication"
                and meta.get("source_lifecycle") == "active"
                and isinstance(meta.get("id"), str)
                and meta.get("title")):
            pid = meta["id"]
            titles.setdefault(pid, {})[str(meta.get("language"))] = str(
                meta["title"])
            years[pid] = int(meta.get("year") or 0)
    pubs = {
        pid: Publication(pid, years[pid], titles[pid])
        for pid in titles
    }
    log.info("catalog loaded",
             extra={"case_studies": len(cases), "publications": len(pubs)})
    return Catalog(case_studies=cases, publications=pubs)

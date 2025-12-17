# Example:
# * https://portal.mardi4nfdi.de/w/api.php?action=query&list=search&srsearch=joswig&srlimit=5&srnamespace=0&format=json
# * https://portal.mardi4nfdi.de/w/api.php?action=query&pageids=14629161&prop=revisions&rvprop=content&rvslots=main&format=json

import re
import requests
import time
from typing import List, Optional, Tuple
from app.logger import get_logger

from bs4 import BeautifulSoup
from langchain_core.documents import Document

HARD_TIMEOUT_SECONDS = 3.0

MAIN_NAMESPACE = "0"

ENTITY_NAMESPACES = "|".join([
    "0",      # Main / Portal
    "120",    # Item (QIDs)
    "4200",   # Formula
    "4202",   # Person
    "4206",   # Publication
    "4208",   # Software
    "4210",   # Dataset
    "4214",   # Workflow
    "4216",   # Algorithm
    "4218",   # Service
    "4220",   # Theorem
    "4222",   # Research field
    "4224",   # Research problem
    "4226",   # Model
#    "4228",   # Quantity
#    "4230",   # Task
#    "4232",   # Academic discipline
])

logger = get_logger("MardiWikiRetriever")

class MardiWikiRetriever:
    """
    MediaWiki retriever for the MaRDI Portal.

    Implements:
      (2) Extract and attach MaRDI QIDs (e.g. Q6774181) from rendered HTML
      (3) Section-level chunking (split by headings and turn each section into a Document)

    Notes:
      - Uses action=query&list=search for recall
      - Uses action=parse for expanded HTML (template-driven pages)
      - Produces multiple Documents per page (one per section)
    """

    QID_RE = re.compile(r"\bQ\d+\b")

    def __init__(
        self,
        api_url: str,
        top_k: int = 5,
        timeout: int = 10,
        max_section_chars: int = 6000,
        include_intro: bool = True,
    ):
        self.api_url = api_url.rstrip("/")
        self.top_k = top_k
        self.timeout = timeout
        self.max_section_chars = max_section_chars
        self.include_intro = include_intro

    # --------------------------------------------------
    # Public API (LangChain-style)
    # --------------------------------------------------

    def get_relevant_documents(self, query: str) -> List[Document]:
        start = time.monotonic()
        deadline = start + HARD_TIMEOUT_SECONDS

        search_hits = self._search(query)

        out_docs: List[Document] = []

        for hit in search_hits:
            # ---- hard timeout check ----
            if time.monotonic() >= deadline:
                break

            pageid = hit["pageid"]
            title = hit["title"]

            try:
                html = self._fetch_parsed_html(pageid)
            except Exception:
                continue

            # Another timeout check after parse
            if time.monotonic() >= deadline:
                break

            qid = self._extract_mardi_qid(html)
            sections = self._extract_sections(html)

            if not sections:
                whole = self._extract_text(html).strip()
                if whole:
                    sections = [("Page", whole)]

            page_url = self._build_page_url(title)

            for idx, (section_title, section_text) in enumerate(sections):
                if time.monotonic() >= deadline:
                    break

                text = (section_text or "").strip()
                if not text:
                    continue

                if self.max_section_chars and len(text) > self.max_section_chars:
                    text = text[: self.max_section_chars].rstrip() + " …"

                out_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "retriever": "mediawiki",
                            "pageid": pageid,
                            "title": title,
                            "page": title,
                            "url": page_url,
                            "package": "MaRDI Portal",
                            "version": "live",
                            "qid": qid,
                            "section": section_title,
                            "component": f"mediawiki_section:{self._slug(section_title)}",
                            "chunk_index": idx,
                        },
                    )
                )

        elapsed = time.monotonic() - start
        if elapsed >= HARD_TIMEOUT_SECONDS:
            logger.warning(
                "MaRDI Wiki retrieval aborted after %.2fs (hard timeout)",
                elapsed,
            )

        return out_docs

    # --------------------------------------------------
    # MediaWiki API helpers
    # --------------------------------------------------

    def _search(self, query: str) -> List[dict]:
        """
        Query-dependent namespace routing.
        - Lexical / entity queries → portal + KG namespaces
        - Semantic queries → portal pages only
        """

        if _is_lexical_query(query):
            srnamespace = ENTITY_NAMESPACES
        else:
            srnamespace = MAIN_NAMESPACE

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self.top_k,
            "srnamespace": srnamespace,
            "format": "json",
        }

        r = requests.get(self.api_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("query", {}).get("search", [])


    def _fetch_parsed_html(self, pageid: int) -> str:
        params = {
            "action": "parse",
            "pageid": pageid,
            "prop": "text",
            "format": "json",
        }
        r = requests.get(self.api_url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["parse"]["text"]["*"]


    def _build_page_url(self, title: str) -> str:
        # Works for MaRDI portal; keep simple and stable
        return f"https://portal.mardi4nfdi.de/wiki/{title.replace(' ', '_')}"

    # --------------------------------------------------
    # (2) Extract MaRDI QID from rendered HTML
    # --------------------------------------------------

    def _extract_mardi_qid(self, html: str) -> Optional[str]:
        """
        Extracts the first MaRDI QID from links such as /wiki/Item:Q6774181
        or from visible text containing Q-numbers.

        Returns:
            'Q#######' or None
        """
        soup = BeautifulSoup(html, "html.parser")

        # Prefer href patterns that explicitly point to Item:Q...
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"(?:^|/wiki/)(?:Item:)?(Q\d+)", href)
            if m:
                return m.group(1)

        # Fallback: look for QID in link text or surrounding text
        text = soup.get_text(" ", strip=True)
        m2 = self.QID_RE.search(text)
        return m2.group(0) if m2 else None

    # --------------------------------------------------
    # (3) Section-level chunking
    # --------------------------------------------------

    def _extract_sections(self, html: str) -> List[Tuple[str, str]]:
        """
        Splits rendered HTML into sections based on headings (h2/h3/h4),
        returning [(section_title, section_text), ...].

        Includes an optional intro section (text before the first heading).
        """
        soup = BeautifulSoup(html, "html.parser")
        self._cleanup_soup(soup)

        root = soup.find("div", class_="mw-parser-output") or soup

        # Collect intro until first heading
        sections: List[Tuple[str, str]] = []
        if self.include_intro:
            intro_nodes = []
            for child in list(root.children):
                if getattr(child, "name", None) in ("h1", "h2", "h3", "h4"):
                    break
                intro_nodes.append(child)
            intro_text = self._nodes_to_text(intro_nodes)
            intro_text = intro_text.strip()
            if intro_text:
                sections.append(("Intro", intro_text))

        # Chunk by headings (h2/h3/h4)
        current_title: Optional[str] = None
        current_nodes = []

        for node in list(root.children):
            tag = getattr(node, "name", None)
            if tag in ("h2", "h3", "h4"):
                # flush previous
                if current_title is not None:
                    text = self._nodes_to_text(current_nodes).strip()
                    if text:
                        sections.append((current_title, text))
                # start new
                current_title = node.get_text(" ", strip=True) or "Section"
                current_nodes = []
            else:
                if current_title is not None:
                    current_nodes.append(node)

        # flush last
        if current_title is not None:
            text = self._nodes_to_text(current_nodes).strip()
            if text:
                sections.append((current_title, text))

        # If nothing meaningful, return empty to trigger whole-page fallback
        return [(t, s) for (t, s) in sections if s.strip()]

    # --------------------------------------------------
    # HTML cleanup + text extraction utilities
    # --------------------------------------------------

    def _cleanup_soup(self, soup: BeautifulSoup) -> None:
        # Remove irrelevant elements
        for tag in soup(["style", "script"]):
            tag.decompose()

        for el in soup.select(".DeepChat, .wikiChartContainer, .mw-editsection"):
            el.decompose()

        # Optional: remove warning boxes (often boilerplate)
        for table in soup.select(".ambox"):
            table.decompose()

        # Flatten tables to readable lines (in-place)
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [
                    c.get_text(" ", strip=True)
                    for c in tr.find_all(["th", "td"])
                ]
                if cells:
                    rows.append(" | ".join(cells))
            table.replace_with("\n".join(rows))

    def _nodes_to_text(self, nodes) -> str:
        # Convert a list of soup nodes into cleaned text
        tmp = BeautifulSoup("", "html.parser")
        container = tmp.new_tag("div")
        tmp.append(container)

        for n in nodes:
            try:
                container.append(n)
            except Exception:
                # Some nodes cannot be re-parented; ignore
                pass

        text = container.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        self._cleanup_soup(soup)
        text = soup.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    @staticmethod
    def _slug(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
        return s or "section"

def _is_lexical_query(query: str) -> bool:
    q = query.strip()
    return (
        len(q) <= 30
        or q.isupper()
        or any(c.isdigit() for c in q)
    )
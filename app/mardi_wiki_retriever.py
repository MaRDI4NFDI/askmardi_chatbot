import requests
from typing import List, Optional

from bs4 import BeautifulSoup
from langchain_core.documents import Document

class MardiWikiRetriever:
    """
    MediaWiki retriever for the MaRDI Portal.

    - Uses MediaWiki search API
    - Fetches expanded HTML via action=parse
    - Extracts readable text from template-driven pages
    - Returns LangChain Documents with rich metadata
    """

    def __init__(
        self,
        api_url: str,
        top_k: int = 5,
        timeout: int = 10,
    ):
        self.api_url = api_url.rstrip("/")
        self.top_k = top_k
        self.timeout = timeout

    # --------------------------------------------------
    # Public API (LangChain-style)
    # --------------------------------------------------

    def get_relevant_documents(self, query: str) -> List[Document]:
        search_hits = self._search(query)

        docs: List[Document] = []
        for idx, hit in enumerate(search_hits):
            pageid = hit["pageid"]
            title = hit["title"]

            try:
                html = self._fetch_parsed_html(pageid)
                text = self._extract_text(html)
            except Exception:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "retriever": "mediawiki",
                        "pageid": pageid,
                        "title": title,
                        "page": title,
                        "chunk_index": idx,
                        "component": "entity",
                        "package": "MaRDI Portal",
                        "version": "live",
                        "url": f"https://portal.mardi4nfdi.de/wiki/{title.replace(' ', '_')}",
                    },
                )
            )

        return docs

    # --------------------------------------------------
    # MediaWiki API helpers
    # --------------------------------------------------

    def _search(self, query: str) -> List[dict]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self.top_k,
            "srnamespace": 0,
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

    # --------------------------------------------------
    # HTML → clean text extraction (MaRDI-specific)
    # --------------------------------------------------

    def _extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        # Remove irrelevant elements
        for tag in soup(["style", "script"]):
            tag.decompose()

        for div in soup.select(
            ".DeepChat, .wikiChartContainer, .mw-editsection"
        ):
            div.decompose()

        # Remove warning boxes (optional but recommended)
        for table in soup.select(".ambox"):
            table.decompose()

        # Flatten tables into readable rows
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

        text = soup.get_text(separator="\n", strip=True)

        # Normalize whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

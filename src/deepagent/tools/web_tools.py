# src/deepagent/tools/web_tools.py
import re
import urllib.parse

import aiohttp

from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


async def _fetch_html(url: str, timeout: int = 15) -> str | None:
    """Fetch a URL asynchronously and return the decoded HTML text, or None on failure."""
    headers = {"User-Agent": _USER_AGENT}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status != 200:
                    return None
                raw = await resp.read()
                charset = resp.charset or "utf-8"
                return raw.decode(charset, errors="replace")
    except Exception:
        return None


def create_web_tools(registry: ToolRegistry) -> list:
    """Create and register web tools (web_search, web_fetch)."""

    @tool(
        registry=registry,
        description="Search the web using DuckDuckGo (no API key needed). Returns titles, URLs, and snippets for each result.",
        safety_level=SafetyLevel.READONLY,
    )
    async def web_search(query: str, max_results: int = 5) -> dict:
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            html = await _fetch_html(url, timeout=15)
            if html is None:
                return {
                    "success": False,
                    "content": "",
                    "error": "Failed to fetch search results from DuckDuckGo",
                    "metadata": None,
                }

            # Extract result links: <a class="result__a" href="...">Title</a>
            link_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>\s*(.*?)\s*</a>',
                re.IGNORECASE | re.DOTALL,
            )
            links = link_pattern.findall(html)

            # Extract snippets: <a class="result__snippet">...snippet...</a>
            snippet_pattern = re.compile(
                r'<a[^>]*class="result__snippet"[^>]*>\s*(.*?)\s*</a>',
                re.IGNORECASE | re.DOTALL,
            )
            snippets = snippet_pattern.findall(html)

            if not links:
                return {
                    "success": True,
                    "content": f"No results found for '{query}'",
                    "error": None,
                    "metadata": {"result_count": 0},
                }

            results = []
            for i, (href, title) in enumerate(links):
                if i >= max_results:
                    break
                clean_href = href.strip()
                clean_title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = ""
                if i < len(snippets):
                    snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                results.append(
                    f"{i + 1}. {clean_title}\n"
                    f"   URL: {clean_href}\n"
                    f"   {snippet}"
                )

            content = "\n\n".join(results)
            return {
                "success": True,
                "content": content,
                "error": None,
                "metadata": {
                    "result_count": min(len(links), max_results),
                    "total_available": len(links),
                },
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "content": "",
                "error": f"Network error during web search: {e}",
                "metadata": None,
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Error in web_search: {e}",
                "metadata": None,
            }

    @tool(
        registry=registry,
        description="Fetch and extract plain text content from a URL. HTML tags are stripped.",
        safety_level=SafetyLevel.READONLY,
    )
    async def web_fetch(url: str, max_chars: int = 10000) -> dict:
        try:
            html = await _fetch_html(url, timeout=15)
            if html is None:
                return {
                    "success": False,
                    "content": "",
                    "error": f"Failed to fetch URL: {url}",
                    "metadata": None,
                }

            # Strip script and style blocks before removing tags
            text = re.sub(
                r"<(script|style)[^>]*>.*?</\1>",
                "",
                html,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Remove all remaining HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()

            original_len = len(text)
            if len(text) > max_chars:
                text = text[:max_chars]

            return {
                "success": True,
                "content": text,
                "error": None,
                "metadata": {
                    "char_count": len(text),
                    "original_length": original_len,
                    "truncated": original_len > max_chars,
                },
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "content": "",
                "error": f"Network error fetching URL: {e}",
                "metadata": None,
            }
        except ValueError as e:
            return {
                "success": False,
                "content": "",
                "error": f"Invalid URL: {e}",
                "metadata": None,
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Error in web_fetch: {e}",
                "metadata": None,
            }

    return [web_search, web_fetch]

"""Memory data model — matches the Claude Code memory frontmatter format."""

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """A single memory entry stored as a Markdown file with YAML frontmatter."""

    name: str  # kebab-case slug, e.g. "user-role"
    description: str  # one-line summary, for relevance matching
    memory_type: str  # "user" | "feedback" | "project" | "reference"
    content: str  # body (everything after the frontmatter)
    file_path: str = ""  # path to the .md file on disk

    @staticmethod
    def parse_frontmatter(raw: str, file_path: str = "") -> "MemoryEntry | None":
        """Parse a memory .md file with YAML frontmatter.

        Expected format:
        ---
        name: short-slug
        description: one-line summary
        metadata:
          type: user  # or feedback, project, reference
        ---

        Body content follows the second ---.
        """
        if not raw.startswith("---"):
            return None
        end_idx = raw.find("---", 3)
        if end_idx == -1:
            return None
        frontmatter_str = raw[3:end_idx].strip()
        content = raw[end_idx + 3:].strip()

        fields: dict[str, str] = {}
        for line in frontmatter_str.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                # Skip nested keys (metadata.type handled separately)
                key = key.strip()
                if key == "metadata":
                    continue
                fields[key] = value.strip()

        memory_type = "user"
        # Extract metadata.type if present
        for line in frontmatter_str.split("\n"):
            if "type:" in line and ("metadata" in frontmatter_str.split(line)[0] if True else False):
                # Just look for type: after metadata
                stripped = line.strip()
                if stripped.startswith("type:"):
                    # If it's on its own line with indentation, it's metadata.type
                    memory_type = stripped.split(":", 1)[1].strip()
                    break

        # More robust: find metadata block
        import re
        m = re.search(r'metadata:\s*\n\s*type:\s*(\w+)', frontmatter_str)
        if m:
            memory_type = m.group(1)

        name = fields.get("name", "")
        if not name:
            return None

        return MemoryEntry(
            name=name,
            description=fields.get("description", ""),
            memory_type=memory_type,
            content=content,
            file_path=file_path,
        )

    def to_frontmatter(self) -> str:
        """Serialize back to the Markdown + YAML frontmatter format."""
        return (
            f"---\n"
            f"name: {self.name}\n"
            f"description: {self.description}\n"
            f"metadata:\n"
            f"  type: {self.memory_type}\n"
            f"---\n"
            f"\n"
            f"{self.content}\n"
        )

    @property
    def index_line(self) -> str:
        """Render as a single MEMORY.md index line."""
        return f"- [{self.name}]({self.name}.md) — {self.description}"

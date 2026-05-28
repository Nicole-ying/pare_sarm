"""Tool registry for agent file I/O and memory access.

Agents use these tools to read files from the experiment directory,
query the memory system, and validate Python code — all without
hallucinating file contents or memory entries.
"""

from pathlib import Path


class Tool:
    """A callable tool with name, description, and JSON schema for LLM function calling."""

    def __init__(self, name: str, description: str, schema: dict, fn):
        self.name = name
        self.description = description
        self.schema = schema
        self.fn = fn

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class ToolRegistry:
    """Registry of tools that agents can call.

    Provides both direct Python access and OpenAI-compatible function schemas.
    """

    def __init__(self, exp_dir: Path = None, memory_system=None):
        self._tools: dict[str, Tool] = {}
        self.exp_dir = exp_dir
        self.memory = memory_system
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in tools."""
        self.register(
            "read_file",
            "Read the contents of a file within the experiment directory.",
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file's contents. Path must be relative to the experiment directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Relative path to the file to read, e.g. 'round0/task_manifest.md'"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            self._read_file
        )

        self.register(
            "query_memory",
            "Search episodic memory for relevant past rounds and lessons.",
            {
                "type": "function",
                "function": {
                    "name": "query_memory",
                    "description": "Search cross-round memory for relevant lessons and patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, e.g. 'fuel penalty too strong'"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            self._query_memory
        )

        self.register(
            "query_archival",
            "Search archival memory for cross-experiment design principles.",
            {
                "type": "function",
                "function": {
                    "name": "query_archival",
                    "description": "Search cross-experiment archival patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results (default 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            self._query_archival
        )

        self.register(
            "validate_python",
            "Check if a Python code string has valid syntax.",
            {
                "type": "function",
                "function": {
                    "name": "validate_python",
                    "description": "Check Python code syntax without executing it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python source code to validate"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            self._validate_python
        )

    def register(self, name: str, description: str, schema: dict, fn):
        """Register a new tool."""
        self._tools[name] = Tool(name, description, schema, fn)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self, names: list[str] = None) -> list[dict]:
        """Get OpenAI-compatible function schemas for tools."""
        tools = [self._tools[n] for n in (names or self._tools.keys()) if n in self._tools]
        return [t.schema for t in tools]

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found. Available: {self.list_tools()}"
        try:
            return str(tool(**arguments))
        except Exception as e:
            return f"Error executing {name}: {e}"

    # ── Built-in tool implementations ──────────────────────────────────────

    def _read_file(self, filepath: str) -> str:
        """Read a file, enforcing path safety within experiment directory."""
        if self.exp_dir is None:
            return "Error: no experiment directory configured"
        full = (self.exp_dir / filepath).resolve()
        if not str(full).startswith(str(self.exp_dir.resolve())):
            return f"Error: path traversal denied — '{filepath}' is outside experiment directory"
        if not full.exists():
            return f"Error: file not found: {filepath}"
        content = full.read_text("utf-8")
        if len(content) > 8000:
            return content[:8000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content

    def _query_memory(self, query: str, max_results: int = 5) -> str:
        """Search episodic memory."""
        if self.memory is None:
            return "No memory system configured."
        results = self.memory.episodic.search(query, max_results)
        if not results:
            return f"No results found for: {query}"
        lines = []
        for r in results:
            lines.append(
                f"Round {r.get('round', '?')}: "
                f"score={r.get('health_score', '?')}, "
                f"summary={r.get('summary', 'N/A')[:200]}"
            )
        return "\n".join(lines)

    def _query_archival(self, query: str, max_results: int = 5) -> str:
        """Search archival memory."""
        if self.memory is None:
            return "No memory system configured."
        patterns = self.memory.archival.search(query, max_results)
        if not patterns:
            return f"No archival patterns found for: {query}"
        return "\n".join(f"- {p}" for p in patterns)

    def _validate_python(self, code: str) -> str:
        """Check Python syntax."""
        try:
            compile(code, "<validate>", "exec")
            return "OK: valid Python syntax"
        except SyntaxError as e:
            return f"SyntaxError: {e}"

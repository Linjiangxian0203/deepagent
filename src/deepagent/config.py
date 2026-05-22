# src/deepagent/config.py
import os


class Config:
    def __init__(self, _env: dict[str, str] | None = None):
        env = _env if _env is not None else os.environ
        self.api_key = env.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required. "
                "Set it in your environment or a .env file."
            )
        self.base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = env.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.max_iterations = int(env.get("DEEPSEEK_MAX_ITERATIONS", "50"))
        self.max_tools_per_turn = int(env.get("DEEPSEEK_MAX_TOOLS_PER_TURN", "10"))

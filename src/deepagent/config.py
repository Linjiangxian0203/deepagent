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
        self.model = env.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        # Thinking mode (DeepSeek V4)
        self.thinking_enabled = env.get("DEEPSEEK_THINKING_ENABLED", "1") not in ("0", "false", "False")
        self.reasoning_effort = env.get("DEEPSEEK_REASONING_EFFORT", "max")  # high | max
        # Sampling
        self.max_tokens = int(env.get("DEEPSEEK_MAX_TOKENS", "8192"))
        self.temperature = float(env.get("DEEPSEEK_TEMPERATURE", "1.0"))
        self.top_p = float(env.get("DEEPSEEK_TOP_P", "1.0"))
        # Loop control
        self.max_iterations = int(env.get("DEEPSEEK_MAX_ITERATIONS", "50"))
        self.max_tools_per_turn = int(env.get("DEEPSEEK_MAX_TOOLS_PER_TURN", "10"))

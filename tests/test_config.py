# tests/test_config.py
import pytest
from deepagent.config import Config


def test_config_loads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    cfg = Config()
    assert cfg.api_key == "sk-test-123"


def test_config_missing_api_key_raises():
    """Config raises ValueError when DEEPSEEK_API_KEY is not set."""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        Config(_env={"DOES_NOT_EXIST": "1"})


def test_config_base_url_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.base_url == "https://api.deepseek.com"


def test_config_model_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.model == "deepseek-v4-pro"


def test_config_thinking_enabled_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.thinking_enabled is True


def test_config_thinking_disabled():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_THINKING_ENABLED": "0"})
    assert cfg.thinking_enabled is False


def test_config_reasoning_effort_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.reasoning_effort == "max"


def test_config_max_tokens_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.max_tokens == 8192


def test_config_temperature_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.temperature == 1.0


def test_config_top_p_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.top_p == 1.0


def test_config_custom_model():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MODEL": "deepseek-reasoner"})
    assert cfg.model == "deepseek-reasoner"


def test_config_max_iterations_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.max_iterations == 50


def test_config_max_tools_per_turn_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.max_tools_per_turn == 10


def test_config_custom_base_url():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_BASE_URL": "https://custom.api.com"})
    assert cfg.base_url == "https://custom.api.com"


def test_config_custom_max_iterations():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MAX_ITERATIONS": "100"})
    assert cfg.max_iterations == 100


def test_config_custom_max_tools_per_turn():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MAX_TOOLS_PER_TURN": "25"})
    assert cfg.max_tools_per_turn == 25

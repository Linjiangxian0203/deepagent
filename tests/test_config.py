# tests/test_config.py
import os
import pytest
from deepagent.config import Config


def test_config_loads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    cfg = Config()
    assert cfg.api_key == "sk-test-123"


def test_config_defaults():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        Config(_env={"DOES_NOT_EXIST": "1"})


def test_config_base_url_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.base_url == "https://api.deepseek.com"


def test_config_model_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.model == "deepseek-chat"


def test_config_custom_model():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test", "DEEPSEEK_MODEL": "deepseek-reasoner"})
    assert cfg.model == "deepseek-reasoner"


def test_config_max_iterations_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.max_iterations == 50


def test_config_max_tools_per_turn_default():
    cfg = Config(_env={"DEEPSEEK_API_KEY": "sk-test"})
    assert cfg.max_tools_per_turn == 10

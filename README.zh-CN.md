# Deepagent

[English](README.md) | 中文

一个基于 DeepSeek API 的 CLI 编程助手。在终端中运行，阅读代码库、编辑文件、执行 Shell 命令、派发子任务——全部在多轮 ReAct 循环中完成，流式输出，支持思考模式。

## 特性

- **多轮 ReAct 循环** -- 智能体循环：思考 → 调用工具 → 观察结果 → 继续，直到任务完成或达到迭代上限。
- **12 个内置工具** -- `read_file`、`write_file`、`edit_file`、`grep`、`glob`、`run_shell`、`git_diff`、`git_log`、`git_status`、`web_search`、`web_fetch`、`delegate`（子智能体并发派发）。
- **流式输出** -- LLM 文本实时流到终端。DeepSeek 思考模式内容单独显示并标注字符数。
- **Token 预算管理** -- 跟踪累计 token 用量（1M 上下文窗口，980K 有效上限），接近上限时自动压缩早期消息为摘要。
- **长期记忆** -- 基于文件系统的持久化记忆，使用 Markdown + YAML frontmatter 格式，兼容 Claude Code 记忆格式（`~/.claude/projects/<slug>/memory/`）。
- **安全确认** -- 写入和 Shell 级别工具需要交互式 `y/N` 确认才能执行。文件操作可限定在安全根目录内。
- **子智能体派发** -- `delegate` 工具可并行派发子智能体处理独立任务，默认并发上限 5 个。
- **CLAUDE.md 感知** -- 自动加载 `~/.claude/CLAUDE.md` 和项目级 `CLAUDE.md` 文件，注入到系统提示词中。
- **Windows 原生支持** -- 在 Windows 上使用原生 cmd.exe，路径用反斜杠，命令用 Windows 风格（dir 而非 ls，findstr 而非 grep）。

## 快速开始

### 环境要求

- Python 3.11+
- [DeepSeek API Key](https://platform.deepseek.com/)

### 安装

```bash
git clone <仓库地址> deepagent
cd deepagent
pip install -e .
```

### 运行

```bash
# Windows (cmd / PowerShell)
set DEEPSEEK_API_KEY=sk-你的密钥
deepagent

# Linux / macOS
export DEEPSEEK_API_KEY="sk-你的密钥"
deepagent
```

在 `> ` 提示符后输入任务。输入 `/exit` 或 `Ctrl+D` 退出，`Ctrl+C` 中断当前任务。

## 配置

所有配置通过环境变量控制。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(必填)* | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 端点地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | 模型名称（如 `deepseek-v4-pro`、`deepseek-v4-flash`） |
| `DEEPSEEK_THINKING_ENABLED` | `1` | 启用思考/推理模式。设为 `0` 或 `false` 关闭 |
| `DEEPSEEK_REASONING_EFFORT` | `max` | 推理深度。可选值：`high`、`max` |
| `DEEPSEEK_MAX_TOKENS` | `8192` | 每次 LLM 调用的最大输出 token 数 |
| `DEEPSEEK_TEMPERATURE` | `1.0` | 采样温度 |
| `DEEPSEEK_TOP_P` | `1.0` | 核采样参数 |
| `DEEPSEEK_MAX_ITERATIONS` | `50` | 每次请求的最大 ReAct 循环次数 |
| `DEEPSEEK_MAX_TOOLS_PER_TURN` | `10` | 单轮允许的最大工具调用数 |

## 可用工具

| 工具 | 安全级别 | 说明 |
|---|---|---|
| `read_file` | 只读 | 读取文件，带行号输出，支持偏移和行数限制 |
| `write_file` | 写入 | 创建新文件或覆盖已有文件 |
| `edit_file` | 写入 | 精确替换文件中的字符串，匹配必须唯一 |
| `grep` | 只读 | 用正则表达式搜索文件内容，支持文件过滤 |
| `glob` | 只读 | 用 glob 模式匹配文件（如 `**/*.py` 递归搜索） |
| `run_shell` | Shell | 执行 Shell 命令并返回 stdout/stderr，可设超时 |
| `git_diff` | 只读 | 显示工作区变更，支持 --staged 和路径过滤 |
| `git_log` | 只读 | 显示最近提交历史（`git log --oneline`），可设数量 |
| `git_status` | 只读 | 显示工作区状态（`git status --short`） |
| `web_search` | 只读 | 用 DuckDuckGo 搜索网页，返回标题、链接和摘要 |
| `web_fetch` | 只读 | 获取网页纯文本内容，去除 HTML 标签 |
| `delegate` | 只读 | 派发子智能体自主处理独立子任务，单轮可并行多个 |

**安全级别说明**：`只读` 工具无需确认直接执行。`写入` 和 `Shell` 工具执行前会提示 `y/N` 确认。

## 项目结构

```
deepagent/
  pyproject.toml                  # 项目元数据、依赖、入口点
  src/deepagent/
    __init__.py                   # 版本号
    main.py                       # CLI 入口
    config.py                     # 环境变量配置
    cli/
      __init__.py
      app.py                      # CLI 循环、流式渲染、系统提示词构建
    core/
      __init__.py
      context.py                  # ContextManager：Token 预算和上下文压缩
      events.py                   # 事件数据类（TextDelta、ToolCallEvent 等）
      llm_client.py               # OpenAI 兼容的异步 LLM 客户端
      loop.py                     # AgentLoop：多轮 ReAct 引擎
      sub_agent.py                # SubAgentRunner：并发控制
    tools/
      __init__.py
      protocol.py                 # SafetyLevel 枚举、ToolProtocol
      registry.py                 # ToolRegistry 和 @tool 装饰器
      file_tools.py               # read_file、write_file、edit_file
      search_tools.py             # grep、glob
      shell_tools.py              # run_shell
      git_tools.py                # git_diff、git_log、git_status
      web_tools.py                # web_search、web_fetch
      delegate_tools.py           # delegate（子智能体派发）
    memory/
      __init__.py
      models.py                   # MemoryEntry 数据类（含 frontmatter 解析）
      store.py                    # MemoryStore：Markdown 持久化记忆
```

## 开发

### 环境搭建

```bash
git clone <仓库地址> deepagent
cd deepagent
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -e .
```

### 依赖

所有运行时依赖在 `pyproject.toml` 中声明：

- `openai` >= 1.0.0 -- OpenAI 兼容客户端，连接 DeepSeek API
- `rich` >= 13.0.0 -- 终端格式化和颜色
- `prompt-toolkit` >= 3.0.0 -- 交互式提示符和按键绑定
- `pydantic` >= 2.0.0 -- 数据校验（LLM 客户端使用）

### 运行测试

```bash
# 在项目根目录下运行
python -m pytest tests/
```

### 架构说明

- **ReAct 循环**：每个用户消息启动一个新的 `AgentLoop`，拥有独立的 `ContextManager`。循环流式接收 LLM 响应，收集工具调用，按安全级别执行（遵守确认规则），将结果反馈给 LLM，直到模型不再调用工具或达到迭代上限。
- **上下文压缩**：当估算的 token 数超过有效预算（1M 上下文的 980K）时，最早约 1/3 的消息会被总结为摘要释放空间。
- **记忆系统**：记忆以独立 `.md` 文件存储在 `~/.claude/projects/<slug>/memory/`，使用 YAML frontmatter 格式。`MEMORY.md` 索引被注入系统提示词，让智能体感知可用记忆。
- **子智能体**：`delegate` 工具使用 `SubAgentRunner`，默认限制 5 个并生子智能体以遵守 DeepSeek API 速率限制。每个子智能体运行独立的轻量 `AgentLoop`。

"""Cron tools — schedule/list/cancel cron jobs for periodic agent tasks."""

from __future__ import annotations

from deepagent.core.cron import CronScheduler
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


def create_cron_tools(reg: ToolRegistry, scheduler: CronScheduler) -> None:
    """Register cron-related tools into *reg*."""

    @tool(
        reg,
        name="schedule_cron",
        description="Schedule a cron job. cron is a 5-field expression: minute hour day-of-month month day-of-week. Supports *, */N, comma lists, and ranges.",
        safety_level=SafetyLevel.WRITE,
    )
    async def schedule_cron(
        cron: str,
        prompt: str,
        *,
        recurring: bool = True,
        durable: bool = True,
        max_fires: int | None = None,
    ) -> dict:
        result = scheduler.schedule(
            cron, prompt,
            recurring=recurring, durable=durable, max_fires=max_fires,
        )
        if isinstance(result, str):
            return {"success": False, "content": "", "error": result}
        return {
            "success": True,
            "content": f"Scheduled {result.id}: '{result.cron}' -> {result.prompt}",
        }

    @tool(
        reg,
        name="list_crons",
        description="List all scheduled cron jobs with their status.",
        safety_level=SafetyLevel.READONLY,
    )
    async def list_crons() -> dict:
        jobs = scheduler.list_all()
        if not jobs:
            return {"success": True, "content": "No cron jobs."}

        lines = []
        for j in jobs:
            kind = "recurring" if j.recurring else "one-shot"
            persistence = "durable" if j.durable else "session"
            fires = f", fires={j.fires}/{j.max_fires}" if j.max_fires else f", fires={j.fires}"
            lines.append(
                f"  {j.id}: '{j.cron}' -> {j.prompt[:60]}"
                f" [{kind}, {persistence}{fires}]"
            )
        return {"success": True, "content": "\n".join(lines)}

    @tool(
        reg,
        name="cancel_cron",
        description="Cancel a scheduled cron job by its ID.",
        safety_level=SafetyLevel.WRITE,
    )
    async def cancel_cron(job_id: str) -> dict:
        result = scheduler.cancel(job_id)
        if "not found" in result.lower():
            return {"success": False, "content": "", "error": result}
        return {"success": True, "content": result}

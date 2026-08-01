"""Tests for CronScheduler, cron_matches, validate_cron, and cron tools."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest
from deepagent.core.cron import CronJob, CronScheduler, cron_matches, validate_cron
from deepagent.tools.cron_tools import create_cron_tools
from deepagent.tools.registry import ToolRegistry


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """A fresh CronScheduler backed by a temporary durable file."""
    path = tmp_path / ".scheduled_tasks.json"
    return CronScheduler(path)


@pytest.fixture
def durable_path(tmp_path):
    """Path to a .scheduled_tasks.json file in a temp directory."""
    return tmp_path / ".scheduled_tasks.json"


# ==============================================================================
# cron_matches()
# ==============================================================================


def test_cron_matches_every_minute():
    """* * * * * matches any datetime."""
    dt = datetime(2026, 7, 5, 14, 30)
    assert cron_matches("* * * * *", dt) is True


def test_cron_matches_specific_minute():
    """30 9 * * * matches 9:30 on any day."""
    assert cron_matches("30 9 * * *", datetime(2026, 7, 5, 9, 30)) is True
    assert cron_matches("30 9 * * *", datetime(2026, 7, 5, 9, 31)) is False
    assert cron_matches("30 9 * * *", datetime(2026, 7, 5, 10, 30)) is False


def test_cron_matches_every_5_minutes():
    """*/5 * * * * matches minutes divisible by 5."""
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 0)) is True
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 5)) is True
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 10)) is True
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 55)) is True
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 3)) is False
    assert cron_matches("*/5 * * * *", datetime(2026, 7, 5, 14, 7)) is False


def test_cron_matches_range_of_hours():
    """0 9-17 * * * matches 9:00 through 17:00."""
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 9, 0)) is True
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 14, 0)) is True
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 17, 0)) is True
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 8, 0)) is False
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 18, 0)) is False
    # Minute must also match
    assert cron_matches("0 9-17 * * *", datetime(2026, 7, 5, 14, 30)) is False


def test_cron_matches_comma_list_of_days():
    """0 9 1,15 * * matches the 1st and 15th at 9:00."""
    assert cron_matches("0 9 1,15 * *", datetime(2026, 7, 1, 9, 0)) is True
    assert cron_matches("0 9 1,15 * *", datetime(2026, 7, 15, 9, 0)) is True
    assert cron_matches("0 9 1,15 * *", datetime(2026, 7, 2, 9, 0)) is False
    assert cron_matches("0 9 1,15 * *", datetime(2026, 7, 16, 9, 0)) is False


def test_cron_matches_specific_day_of_week():
    """0 9 * * 1 matches Monday at 9:00."""
    # 2026-07-06 is a Monday
    monday = datetime(2026, 7, 6, 9, 0)
    assert cron_matches("0 9 * * 1", monday) is True
    # Tuesday
    tuesday = datetime(2026, 7, 7, 9, 0)
    assert cron_matches("0 9 * * 1", tuesday) is False


def test_cron_matches_weekend_only():
    """0 9 * * 0,6 matches Sunday and Saturday at 9:00."""
    # 2026-07-05 is a Sunday
    sunday = datetime(2026, 7, 5, 9, 0)
    # 2026-07-11 is a Saturday
    saturday = datetime(2026, 7, 11, 9, 0)
    # 2026-07-06 is a Monday
    monday = datetime(2026, 7, 6, 9, 0)

    assert cron_matches("0 9 * * 0,6", sunday) is True
    assert cron_matches("0 9 * * 0,6", saturday) is True
    assert cron_matches("0 9 * * 0,6", monday) is False


def test_cron_matches_dom_dow_or_semantics():
    """0 9 13 * 5 matches Friday the 13th, OR any Friday, OR any 13th."""
    # Friday the 13th (2026-03-13 is a Friday)
    friday_13th = datetime(2026, 3, 13, 9, 0)
    assert cron_matches("0 9 13 * 5", friday_13th) is True

    # Friday but not the 13th (2026-03-06 is a Friday)
    friday_not_13th = datetime(2026, 3, 6, 9, 0)
    assert cron_matches("0 9 13 * 5", friday_not_13th) is True

    # 13th but not Friday (2026-04-13 is a Monday)
    day13_not_friday = datetime(2026, 4, 13, 9, 0)
    assert cron_matches("0 9 13 * 5", day13_not_friday) is True

    # Not Friday and not the 13th (2026-04-14 is a Tuesday)
    neither = datetime(2026, 4, 14, 9, 0)
    assert cron_matches("0 9 13 * 5", neither) is False


def test_cron_matches_midnight():
    """0 0 * * * matches midnight."""
    assert cron_matches("0 0 * * *", datetime(2026, 7, 5, 0, 0)) is True
    assert cron_matches("0 0 * * *", datetime(2026, 7, 5, 0, 1)) is False
    assert cron_matches("0 0 * * *", datetime(2026, 7, 5, 1, 0)) is False
    assert cron_matches("0 0 * * *", datetime(2026, 7, 5, 23, 59)) is False


def test_cron_matches_end_of_day():
    """59 23 * * * matches 23:59."""
    assert cron_matches("59 23 * * *", datetime(2026, 7, 5, 23, 59)) is True
    assert cron_matches("59 23 * * *", datetime(2026, 7, 5, 23, 58)) is False
    assert cron_matches("59 23 * * *", datetime(2026, 7, 5, 22, 59)) is False


def test_cron_matches_invalid_too_few_fields():
    """Too few fields returns False."""
    assert cron_matches("* * * *", datetime(2026, 7, 5, 14, 30)) is False
    assert cron_matches("*", datetime(2026, 7, 5, 14, 30)) is False


def test_cron_matches_step_with_zero_handled():
    """*/0 step field returns False (step must be > 0)."""
    assert cron_matches("*/0 * * * *", datetime(2026, 7, 5, 14, 0)) is False
    assert cron_matches("*/0 * * * *", datetime(2026, 7, 5, 14, 30)) is False


def test_cron_matches_invalid_field_value_returns_false():
    """Non-numeric field where digit is expected returns False."""
    assert cron_matches("abc * * * *", datetime(2026, 7, 5, 14, 30)) is False


# ==============================================================================
# validate_cron()
# ==============================================================================


def test_validate_cron_valid_complex_expression():
    """*/5 9-17 1,15 * 1-5 is valid."""
    assert validate_cron("*/5 9-17 1,15 * 1-5") is None


def test_validate_cron_valid_wildcard():
    """* * * * * is valid."""
    assert validate_cron("* * * * *") is None


def test_validate_cron_too_few_fields():
    """Too few fields returns an error message."""
    err = validate_cron("* * * *")
    assert err is not None
    assert "Expected 5 fields" in err or "got" in err


def test_validate_cron_too_many_fields():
    """Too many fields returns an error message."""
    err = validate_cron("* * * * * *")
    assert err is not None
    assert "Expected 5 fields" in err or "got" in err


def test_validate_cron_invalid_minute():
    """Minute value 60 is out of bounds [0-59]."""
    err = validate_cron("60 * * * *")
    assert err is not None
    assert "60" in err


def test_validate_cron_invalid_hour():
    """Hour value 24 is out of bounds [0-23]."""
    err = validate_cron("* 24 * * *")
    assert err is not None
    assert "24" in err


def test_validate_cron_invalid_dom():
    """Day-of-month value 0 is out of bounds [1-31]."""
    err = validate_cron("* * 0 * *")
    assert err is not None
    assert "0" in err


def test_validate_cron_negative_step():
    """Negative step is an error."""
    err = validate_cron("*-5 * * * *")
    assert err is not None


def test_validate_cron_empty_string():
    """Empty string is an error."""
    err = validate_cron("")
    assert err is not None


def test_validate_cron_valid_every_5_minutes():
    """*/5 * * * * is valid."""
    assert validate_cron("*/5 * * * *") is None


def test_validate_cron_valid_specific_hour_and_minute():
    """30 9 * * * is valid."""
    assert validate_cron("30 9 * * *") is None


def test_validate_cron_invalid_range_start_gt_end():
    """Range where start > end is an error."""
    err = validate_cron("* 17-9 * * *")
    assert err is not None


# ==============================================================================
# CronJob dataclass
# ==============================================================================


def test_cron_job_default_values():
    """CronJob uses correct default values for recurring, durable, fires."""
    job = CronJob(id="cron_000001", cron="* * * * *", prompt="test")
    assert job.id == "cron_000001"
    assert job.cron == "* * * * *"
    assert job.prompt == "test"
    assert job.recurring is True
    assert job.durable is True
    assert job.max_fires is None
    assert job.fires == 0


def test_cron_job_all_fields_set():
    """CronJob with all fields explicitly set."""
    job = CronJob(
        id="cron_000002",
        cron="0 9 * * 1-5",
        prompt="Standup reminder",
        recurring=False,
        durable=False,
        max_fires=10,
        fires=5,
    )
    assert job.id == "cron_000002"
    assert job.cron == "0 9 * * 1-5"
    assert job.prompt == "Standup reminder"
    assert job.recurring is False
    assert job.durable is False
    assert job.max_fires == 10
    assert job.fires == 5


def test_cron_job_fires_counter_mutable():
    """fires is a mutable counter that can be incremented."""
    job = CronJob(id="cron_000003", cron="* * * * *", prompt="count me")
    assert job.fires == 0
    job.fires += 1
    assert job.fires == 1
    job.fires += 5
    assert job.fires == 6


# ==============================================================================
# CronScheduler
# ==============================================================================


def test_schedule_valid_job_returns_cron_job(scheduler):
    """schedule() with valid cron returns a CronJob instance."""
    result = scheduler.schedule("0 9 * * *", "Morning check")
    assert isinstance(result, CronJob)
    assert result.id.startswith("cron_")
    assert result.cron == "0 9 * * *"
    assert result.prompt == "Morning check"


def test_schedule_invalid_cron_returns_error_string(scheduler):
    """schedule() with invalid cron returns an error string."""
    result = scheduler.schedule("invalid", "Bad cron")
    assert isinstance(result, str)
    assert "error" in result.lower() or "Expected" in result or "got" in result


def test_schedule_with_max_fires(scheduler):
    """schedule() with max_fires stores the cap."""
    result = scheduler.schedule("0 9 * * *", "Capped job", max_fires=3)
    assert isinstance(result, CronJob)
    assert result.max_fires == 3


def test_cancel_existing_job(scheduler):
    """cancel() an existing job returns a confirmation message."""
    job = scheduler.schedule("0 9 * * *", "To cancel")
    assert isinstance(job, CronJob)
    msg = scheduler.cancel(job.id)
    assert "Cancelled" in msg
    assert job.id in msg


def test_cancel_nonexistent_job(scheduler):
    """cancel() a non-existent job returns a 'not found' message."""
    msg = scheduler.cancel("cron_999999")
    assert "not found" in msg.lower()


def test_list_all_returns_jobs(scheduler):
    """list_all() returns all scheduled jobs."""
    scheduler.schedule("0 9 * * *", "Job A")
    scheduler.schedule("30 14 * * 1-5", "Job B")
    jobs = scheduler.list_all()
    assert len(jobs) == 2
    prompts = {j.prompt for j in jobs}
    assert prompts == {"Job A", "Job B"}


def test_list_all_empty(scheduler):
    """list_all() returns empty list when no jobs are scheduled."""
    assert scheduler.list_all() == []


def test_job_count(scheduler):
    """job_count reflects the number of scheduled jobs."""
    assert scheduler.job_count == 0
    scheduler.schedule("0 9 * * *", "Job 1")
    assert scheduler.job_count == 1
    scheduler.schedule("0 10 * * *", "Job 2")
    assert scheduler.job_count == 2
    job = scheduler.schedule("0 11 * * *", "Job 3")
    assert scheduler.job_count == 3
    scheduler.cancel(job.id)
    assert scheduler.job_count == 2


def test_consume_returns_none_when_queue_empty(scheduler):
    """consume() returns None if no jobs have been fired yet."""

    async def _test():
        return await scheduler.consume()

    result = asyncio.run(_test())
    assert result is None


def test_has_work_false_when_queue_empty(scheduler):
    """has_work is False when no jobs have been fired."""
    assert scheduler.has_work is False


def test_start_loads_durable_jobs_from_json_file(durable_path):
    """start() loads previously persisted durable jobs from disk."""
    # Pre-populate the durable file
    jobs_data = [
        {
            "id": "cron_123456",
            "cron": "0 9 * * *",
            "prompt": "Morning standup",
            "recurring": True,
            "durable": True,
            "max_fires": None,
            "fires": 0,
        },
        {
            "id": "cron_789012",
            "cron": "30 17 * * 1-5",
            "prompt": "End of day wrap-up",
            "recurring": True,
            "durable": True,
            "max_fires": None,
            "fires": 3,
        },
    ]
    durable_path.write_text(json.dumps(jobs_data))

    scheduler = CronScheduler(durable_path)

    async def _test():
        await scheduler.start()
        jobs = scheduler.list_all()
        assert len(jobs) == 2
        prompts = {j.prompt for j in jobs}
        assert "Morning standup" in prompts
        assert "End of day wrap-up" in prompts
        await scheduler.stop()

    asyncio.run(_test())


def test_durable_jobs_persisted_to_disk_on_schedule(durable_path):
    """schedule() with durable=True writes to the durable JSON file."""
    scheduler = CronScheduler(durable_path)
    scheduler.schedule("0 9 * * *", "Persisted job", durable=True)
    scheduler.schedule("30 14 * * 1-5", "Another", durable=True)

    assert durable_path.exists()
    raw = json.loads(durable_path.read_text())
    assert isinstance(raw, list)
    assert len(raw) == 2
    prompts = {entry["prompt"] for entry in raw}
    assert "Persisted job" in prompts
    assert "Another" in prompts


def test_durable_jobs_removed_from_disk_on_cancel(durable_path):
    """cancel() a durable job removes it from the JSON file."""
    scheduler = CronScheduler(durable_path)
    job = scheduler.schedule("0 9 * * *", "Temp job", durable=True)
    assert isinstance(job, CronJob)

    scheduler.cancel(job.id)

    raw = json.loads(durable_path.read_text())
    assert len(raw) == 0


def test_schedule_non_durable_not_persisted(durable_path):
    """schedule() with durable=False does NOT write to the JSON file."""
    scheduler = CronScheduler(durable_path)
    scheduler.schedule("0 9 * * *", "Session-only job", durable=False)

    # No file should be created since no durable jobs exist
    assert not durable_path.exists()


def test_cancel_last_durable_job_clears_file(durable_path):
    """Cancel the only durable job, file should have an empty list (not removed)."""
    scheduler = CronScheduler(durable_path)
    job1 = scheduler.schedule("0 9 * * *", "Only job", durable=True)
    assert isinstance(job1, CronJob)

    scheduler.cancel(job1.id)

    # File should still exist but contain empty list (save_durable writes empty list)
    assert durable_path.exists()
    raw = json.loads(durable_path.read_text())
    assert raw == []


# ==============================================================================
# Tool tests (ToolRegistry + create_cron_tools)
# ==============================================================================


@pytest.fixture
def reg_and_scheduler(scheduler):
    """Create a ToolRegistry with cron tools registered."""
    reg = ToolRegistry()
    create_cron_tools(reg, scheduler)
    return reg, scheduler


@pytest.mark.asyncio
async def test_schedule_cron_tool_valid(reg_and_scheduler):
    """schedule_cron tool with valid cron returns success."""
    reg, _ = reg_and_scheduler
    tool = reg.get("schedule_cron")
    result = await tool(cron="0 9 * * 1-5", prompt="Weekday morning check")
    assert result["success"] is True
    assert "cron_" in result["content"]


@pytest.mark.asyncio
async def test_schedule_cron_tool_invalid(reg_and_scheduler):
    """schedule_cron tool with invalid cron returns error."""
    reg, _ = reg_and_scheduler
    tool = reg.get("schedule_cron")
    result = await tool(cron="bad cron", prompt="This will fail")
    assert result["success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_list_crons_tool_empty(reg_and_scheduler):
    """list_crons tool when no jobs exist returns 'No cron jobs'."""
    reg, _ = reg_and_scheduler
    tool = reg.get("list_crons")
    result = await tool()
    assert result["success"] is True
    assert "No cron jobs" in result["content"]


@pytest.mark.asyncio
async def test_list_crons_tool_with_jobs(reg_and_scheduler):
    """list_crons tool lists all scheduled jobs."""
    reg, scheduler = reg_and_scheduler
    scheduler.schedule("0 9 * * *", "Standup")
    scheduler.schedule("30 14 * * 1-5", "Afternoon check")

    tool = reg.get("list_crons")
    result = await tool()
    assert result["success"] is True
    assert "Standup" in result["content"]
    assert "Afternoon check" in result["content"]


@pytest.mark.asyncio
async def test_cancel_cron_tool_existing(reg_and_scheduler):
    """cancel_cron tool with valid job ID returns success."""
    reg, scheduler = reg_and_scheduler
    job = scheduler.schedule("0 9 * * *", "Will be cancelled")
    assert isinstance(job, CronJob)

    tool = reg.get("cancel_cron")
    result = await tool(job_id=job.id)
    assert result["success"] is True
    assert "Cancelled" in result["content"]


@pytest.mark.asyncio
async def test_cancel_cron_tool_nonexistent(reg_and_scheduler):
    """cancel_cron tool with non-existent job ID returns error."""
    reg, _ = reg_and_scheduler
    tool = reg.get("cancel_cron")
    result = await tool(job_id="cron_999999")
    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_schedule_cron_tool_registers_all_three_tools(reg_and_scheduler):
    """create_cron_tools registers exactly 3 tools."""
    reg, _ = reg_and_scheduler
    names = reg.list_names()
    assert "schedule_cron" in names
    assert "list_crons" in names
    assert "cancel_cron" in names
    assert len(names) == 3


# ==============================================================================
# Async scheduler lifecycle
# ==============================================================================


@pytest.mark.asyncio
async def test_start_stop_lifecycle_no_crash(scheduler):
    """start() then stop() should not raise any exceptions."""
    await scheduler.start()
    await scheduler.stop()
    # Should reach here without errors
    assert scheduler._task is None


@pytest.mark.asyncio
async def test_fired_jobs_appear_in_queue(scheduler):
    """A job with '*' cron fires and appears in the queue after scheduler pulse."""
    async with asyncio.timeout(15):
        await scheduler.start()

        # "* * * * *" matches every minute, so it should fire quickly
        scheduler.schedule("* * * * *", "Minute tick", recurring=True)

        # Poll for a fired job (poll interval is 5s)
        job = None
        for _ in range(20):
            job = await scheduler.consume()
            if job is not None:
                break
            await asyncio.sleep(0.5)

        await scheduler.stop()

        assert job is not None
        assert isinstance(job, CronJob)
        assert "Minute tick" in job.prompt
        assert job.fires >= 1


@pytest.mark.asyncio
async def test_has_work_true_after_fire(scheduler):
    """has_work returns True after a job fires into the queue."""
    await scheduler.start()
    scheduler.schedule("* * * * *", "Frequent check", recurring=True)

    # Wait for something to appear (poll interval is 5s)
    fired = False
    for _ in range(20):
        if scheduler.has_work:
            fired = True
            break
        await asyncio.sleep(0.5)

    await scheduler.stop()

    assert fired is True


@pytest.mark.asyncio
async def test_non_recurring_job_removed_after_fire(durable_path):
    """A non-recurring job fires once and is removed from the scheduler."""
    scheduler = CronScheduler(durable_path)
    await scheduler.start()

    job = scheduler.schedule("* * * * *", "One-time", recurring=False, durable=False)
    assert isinstance(job, CronJob)
    assert scheduler.job_count == 1

    # Wait for it to fire (poll interval is 5s)
    for _ in range(20):
        consumed = await scheduler.consume()
        if consumed is not None:
            break
        await asyncio.sleep(0.5)

    await scheduler.stop()

    # After firing, the non-recurring job should be gone
    assert scheduler.job_count == 0


@pytest.mark.asyncio
async def test_max_fires_cap_enforced(scheduler):
    """A job with max_fires is cancelled after reaching the cap."""
    await scheduler.start()

    scheduler.schedule("* * * * *", "Capped at 1", recurring=True, max_fires=1)
    assert scheduler.job_count == 1

    # Wait for it to fire and be removed (poll interval is 5s)
    for _ in range(20):
        consumed = await scheduler.consume()
        if consumed is not None:
            break
        await asyncio.sleep(0.5)

    await scheduler.stop()

    # After reaching max_fires, the job should be gone
    assert scheduler.job_count == 0


@pytest.mark.asyncio
async def test_start_loads_durable_jobs_with_validation(scheduler, durable_path):
    """start() skips durable jobs with invalid cron expressions."""
    jobs_data = [
        {
            "id": "cron_111111",
            "cron": "0 9 * * *",
            "prompt": "Valid job",
            "recurring": True,
            "durable": True,
            "max_fires": None,
            "fires": 0,
        },
        {
            "id": "cron_222222",
            "cron": "bad cron expr",
            "prompt": "Invalid job",
            "recurring": True,
            "durable": True,
            "max_fires": None,
            "fires": 0,
        },
    ]
    durable_path.write_text(json.dumps(jobs_data))

    # Use the scheduler fixture's path isn't right — need same path
    scheduler2 = CronScheduler(durable_path)
    await scheduler2.start()
    jobs = scheduler2.list_all()
    await scheduler2.stop()

    assert len(jobs) == 1
    assert jobs[0].prompt == "Valid job"


@pytest.mark.asyncio
async def test_schedule_with_all_parameters(scheduler):
    """schedule() accepts all optional keyword parameters."""
    job = scheduler.schedule(
        "0 9 * * *",
        "Full config job",
        recurring=True,
        durable=True,
        max_fires=5,
    )
    assert isinstance(job, CronJob)
    assert job.recurring is True
    assert job.durable is True
    assert job.max_fires == 5

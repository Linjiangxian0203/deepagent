"""Cron scheduler — asyncio-based job scheduling with durable persistence.

Uses asyncio.create_task for the poll loop (not threading), matching
deepagent's async architecture. Jobs are registered via schedule(), validated
against 5-field cron expressions, and fired into an asyncio.Queue consumed by
the agent loop.

Reference: learn-claude-code s14_cron_scheduler. Adapted to asyncio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CronJob dataclass
# ---------------------------------------------------------------------------

@dataclass
class CronJob:
    """A scheduled cron job.

    Attributes:
        id: Unique identifier, e.g. "cron_123456".
        cron: 5-field cron expression ("M H DoM Mo DoW").
        prompt: Message injected into the agent conversation when fired.
        recurring: If True, keeps firing; if False, fire once then remove.
        durable: If True, persisted to .scheduled_tasks.json across restarts.
        max_fires: Optional cap on total number of fires.
        fires: Internal counter of how many times this job has fired.
    """

    id: str
    cron: str
    prompt: str
    recurring: bool = True
    durable: bool = True
    max_fires: int | None = None
    fires: int = 0


# ---------------------------------------------------------------------------
# Cron field matching helpers
# ---------------------------------------------------------------------------

def _cron_field_matches(field: str, value: int) -> bool:
    """Match a single cron field against an integer value.

    Supports: ``*``, ``*/N`` (step), comma-separated lists, and ranges.
    """
    if field == "*":
        return True
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return False
        step = int(step_str)
        return step > 0 and value % step == 0
    if "," in field:
        return any(
            _cron_field_matches(part.strip(), value)
            for part in field.split(",")
        )
    if "-" in field:
        lo_str, hi_str = field.split("-", 1)
        if not lo_str.isdigit() or not hi_str.isdigit():
            return False
        return int(lo_str) <= value <= int(hi_str)
    if not field.isdigit():
        return False
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check if a 5-field cron expression matches the given datetime.

    Standard cron semantics:
    - Minute, hour, and month must all match.
    - Day-of-month and day-of-week use OR semantics: if both are
      constrained (non-``*``), either matching is sufficient.

    Python ``datetime.weekday()`` returns Monday=0. Cron Sunday=0, so we
    convert: ``(dt.weekday() + 1) % 7``.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, dom, month, dow = fields

    # Python Monday=0 -> cron Sunday=0
    cron_dow = (dt.weekday() + 1) % 7

    # Minute, hour, month must all match
    if not _cron_field_matches(minute, dt.minute):
        return False
    if not _cron_field_matches(hour, dt.hour):
        return False
    if not _cron_field_matches(month, dt.month):
        return False

    # DOM / DOW: OR semantics when both are constrained
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"

    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return _cron_field_matches(dow, cron_dow)
    if dow_unconstrained:
        return _cron_field_matches(dom, dt.day)
    return _cron_field_matches(dom, dt.day) or _cron_field_matches(dow, cron_dow)


# ---------------------------------------------------------------------------
# Cron validation
# ---------------------------------------------------------------------------

# Each field: (name, low_bound, high_bound)
_FIELD_BOUNDS: list[tuple[str, int, int]] = [
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day-of-month", 1, 31),
    ("month", 1, 12),
    ("day-of-week", 0, 6),
]


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """Validate a single cron field value is within [lo, hi].

    Returns an error string or None if valid.
    """
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """Validate a 5-field cron expression.

    Returns an error string describing the first problem found, or None if the
    expression is valid.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    for i, (field, (name, lo, hi)) in enumerate(zip(fields, _FIELD_BOUNDS)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


# ---------------------------------------------------------------------------
# CronScheduler
# ---------------------------------------------------------------------------

class CronScheduler:
    """Asyncio-based cron job scheduler.

    The scheduler runs a background ``asyncio.Task`` that polls every 5 seconds.
    Fired jobs are placed onto an ``asyncio.Queue``; the agent loop calls
    ``consume()`` to retrieve them without blocking.

    Usage::

        scheduler = CronScheduler(Path(".scheduled_tasks.json"))
        await scheduler.start()
        # ... register jobs via scheduler.schedule(...) ...
        # In the agent loop:
        while job := await scheduler.consume():
            ...
        await scheduler.stop()
    """

    def __init__(self, durable_path: str | Path):
        self._durable_path = Path(durable_path)
        self._jobs: dict[str, CronJob] = {}
        self._last_fired: dict[str, str] = {}  # job_id -> "YYYY-MM-DD HH:MM"
        self._queue: asyncio.Queue[CronJob] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Load durable jobs from disk and launch the background poll loop."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._load_durable()
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("CronScheduler started (poll loop active)")

    async def stop(self) -> None:
        """Cancel the background poll loop and perform a clean shutdown."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass  # poll loop may have crashed; always reset state
            self._task = None
        logger.info("CronScheduler stopped")

    # -- Public API ----------------------------------------------------------

    def schedule(
        self,
        cron: str,
        prompt: str,
        *,
        recurring: bool = True,
        durable: bool = True,
        max_fires: int | None = None,
    ) -> CronJob | str:
        """Register a new cron job.

        Args:
            cron: 5-field cron expression.
            prompt: Message injected when the job fires.
            recurring: If True, keeps firing on schedule.
            durable: If True, persists to disk.
            max_fires: Optional total fire cap; job is cancelled when reached.

        Returns:
            The created CronJob on success, or an error string if the cron
            expression is invalid.
        """
        err = validate_cron(cron)
        if err is not None:
            return err

        job_id = f"cron_{random.randint(0, 999999):06d}"
        job = CronJob(
            id=job_id,
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
            max_fires=max_fires,
            fires=0,
        )
        self._jobs[job_id] = job
        if durable:
            self._save_durable()

        logger.info(
            "Cron job registered: %s '%s' recurring=%s durable=%s",
            job_id, cron, recurring, durable,
        )
        return job

    def cancel(self, job_id: str) -> str:
        """Cancel a cron job by ID.

        Returns:
            A confirmation message, or a "not found" message.
        """
        job = self._jobs.pop(job_id, None)
        if job is None:
            return f"Job {job_id} not found"
        if job.durable:
            self._save_durable()
        self._last_fired.pop(job_id, None)
        logger.info("Cron job cancelled: %s", job_id)
        return f"Cancelled {job_id}"

    def list_all(self) -> list[CronJob]:
        """Return a snapshot of all active jobs."""
        return list(self._jobs.values())

    async def consume(self) -> CronJob | None:
        """Non-blocking: return a fired job from the queue, or None if empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def has_work(self) -> bool:
        """True if the queue has pending fired jobs."""
        return self._queue.qsize() > 0

    @property
    def job_count(self) -> int:
        """Number of currently active scheduled jobs."""
        return len(self._jobs)

    # -- Internal: durable persistence ---------------------------------------

    def _save_durable(self) -> None:
        """Persist all durable jobs to the JSON file."""
        durable_jobs = [asdict(j) for j in self._jobs.values() if j.durable]
        try:
            self._durable_path.write_text(
                json.dumps(durable_jobs, indent=2, ensure_ascii=False)
            )
        except OSError as exc:
            logger.error("Failed to save durable cron jobs: %s", exc)

    def _load_durable(self) -> None:
        """Load durable jobs from the JSON file on startup.

        Invalid or corrupt entries are skipped with a warning.
        """
        if not self._durable_path.exists():
            return
        try:
            raw = json.loads(self._durable_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load durable cron jobs: %s", exc)
            return

        if not isinstance(raw, list):
            logger.warning("Durable cron file is not a list; skipping")
            return

        loaded = 0
        for entry in raw:
            try:
                job_id = entry["id"]
                cron_expr = entry["cron"]
            except (KeyError, TypeError):
                logger.warning("Skipping malformed cron entry: %r", entry)
                continue

            err = validate_cron(cron_expr)
            if err is not None:
                logger.warning(
                    "Skipping cron job %s with invalid expression '%s': %s",
                    job_id, cron_expr, err,
                )
                continue

            job = CronJob(
                id=job_id,
                cron=cron_expr,
                prompt=entry.get("prompt", ""),
                recurring=entry.get("recurring", True),
                durable=entry.get("durable", True),
                max_fires=entry.get("max_fires"),
                fires=entry.get("fires", 0),
            )
            self._jobs[job_id] = job
            loaded += 1

        if loaded:
            logger.info("Loaded %d durable cron job(s)", loaded)

    # -- Internal: poll loop -------------------------------------------------

    async def _poll_loop(self) -> None:
        """Background task: poll every 5s, fire matching jobs.

        Each job is evaluated in isolation — an error in one job does not crash
        the scheduler.
        """
        while True:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                return

            now = datetime.now()
            # Date-aware marker prevents duplicate fires within the same minute
            minute_marker = now.strftime("%Y-%m-%d %H:%M")

            # Iterate over a snapshot so we can mutate self._jobs safely
            for job in list(self._jobs.values()):
                try:
                    if not cron_matches(job.cron, now):
                        continue

                    # Prevent firing the same job twice in one minute
                    if self._last_fired.get(job.id) == minute_marker:
                        continue

                    self._queue.put_nowait(job)
                    self._last_fired[job.id] = minute_marker
                    job.fires += 1
                    logger.debug(
                        "Cron fired: %s '%s' (fires=%d)",
                        job.id, job.cron, job.fires,
                    )

                    # Enforce max_fires cap
                    if job.max_fires is not None and job.fires >= job.max_fires:
                        self._jobs.pop(job.id, None)
                        if job.durable:
                            self._save_durable()
                        logger.info(
                            "Cron job %s reached max_fires=%d; cancelled",
                            job.id, job.max_fires,
                        )
                        continue

                    # Non-recurring: fire once then remove
                    if not job.recurring:
                        self._jobs.pop(job.id, None)
                        if job.durable:
                            self._save_durable()
                        logger.debug("Cron one-shot %s removed after fire", job.id)

                except Exception:
                    logger.exception(
                        "Error processing cron job %s; job skipped", job.id,
                    )

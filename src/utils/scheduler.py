"""CleoBot APScheduler setup for 5-minute trading cycles."""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

from src.utils.logger import get_logger

logger = get_logger("scheduler")


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance.
    
    Returns:
        Configured AsyncIOScheduler (not yet started).
    """
    scheduler = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,       # Combine missed runs into one
            "max_instances": 1,     # Only one instance of each job at a time
            "misfire_grace_time": 60,  # Allow up to 60s late execution
        },
    )

    # Add error listener
    scheduler.add_listener(_job_error_listener, EVENT_JOB_ERROR)
    scheduler.add_listener(_job_missed_listener, EVENT_JOB_MISSED)

    return scheduler


def _job_error_listener(event):
    """Handle job execution errors."""
    logger.error(
        f"Job '{event.job_id}' raised an exception: {event.exception}",
        exc_info=event.traceback,
    )


def _job_missed_listener(event):
    """Handle missed job executions."""
    logger.warning(f"Job '{event.job_id}' missed its scheduled run time.")


def add_trading_cycle_job(scheduler: AsyncIOScheduler, callback, job_id: str = "trading_cycle"):
    """Add the 5-minute trading cycle job.
    
    Runs at minutes :02 of every 5-minute slot (e.g., :02, :07, :12, :17, etc.)
    This gives 2 minutes of orderbook data collection before signal generation.
    Per timing spec in Section 6: orderbook snapshot at +2:00 into candle.
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call each cycle.
        job_id: Job identifier.
    """
    trigger = CronTrigger(minute="2,7,12,17,22,27,32,37,42,47,52,57", second=0, timezone="UTC")
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name="Trading Cycle (5-min)",
        replace_existing=True,
    )
    logger.info(f"Trading cycle job '{job_id}' scheduled at :02 of every 5-min slot.")


def add_settlement_check_job(scheduler: AsyncIOScheduler, callback, job_id: str = "settlement_check"):
    """Add the settlement check job.
    
    Runs at minutes :01 of every 5-minute slot to check previous candle's settlement.
    (e.g., :01, :06, :11, :16, etc.)
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call each cycle.
        job_id: Job identifier.
    """
    trigger = CronTrigger(minute="0,5,10,15,20,25,30,35,40,45,50,55", second=5, timezone="UTC")
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name="Settlement Check (5-min)",
        replace_existing=True,
    )
    logger.info(f"Settlement check job '{job_id}' scheduled at :00:05 of every 5-min slot.")


def add_retrain_job(scheduler: AsyncIOScheduler, callback, hour_utc: int = 4, job_id: str = "full_retrain"):
    """Add the daily full retrain job.
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call.
        hour_utc: Hour in UTC to run the retrain (default 4 = 04:00 UTC).
        job_id: Job identifier.
    """
    trigger = CronTrigger(hour=hour_utc, minute=0, second=0, timezone="UTC")
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name=f"Full Retrain (daily at {hour_utc:02d}:00 UTC)",
        replace_existing=True,
    )
    logger.info(f"Full retrain job '{job_id}' scheduled daily at {hour_utc:02d}:00 UTC.")


def add_incremental_update_job(scheduler: AsyncIOScheduler, callback, job_id: str = "incremental_update"):
    """Add the 6-hourly incremental model update job.
    
    Runs at 04:00, 10:00, 16:00, 22:00 UTC as specified in Section 8.
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call.
        job_id: Job identifier.
    """
    trigger = CronTrigger(hour="4,10,16,22", minute=30, second=0, timezone="UTC")
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name="Incremental Model Update (6-hourly)",
        replace_existing=True,
    )
    logger.info(f"Incremental update job '{job_id}' scheduled at 04:30, 10:30, 16:30, 22:30 UTC.")


def add_daily_summary_job(scheduler: AsyncIOScheduler, callback, job_id: str = "daily_summary"):
    """Add the daily summary notification job at 00:00 UTC.
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call.
        job_id: Job identifier.
    """
    trigger = CronTrigger(hour=0, minute=0, second=30, timezone="UTC")
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name="Daily Summary (00:00 UTC)",
        replace_existing=True,
    )
    logger.info(f"Daily summary job '{job_id}' scheduled at 00:00:30 UTC.")


def add_funding_rate_job(scheduler: AsyncIOScheduler, callback, job_id: str = "funding_rate_poll"):
    """Add the funding rate polling job (every 60 seconds per Section 6).
    
    Args:
        scheduler: The APScheduler instance.
        callback: Async function to call.
        job_id: Job identifier.
    """
    trigger = IntervalTrigger(seconds=60)
    scheduler.add_job(
        callback,
        trigger=trigger,
        id=job_id,
        name="Funding Rate Poll (60s)",
        replace_existing=True,
    )
    logger.info(f"Funding rate poll job '{job_id}' scheduled every 60 seconds.")

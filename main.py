from fastapi import FastAPI
from backend.ai_agent.api_endpoint import router as transcribe_router
from backend.auth.signup import router as auth_router
from backend.logs.logs_router import router as logs_router
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from backend.ai_agent.insights import generate_insights_for_all_users
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime
import pytz



# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("APP")

# ====================== SCHEDULER SETUP ======================
jobstores = {
    "default": SQLAlchemyJobStore(url="sqlite:///jobs.sqlite3")
}

scheduler = BackgroundScheduler(
    jobstores=jobstores,
    timezone=pytz.timezone("Asia/Karachi")
)

# ====================== LIFESPAN ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== APPLICATION STARTUP STARTED ===")
    logger.info(f"Environment: {os.getenv('ENV', 'Not Set')}")
    logger.info(f"Server Time (UTC): {datetime.utcnow()}")
    logger.info(f"Server Time (Karachi): {datetime.now(pytz.timezone('Asia/Karachi'))}")

    try:
        scheduler.add_job(
            func=generate_insights_for_all_users,
            trigger=CronTrigger(
                hour=00,
                minute=00,
                timezone="Asia/Karachi"
            ),
            id="generate_insights_for_all_users",
            replace_existing=True,      # Important
        )

        scheduler.start()

        job = scheduler.get_job("generate_insights_for_all_users")

        if job and job.next_run_time:
            next_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"Daily pipeline scheduled at: {next_run}")
        else:
            logger.warning("Job was added but next_run_time is not available")

        logger.info(f"Scheduler running: {scheduler.running}")
        logger.info("Server is ready.")

    except Exception as e:
        logger.error("Failed to start scheduler", exc_info=True)

    yield   # ← FastAPI runs the app here

    # Cleanup on shutdown
    try:
        scheduler.shutdown()
        logger.info("Scheduler shutdown completed.")
    except Exception as e:
        logger.error("Error during scheduler shutdown", exc_info=True)

    logger.info("=== APPLICATION SHUTDOWN COMPLETE ===")


# ====================== FASTAPI APP ======================
app = FastAPI(title="Chronic Pain Management API", lifespan=lifespan)

app.include_router(transcribe_router)
app.include_router(auth_router)
app.include_router(logs_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
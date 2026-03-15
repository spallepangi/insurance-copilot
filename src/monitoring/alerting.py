"""
Optional alerting via Slack or Discord webhook.
If SLACK_WEBHOOK_URL or DISCORD_WEBHOOK_URL is set in .env, high-severity events can be sent.
No signup required beyond creating a free webhook in Slack/Discord.
"""

import json
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from src.utils.config import DISCORD_WEBHOOK_URL, SLACK_WEBHOOK_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _send_slack(text: str, payload: Optional[dict] = None) -> bool:
    if not SLACK_WEBHOOK_URL or not SLACK_WEBHOOK_URL.strip():
        return False
    body = {"text": text}
    if payload:
        body.setdefault("blocks", []).append({"type": "section", "text": {"type": "mrkdwn", "text": f"```{json.dumps(payload, indent=0)[:500]}```"}})
    try:
        req = Request(SLACK_WEBHOOK_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=5)
        return True
    except Exception as e:
        logger.debug("Slack webhook failed: %s", e)
        return False


def _send_discord(text: str, payload: Optional[dict] = None) -> bool:
    if not DISCORD_WEBHOOK_URL or not DISCORD_WEBHOOK_URL.strip():
        return False
    content = text
    if payload:
        content += "\n" + f"```json\n{json.dumps(payload, indent=0)[:500]}\n```"
    try:
        body = {"content": content[:2000]}
        req = Request(DISCORD_WEBHOOK_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=5)
        return True
    except Exception as e:
        logger.debug("Discord webhook failed: %s", e)
        return False


def send_alert(title: str, message: str, payload: Optional[dict] = None) -> None:
    """Send an alert to Slack and/or Discord if webhook URLs are configured."""
    text = f"**{title}**\n{message}"
    _send_slack(text, payload)
    _send_discord(text, payload)

"""
Clinic booking bot, step 1: a bot that receives messages and replies.

Run it:
    1. Copy .env.example to .env and paste your token from @BotFather.
    2. pip install -r requirements.txt
    3. python bot.py
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --- Configuration ----------------------------------------------------------
# load_dotenv() copies the values from .env into environment variables, so the
# token lives outside the code and never ends up in git.
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit(
        "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and paste your token there."
    )

# --- Logging ----------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
# httpx logs every request URL, and those URLs contain your token. Keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("clinic-bot")


# --- Handlers ---------------------------------------------------------------
# A handler is an async function that incoming updates get routed to.
#   update:  the incoming event (who sent what, in which chat)
#   context: access to the bot, plus places to store per-user / per-chat data
# They're async because a bot spends most of its time waiting on the network
# (Telegram now; Google Calendar and an LLM later). While one request waits,
# the bot can keep serving other users.


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/start is what Telegram's Start button sends when someone opens the bot."""
    name = update.effective_user.first_name
    # effective_message works for both new and edited messages
    await update.effective_message.reply_text(
        f"Hi {name}! I'm the clinic assistant (very early version).\n"
        "Send me any message and I'll echo it back. /help lists the commands."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "/start - welcome message\n"
        "/help - this list\n"
        "Anything else gets echoed back."
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Any plain text. Later, this is where 'understand the question' goes."""
    message = update.effective_message
    # Logging the text is handy while developing. Remove it before real
    # patients use the bot: people will send health details.
    log.info("chat=%s user=%s text=%r", message.chat_id, update.effective_user.id, message.text)
    await message.reply_text(f"You said: {message.text}")


# --- Wiring -----------------------------------------------------------------
def main() -> None:
    app = Application.builder().token(TOKEN).build()

    # Each update goes to the first handler that matches, in the order added.
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    # filters.TEXT & ~filters.COMMAND = text messages that aren't /commands
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    log.info("Bot started. Message it in Telegram; press Ctrl+C here to stop.")
    # Long polling: the bot keeps asking Telegram "anything new for me?".
    # No server or public URL needed, so it runs fine from your laptop.
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

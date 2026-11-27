"""
Clinic AI agent, step 3: typed messages go to the LLM agent.
The buttons stay as instant shortcuts that don't use the LLM.

Run it:
    1. Copy .env.example to .env and fill in the Telegram token and the LLM settings.
    2. pip install -r requirements.txt
    3. python bot.py
"""

import asyncio
import contextlib
import logging
import os

from dotenv import load_dotenv
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import Agent
from clinic import load_clinic

# --- Logging ----------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
# httpx logs every request URL, and Telegram URLs contain your token. Keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("clinic-bot")

# --- Configuration ----------------------------------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit(
        "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and paste your token there."
    )

# Read once at startup, so restart the bot after editing clinic.yaml or .env.
CLINIC = load_clinic()
AGENT = Agent(CLINIC)

# --- Screens (the button shortcuts) -----------------------------------------
# Every button carries a hidden "callback data" string. When someone taps it,
# Telegram sends that string back to the bot as a callback query.
MENU, PRICES, HOURS, CONTACTS, BOOK = "menu", "prices", "hours", "contacts", "book"


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("💰 Prices", callback_data=PRICES),
                InlineKeyboardButton("🕒 Opening hours", callback_data=HOURS),
            ],
            [InlineKeyboardButton("📍 Address & phone", callback_data=CONTACTS)],
            [InlineKeyboardButton("📅 Book an appointment", callback_data=BOOK)],
        ]
    )


def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back to menu", callback_data=MENU)]])


def menu_text() -> str:
    # Saying up front that this is an AI is required by Anthropic's usage policy
    # for external-facing products, and it's good practice with any provider.
    return (
        f"Hi! I'm the AI assistant of {CLINIC.name}.\n"
        "Ask me anything about our services, prices or opening hours, or tap a button."
    )


def booking_text() -> str:
    return f"Online booking is coming soon.\nFor now, please call us: {CLINIC.phone}"


# One place that says what each screen shows: (text, buttons).
SCREENS = {
    MENU: (menu_text, main_menu),
    PRICES: (CLINIC.prices_text, back_to_menu),
    HOURS: (CLINIC.hours_text, back_to_menu),
    CONTACTS: (CLINIC.contacts_text, back_to_menu),
    BOOK: (booking_text, back_to_menu),
}


# --- Handlers ---------------------------------------------------------------
async def send_screen(update: Update, screen: str) -> None:
    text, buttons = SCREENS[screen]
    await update.effective_message.reply_text(text(), reply_markup=buttons())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    AGENT.forget(str(update.effective_chat.id))  # /start begins a fresh conversation
    await send_screen(update, MENU)


async def prices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_screen(update, PRICES)


async def hours(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_screen(update, HOURS)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Just type your question, for example: \"How much is a cleaning, and are you open on Saturday?\"\n\n"
        "/start - start over\n"
        "/prices - prices\n"
        "/hours - opening hours"
    )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs when someone taps any button. query.data says which one."""
    query = update.callback_query
    # Tell Telegram the tap arrived; otherwise the button shows a loading spinner.
    await query.answer()
    text, buttons = SCREENS.get(query.data, SCREENS[MENU])
    try:
        # Change the message the button belongs to, so the menu updates in place.
        await query.edit_message_text(text(), reply_markup=buttons())
    except BadRequest as error:
        # A fast double tap asks for the screen that's already showing. That's fine.
        if "not modified" not in str(error).lower():
            raise


@contextlib.asynccontextmanager
async def typing_indicator(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Show "typing..." while the agent thinks. Telegram hides it after about
    5 seconds, so keep sending it until the answer is ready."""

    async def keep_typing() -> None:
        while True:
            with contextlib.suppress(TelegramError):  # a missed indicator isn't worth failing over
                await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)

    task = asyncio.create_task(keep_typing())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Typed messages: the agent answers."""
    chat_id = update.effective_chat.id
    async with typing_indicator(context, chat_id):
        answer = await AGENT.reply(str(chat_id), update.effective_message.text)
    await update.effective_message.reply_text(answer[:4000])  # Telegram allows 4096 characters


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await help_command(update, context)


async def non_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text("For now I can only read text. Please type your question.")


async def post_init(app: Application) -> None:
    """Runs once after startup. Sets the command list behind Telegram's Menu button."""
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Start over"),
            BotCommand("prices", "Prices"),
            BotCommand("hours", "Opening hours"),
            BotCommand("help", "What I can do"),
        ]
    )


# --- Wiring -----------------------------------------------------------------
def main() -> None:
    app = (
        Application.builder()
        .token(TOKEN)
        # Handle several chats at once. Otherwise one slow LLM answer would make
        # every other patient wait. (The agent still takes one message at a time per chat.)
        .concurrent_updates(True)
        .post_init(post_init)
        .build()
    )

    # Each update goes to the first handler that matches, in the order added.
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("prices", prices))
    app.add_handler(CommandHandler("hours", hours))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(on_button))
    # Only new messages in private chats: the agent shouldn't answer edited
    # messages, or talk in groups (the staff group comes in step 4).
    private = filters.ChatType.PRIVATE & filters.UpdateType.MESSAGE
    app.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, chat))
    app.add_handler(MessageHandler(private & filters.COMMAND, unknown_command))
    app.add_handler(MessageHandler(private & ~filters.TEXT, non_text))

    log.info("Bot started for %s, using model %s. Press Ctrl+C to stop.", CLINIC.name, AGENT.model_name)
    # Long polling: the bot keeps asking Telegram "anything new for me?".
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

import discord
from discord.ext import commands
import aiohttp
import json
import base64
import asyncio
import os
import time
import re
import logging
from datetime import datetime
from config import (
    DISCORD_TOKEN,
    BOT_NAME,
    CHAT_ENDPOINT,
    LOADING_MESSAGE,
    API_ERROR_MESSAGE,
    GENERIC_ERROR_MESSAGE,
    MAX_MESSAGE_LENGTH,
)

# ---------- LOGGING SETUP ----------
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(user_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{user_id}-{timestamp}.log")

    logger = logging.getLogger(f"bot_{user_id}_{timestamp}")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# ---------- BOT SETUP ----------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

BOT_IMG_DIR = "bot_imgs"
os.makedirs(BOT_IMG_DIR, exist_ok=True)


# ---------- PERMISSION CHECK ----------
def is_admin(member: discord.Member) -> bool:
    if member.guild_permissions.administrator:
        return True
    return False


# ---------- DISCORD ACTION REGEX ----------
DISCORD_ACTION_PATTERN = re.compile(
    r'<discord\s+([^/>]+)\s*/?>',
    re.IGNORECASE
)

ATTRIBUTE_PATTERN = re.compile(r'(\w+)="([^"]+)"')


# ---------- ALLOWED ACTIONS ----------
async def action_send(channel, content):
    await channel.send(content)


async def action_kick(guild, member, reason=None):
    await member.kick(reason=reason)


async def action_ban(guild, member, reason=None):
    await member.ban(reason=reason)


async def action_timeout(member, minutes):
    until = discord.utils.utcnow() + discord.timedelta(minutes=int(minutes))
    await member.timeout(until)


# Whitelist map
ALLOWED_ACTIONS = {
    "send": action_send,
    "kick": action_kick,
    "ban": action_ban,
    "timeout": action_timeout,
}


# ---------- CONTEXT ----------
async def get_recent_context(channel, limit=5):
    messages = []
    async for msg in channel.history(limit=limit):
        role = "assistant" if msg.author == bot.user else "user"
        messages.append({
            "role": role,
            "content": msg.content
        })
    messages.reverse()
    return messages


# ---------- API ----------
async def call_router_api(message_content, history, logger, image_feedback=None):
    payload = {
        "message": message_content,
        "history": history,
        "image_settings": {"model": "Artfusion Surreal XL.safetensors", "steps": 50, "cfg": 4,
                           "sampler": "euler_ancestral", "width": 500, "height": 500}
    }

    # Add image_feedback if an image was attached
    if image_feedback:
        payload["image_feedback"] = f"data:image/png;base64,{image_feedback}"
        logger.info(f"API REQUEST - Including image_feedback: {len(image_feedback)} chars")

    logger.info("=" * 60)
    logger.info(f"API REQUEST - Endpoint: {CHAT_ENDPOINT}")
    logger.info(f"API REQUEST - Payload keys: {list(payload.keys())}")

    try:
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(CHAT_ENDPOINT, json=payload) as response:
                logger.info(f"API RESPONSE - Status: {response.status}")
                logger.info(f"API RESPONSE - Headers: {dict(response.headers)}")

                response_data = await response.json()

                # Log response but truncate base64 images
                log_data = response_data.copy()
                if "image" in log_data and isinstance(log_data["image"], str) and len(log_data["image"]) > 100:
                    log_data["image"] = f"<base64 image data: {len(log_data['image'])} chars>"

                logger.info(f"API RESPONSE - Body: {json.dumps(log_data, indent=2)}")
                logger.info("=" * 60)

                response.raise_for_status()
                return response_data
    except asyncio.TimeoutError:
        logger.error("API ERROR - Timeout after 300 seconds")
        logger.info("=" * 60)
        return {"error": "⏱️ Request timed out - image generation took too long"}
    except aiohttp.ClientError as e:
        logger.error(f"API ERROR - Network error: {str(e)}")
        logger.info("=" * 60)
        return {"error": f"{API_ERROR_MESSAGE} (Network error: {str(e)})"}
    except Exception as e:
        logger.error(f"API ERROR - Unexpected error: {str(e)}")
        logger.info("=" * 60)
        return {"error": f"{API_ERROR_MESSAGE} ({str(e)})"}


# ---------- ACTION EXECUTOR ----------
async def extract_and_execute_actions(message: discord.Message, ai_text: str):
    matches = DISCORD_ACTION_PATTERN.findall(ai_text)
    if not matches:
        return ai_text

    cleaned_text = DISCORD_ACTION_PATTERN.sub("", ai_text).strip()

    for raw_attrs in matches:
        attrs = dict(ATTRIBUTE_PATTERN.findall(raw_attrs))
        action = attrs.get("action")

        if action not in ALLOWED_ACTIONS:
            await message.channel.send(
                f"❌ `{action}` is not an allowed action."
            )
            continue

        # 🔒 POST-GENERATION USER CHECK
        if not is_admin(message.author):
            await message.channel.send(
                f"🚫 Sorry, {message.author.mention}, you do not have permission for that."
            )
            continue

        try:
            if action == "send":
                await ALLOWED_ACTIONS[action](
                    message.channel,
                    attrs.get("content", "")
                )

            elif action in ("kick", "ban"):
                target = discord.utils.get(
                    message.guild.members,
                    name=attrs.get("target")
                )
                if not target:
                    raise ValueError("User not found")

                await ALLOWED_ACTIONS[action](
                    message.guild,
                    target,
                    attrs.get("reason")
                )

            elif action == "timeout":
                target = discord.utils.get(
                    message.guild.members,
                    name=attrs.get("target")
                )
                await ALLOWED_ACTIONS[action](
                    target,
                    attrs.get("minutes", 5)
                )

        except Exception as e:
            await message.channel.send(f"❌ Action failed: `{e}`")

    return cleaned_text


# ---------- EVENTS ----------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    await bot.change_presence(
        activity=discord.Game(name=f"{BOT_NAME} online")
    )


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    raw_content = message.content

    # ---------- DM AUTO-RESPOND ----------
    is_dm = isinstance(message.channel, discord.DMChannel)

    is_mention = False
    is_name_mentioned = False

    if not is_dm:
        is_mention = bot.user.mentioned_in(message)
        is_name_mentioned = BOT_NAME.lower() in raw_content.lower()

        if not is_mention and not is_name_mentioned:
            return

    # Setup logger for this user
    logger = setup_logger(message.author.id)

    clean = raw_content.replace(f"<@{bot.user.id}>", "")
    clean = re.sub(re.escape(BOT_NAME), "", clean, flags=re.I).strip()

    logger.info(f"MESSAGE RECEIVED - From: {message.author} ({message.author.id})")
    logger.info(f"MESSAGE RECEIVED - Channel: {message.channel}")
    logger.info(f"MESSAGE RECEIVED - Raw: {raw_content}")
    logger.info(f"MESSAGE RECEIVED - Cleaned: {clean}")

    # Check for image attachments
    attached_image_b64 = None
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                logger.info(f"IMAGE ATTACHMENT - Found: {attachment.filename}")
                try:
                    image_bytes = await attachment.read()
                    attached_image_b64 = base64.b64encode(image_bytes).decode()
                    logger.info(f"IMAGE ATTACHMENT - Converted to base64: {len(attached_image_b64)} chars")

                    # Add context to the message so LLM knows an image was attached
                    if clean:
                        clean = f"[User has attached an image for modification/refinement]\n{clean}"
                    else:
                        clean = "[User has attached an image for modification/refinement] Please analyze or refine this image."

                    logger.info(f"IMAGE ATTACHMENT - Modified message to include context")
                    break  # Only use first image
                except Exception as e:
                    logger.error(f"IMAGE ATTACHMENT ERROR - Failed to read: {str(e)}")

    loading = await message.channel.send(
        f"{message.author.mention} {LOADING_MESSAGE}"
    )

    context = await get_recent_context(message.channel)
    logger.info(f"CONTEXT - Retrieved {len(context)} messages")

    api_response = await call_router_api(clean, context, logger, attached_image_b64)

    logger.info(f"RESPONSE PROCESSING - API response keys: {list(api_response.keys())}")

    if "error" in api_response:
        logger.error(f"ERROR DETECTED - {api_response['error']}")
        await loading.edit(content=api_response["error"])
        return

    ai_text = api_response.get("text", "")
    logger.info(f"AI TEXT - Before action extraction: {ai_text}")

    ai_text = await extract_and_execute_actions(message, ai_text)
    logger.info(f"AI TEXT - After action extraction: {ai_text}")

    await loading.delete()
    logger.info("LOADING - Deleted loading message")

    # Handle image if present
    image_file = None
    if "image" in api_response and api_response["image"]:
        try:
            logger.info("IMAGE - Detected in response, processing...")

            # Clean the base64 string
            image_b64 = api_response["image"]

            # Remove data URI prefix if present (e.g., "data:image/png;base64,")
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
                logger.info("IMAGE - Removed data URI prefix")

            # Remove whitespace
            image_b64 = image_b64.strip()

            # Fix padding if needed
            missing_padding = len(image_b64) % 4
            if missing_padding:
                image_b64 += '=' * (4 - missing_padding)
                logger.info(f"IMAGE - Added {4 - missing_padding} padding characters")

            logger.info(f"IMAGE - Base64 length after cleanup: {len(image_b64)}")

            image_data = base64.b64decode(image_b64)
            logger.info(f"IMAGE - Decoded {len(image_data)} bytes")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{message.author.id}_{timestamp}.png"
            filepath = os.path.join(BOT_IMG_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(image_data)

            logger.info(f"IMAGE - Saved to {filepath}")
            image_file = discord.File(filepath, filename=filename)
            logger.info("IMAGE - Created Discord File object")
        except Exception as e:
            logger.error(f"IMAGE ERROR - Failed to process: {str(e)}")
            logger.exception("IMAGE ERROR - Full traceback:")

    # Send response
    if ai_text or image_file:
        logger.info(f"SENDING - Text: {bool(ai_text)}, Image: {bool(image_file)}")
        await message.channel.send(
            content=f"{message.author.mention} {ai_text}" if ai_text else f"{message.author.mention}",
            file=image_file
        )
        logger.info("SENDING - Message sent successfully")
    else:
        logger.warning("WARNING - No AI text or image to send!")

    await bot.process_commands(message)


# ---------- RUN ----------
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
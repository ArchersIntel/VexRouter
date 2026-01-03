# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- DISCORD BOT CONFIG ---
# Get your Discord bot token from the Discord Developer Portal
DISCORD_TOKEN = "DISCORD_TOKEN"

# The name the bot will respond to (e.g., "Vex", "RouterAI").
# The bot will only process messages that start with a mention of its name.
BOT_NAME = "Vex"

# --- MMAI ROUTER API CONFIG ---
# The base URL for your MMAI Router API (e.g., "http://localhost:8765")
API_HOST = "http://localhost:8765"

# API Endpoint to use for chat/image generation requests
CHAT_ENDPOINT = f"{API_HOST}/chat"

# --- LOGGING CONFIG ---
# Set to True to save every user request to a log file.
ENABLE_REQUEST_LOGGING = False
# Directory where log files will be saved.
LOG_DIR = "logs"

# --- Other Settings ---
# Message to display while the AI is processing the request
LOADING_MESSAGE = "🧠 Processing your request... please wait."

# Message if the bot cannot reach the API
API_ERROR_MESSAGE = "❌ Sorry, I couldn't reach the AI router API. Please check the API_HOST configuration."

# Message if the API returns a non-200 status code
GENERIC_ERROR_MESSAGE = "⚠️ An error occurred during processing. Please check the server logs."

# The file to store and load chat history.
# NOTE: For simplicity, this example does NOT implement history yet,
# but uses the history structure for the API call.
# In a real app, you would manage history per user/channel.
HISTORY_FILE = "chat_history.json"

# Maximum length of the image generation prompt/message to prevent abuse
MAX_MESSAGE_LENGTH = 500

# This is the number of messages the bot will look at before responding.
# Five is usually enough.  This is also determined by your context on the textgen web ui.
# If you pull more than you configured, it will be truncated which may lead to worse responses
CONTEXT_LENGTH = 5

# These are the image settings.  Make sure the model you want exists and you need the full file name
IMAGE_SETTINGS = {"model": "Artfusion Surreal XL.safetensors", "steps": 50, "cfg": 4,
                           "sampler": "euler_ancestral", "width": 500, "height": 500}
import os
import re
import json
import asyncio
import base64
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import subprocess
from fastapi.responses import HTMLResponse, Response
import routerconfig as config

app = FastAPI()

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TEXT_GEN_URL = os.getenv("TEXT_GEN_URL", "http://tripz-lab:5000")
COMFY_URL = os.getenv("COMFY_URL", "http://tripz-lab:8188")
CHATS_DIR = Path("chats")
CHATS_DIR.mkdir(exist_ok=True)

# Moved to config
SYSTEM_PROMPT = config.SYSTEM_PROMPT + """ 
CAPABILITIES:
- System info → propose Linux shell command
- Current info / news → request specific URL
- Docs / websites → fetch exact URL
- Discord admin → <discord action="..." />

PROTOCOL:
Image generation:
<pos_pro>...</pos_pro>
<neg_pro>...</neg_pro>
If you are refining an image use:
<img2img>true</img2img>
<denoise>0.5</denoise>

System commands:
<sys_cmd>...</sys_cmd>
<cmd_desc>...</cmd_desc>

Web access:
<web_url>...</web_url>
<web_reason>...</web_reason>

Rules:
- Use tags exactly as defined
- No markdown inside protocol tags
- Prefer wit over verbosity
"""

class Message(BaseModel):
    role: str
    content: str
    image: Optional[str] = None
    prompt: Optional[str] = None
    negative: Optional[str] = None
    command: Optional[str] = None
    command_desc: Optional[str] = None
    command_output: Optional[str] = None
    web_url: Optional[str] = None
    web_reason: Optional[str] = None
    web_content: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []
    image_settings: Optional[Dict] = None
    image_feedback: Optional[str] = None  # Base64 encoded image for LLM feedback


class SaveChatRequest(BaseModel):
    messages: List[Message]
    title: Optional[str] = None


def get_gpu_stats():
    """Return VRAM, GPU load, temperature for the first GPU using nvidia-smi."""
    try:
        # Query memory used, memory total, utilization.gpu, temperature.gpu for GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout:
            # take first GPU line (single-GPU setups)
            first_line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(',')]
            if len(parts) >= 4:
                used = int(parts[0])
                total = int(parts[1])
                load = int(parts[2])
                temp = int(parts[3])
                return {
                    "vram_used": used,
                    "vram_total": total,
                    "gpu_load": load,
                    "gpu_temp": temp
                }
    except Exception:
        pass

    # fallback values if nvidia-smi not available / parse failed
    return {"vram_used": 0, "vram_total": 0, "gpu_load": 0, "gpu_temp": 0}


def get_vram_usage():
    """Compatibility wrapper used by existing endpoints - returns used/total keys."""
    stats = get_gpu_stats()
    used = stats.get("vram_used", 0)
    total = stats.get("vram_total", 0)
    # maintain previous shape for backward compatibility
    return {"used": used, "total": total}


async def get_comfy_queue():
    """Attempt to query ComfyUI for queue information. Returns dict with queue, running, eta (seconds)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # This endpoint may vary by ComfyUI deployment; commonly /queue or /queue/status.
            # We'll try /queue first, and gracefully fallback if it fails.
            r = await client.get(f"{COMFY_URL}/queue")
            if r.status_code == 200:
                data = r.json()
                # Heuristic parsing depending on returned structure
                # Prefer explicit keys if present, otherwise try to infer lists
                queue_pending = data.get("queue_pending") or data.get("pending") or data.get("pending_jobs") or []
                queue_running = data.get("queue_running") or data.get("running") or data.get("active_jobs") or []

                queue_count = len(queue_pending) if isinstance(queue_pending, list) else int(queue_pending or 0)
                running = len(queue_running) if isinstance(queue_running, list) else int(queue_running or 0)

                # ETA heuristic: average 6s per job (tweak for your hardware)
                eta = queue_count * 6
                return {"queue": queue_count, "running": running, "eta": eta}

            # If /queue isn't present, try /status or /objects that may contain jobs
            r2 = await client.get(f"{COMFY_URL}/status")
            if r2.status_code == 200:
                d2 = r2.json()
                # best-effort parsing
                pending = d2.get("pending_jobs") or d2.get("queue") or []
                running = d2.get("running_jobs") or d2.get("running") or []
                queue_count = len(pending) if isinstance(pending, list) else int(pending or 0)
                running_count = len(running) if isinstance(running, list) else int(running or 0)
                eta = queue_count * 6
                return {"queue": queue_count, "running": running_count, "eta": eta}

        except Exception:
            pass

    # fallback when ComfyUI queue is unreachable
    return {"queue": -1, "running": 0, "eta": 0}


async def call_text_gen(messages: List[Dict], image_data: Optional[str] = None) -> str:
    """Call text-generation-webui API with optional image support"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Build the full prompt with system message and history
        full_prompt = SYSTEM_PROMPT + "\n\n"

        for msg in messages:
            if msg["role"] == "user":
                full_prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                full_prompt += f"Assistant: {msg['content']}\n\n"

        # Inject image generation reminder before the final assistant turn
        full_prompt += SYSTEM_PROMPT + "\n"
        full_prompt += "Assistant:"

        # Try multiple API endpoints
        endpoints = [
            ("/v1/chat/completions", "openai"),  # OpenAI-compatible
            ("/api/v1/generate", "textgen"),     # text-gen-webui native
            ("/v1/completions", "openai_legacy") # Legacy OpenAI
        ]

        for endpoint, api_type in endpoints:
            try:
                if api_type == "openai":
                    # OpenAI-compatible format
                    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT}]

                    for m in messages:
                        # Support vision for last user message with image
                        if m["role"] == "user" and image_data and m == messages[-1]:
                            formatted_messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": m["content"]},
                                    {"type": "image_url", "image_url": {"url": image_data}}
                                ]
                            })
                        else:
                            formatted_messages.append({"role": m["role"], "content": m["content"]})

                    # Inject image generation reminder as a system message at the end
                    formatted_messages.append({"role": "system", "content": SYSTEM_PROMPT})

                    payload = {
                        "messages": formatted_messages,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                elif api_type == "openai_legacy":
                    payload = {
                        "prompt": full_prompt,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                else:  # textgen native
                    payload = {
                        "prompt": full_prompt,
                        "max_new_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "do_sample": True,
                        "stopping_strings": ["User:", "\n\nUser:"]
                    }

                response = await client.post(
                    f"{TEXT_GEN_URL}{endpoint}",
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()

                    # Extract response based on API type
                    if api_type == "openai":
                        return data["choices"][0]["message"]["content"]
                    elif api_type == "openai_legacy":
                        return data["choices"][0]["text"].strip()
                    else:  # textgen
                        result = data["results"][0]["text"]
                        # Remove the prompt echo if present
                        if result.startswith(full_prompt):
                            result = result[len(full_prompt):].strip()
                        return result

            except Exception:
                continue

        raise HTTPException(status_code=500, detail=f"All text generation endpoints failed. Make sure text-gen-webui is running with --api flag on {TEXT_GEN_URL}")


def extract_prompts(text: str) -> tuple:
    """Extract positive and negative prompts, img2img flag, denoise value, system commands, and web URLs from text"""
    pos_match = re.search(r'<pos_pro>(.*?)</pos_pro>', text, re.DOTALL)
    neg_match = re.search(r'<neg_pro>(.*?)</neg_pro>', text, re.DOTALL)
    img2img_match = re.search(r'<img2img>(.*?)</img2img>', text, re.DOTALL)
    denoise_match = re.search(r'<denoise>(.*?)</denoise>', text, re.DOTALL)
    cmd_match = re.search(r'<sys_cmd>(.*?)</sys_cmd>', text, re.DOTALL)
    cmd_desc_match = re.search(r'<cmd_desc>(.*?)</cmd_desc>', text, re.DOTALL)
    web_url_match = re.search(r'<web_url>(.*?)</web_url>', text, re.DOTALL)
    web_reason_match = re.search(r'<web_reason>(.*?)</web_reason>', text, re.DOTALL)

    positive = pos_match.group(1).strip() if pos_match else None
    negative = neg_match.group(1).strip() if neg_match else ""
    is_img2img = img2img_match and img2img_match.group(1).strip().lower() == 'true'
    denoise = float(denoise_match.group(1).strip()) if denoise_match else 0.5
    command = cmd_match.group(1).strip() if cmd_match else None
    command_desc = cmd_desc_match.group(1).strip() if cmd_desc_match else ""
    web_url = web_url_match.group(1).strip() if web_url_match else None
    web_reason = web_reason_match.group(1).strip() if web_reason_match else ""

    # if command != None:
    #     command = execute_command(command)

    return positive, negative, is_img2img, denoise, command, command_desc, web_url, web_reason


def remove_prompt_tags(text: str) -> str:
    """Remove prompt tags from text"""
    text = re.sub(r'<pos_pro>.*?</pos_pro>', '', text, flags=re.DOTALL)
    text = re.sub(r'<neg_pro>.*?</neg_pro>', '', text, flags=re.DOTALL)
    text = re.sub(r'<img2img>.*?</img2img>', '', text, flags=re.DOTALL)
    text = re.sub(r'<denoise>.*?</denoise>', '', text, flags=re.DOTALL)
    text = re.sub(r'<sys_cmd>.*?</sys_cmd>', '', text, flags=re.DOTALL)
    text = re.sub(r'<cmd_desc>.*?</cmd_desc>', '', text, flags=re.DOTALL)
    text = re.sub(r'<web_url>.*?</web_url>', '', text, flags=re.DOTALL)
    text = re.sub(r'<web_reason>.*?</web_reason>', '', text, flags=re.DOTALL)
    return text.strip()


async def upload_image_to_comfy(image_data: str) -> str:
    """Upload a base64 image to ComfyUI's input folder and return the filename"""
    import io
    from PIL import Image
    import uuid

    # Remove data URL prefix if present
    if image_data.startswith('data:'):
        image_data = image_data.split(',')[1]

    # Decode base64
    image_bytes = base64.b64decode(image_data)

    # Generate unique filename
    filename = f"feedback_{uuid.uuid4().hex[:8]}.png"

    # Upload to ComfyUI using its upload API
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {"image": (filename, image_bytes, "image/png")}
        data = {"overwrite": "true"}

        response = await client.post(
            f"{COMFY_URL}/upload/image",
            files=files,
            data=data
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("name", filename)
        else:
            raise HTTPException(status_code=500, detail=f"Failed to upload image to ComfyUI: {response.text}")


async def call_comfyui(positive_prompt: str, negative_prompt: str, settings: Optional[Dict] = None, source_image: Optional[str] = None, denoise: float = 0.5) -> str:
    """Call ComfyUI API with a workflow matching user's setup, supports img2img"""

    import random

    # Default settings
    default_settings = {
        "model": "Artfusion Surreal XL.safetensors",
        "steps": 50,
        "cfg": 6,
        "sampler": "euler_ancestral",
        "width": 500,
        "height": 500
    }

    # Merge with provided settings
    if settings:
        default_settings.update(settings)

    # Build workflow based on whether we're doing img2img or txt2img
    if source_image:
        # Img2img workflow - use source image
        workflow = {
            "3": {
                "inputs": {
                    "seed": random.randint(0, 18446744073709551615),
                    "steps": default_settings["steps"],
                    "cfg": 6,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": denoise,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["10", 0]  # From VAEEncode
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": default_settings["model"]
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "6": {
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt+', nsfw, sexy, naked, nude',
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            },
            "10": {
                "inputs": {
                    "pixels": ["11", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEEncode"
            },
            "11": {
                "inputs": {
                    "image": source_image
                },
                "class_type": "LoadImage"
            }
        }
    else:
        # Standard txt2img workflow
        workflow = {
            "3": {
                "inputs": {
                    "seed": random.randint(0, 18446744073709551615),
                    "steps": default_settings["steps"],
                    "cfg": default_settings["cfg"],
                    "sampler_name": default_settings["sampler"],
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": default_settings["model"]
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": default_settings["width"],
                    "height": default_settings["height"],
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }

    async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout
        try:
            print(f"Sending workflow to ComfyUI...")
            print(f"Model: {default_settings['model']} | Positive prompt: {positive_prompt[:100]}...")

            # Queue the prompt
            response = await client.post(
                f"{COMFY_URL}/prompt",
                json={"prompt": workflow}
            )

            # Debug the response
            print(f"ComfyUI response status: {response.status_code}")
            if response.status_code != 200:
                print(f"ComfyUI error response: {response.text}")
                raise HTTPException(status_code=500, detail=f"ComfyUI rejected workflow: {response.text}")

            data = response.json()
            prompt_id = data["prompt_id"]
            print(f"Prompt queued with ID: {prompt_id}")

            # Poll for completion - increased for larger images
            max_attempts = 180  # 6 minutes total (180 * 2 seconds)
            for attempt in range(max_attempts):
                await asyncio.sleep(2)

                history_response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
                history_data = history_response.json()

                if prompt_id in history_data:
                    outputs = history_data[prompt_id].get("outputs", {})

                    # Find the saved image
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            image_info = node_output["images"][0]
                            filename = image_info["filename"]
                            subfolder = image_info.get("subfolder", "")

                            # Get the image
                            img_response = await client.get(
                                f"{COMFY_URL}/view",
                                params={"filename": filename, "subfolder": subfolder, "type": "output"}
                            )

                            # Convert to base64
                            img_base64 = base64.b64encode(img_response.content).decode()
                            return f"data:image/png;base64,{img_base64}"

                print(f"Waiting for completion... ({attempt + 1}/{max_attempts})")

            raise Exception("Image generation timed out")

        except httpx.HTTPStatusError as e:
            print(f"ComfyUI HTTP error: {e}")
            print(f"Response: {e.response.text}")
            raise HTTPException(status_code=500, detail=f"ComfyUI error: {e.response.text}")
        except Exception as e:
            print(f"ComfyUI error: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


async def fetch_web_content(url: str) -> str:
    """Fetch and extract text content from a URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return f"Error: HTTP {response.status_code}"

            # Simple text extraction - just get the main content
            html_content = response.text

            # Remove script and style tags
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

            # Remove HTML tags
            text_content = re.sub(r'<[^>]+>', ' ', html_content)

            # Clean up whitespace
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = text_content.strip()

            # Limit content length
            max_length = 8000
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "...(truncated)"

            return text_content

    except Exception as e:
        return f"Error fetching URL: {str(e)}"


@app.post("/fetch_url")
async def fetch_url(url: str):
    """Fetch content from a URL"""
    try:
        content = await fetch_web_content(url)
        return {
            "url": url,
            "content": content,
            "success": not content.startswith("Error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {str(e)}")


@app.post("/execute_command")
async def execute_command(command: str):
    """Execute a system command and return the output"""
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )

        output = result.stdout if result.stdout else result.stderr
        return {
            "output": output,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out after 30 seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {str(e)}")


@app.get("/models")
async def get_models():
    """Get available checkpoints from ComfyUI"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Use /object_info to get a list of all available checkpoints
            response = await client.get(f"{COMFY_URL}/object_info/CheckpointLoaderSimple")
            data = response.json()
            # The structure is data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            models = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            return {"models": models}
        except Exception as e:
            print(f"Error fetching models: {e}")
            # Fallback to the default if the API call fails
            return {"models": ["electricDreams_electricDreamsV04.safetensors", "sd_xl_base_1.0.safetensors"]}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint that routes to text-gen or ComfyUI"""

    # Get VRAM usage (backwards-compatible small shape)
    vram = get_vram_usage()

    # Build message history
    messages = [{"role": m.role, "content": m.content} for m in request.history]
    messages.append({"role": "user", "content": request.message})

    # Call text generation with optional image feedback
    response_text = await call_text_gen(messages, request.image_feedback)

    # Check if response contains image generation prompts, system commands, or web requests
    positive_prompt, negative_prompt, is_img2img, denoise, command, command_desc, web_url, web_reason = extract_prompts(response_text)

    response = {
        "text": remove_prompt_tags(response_text),
        "vram": vram
    }

    # Handle web fetch request
    if web_url:
        response["web_url"] = web_url
        response["web_reason"] = web_reason

    # Handle system command proposal
    if command:
        response["command"] = command
        response["command_desc"] = command_desc

    # Handle image generation
    if positive_prompt:
        # Get image settings from request if provided
        settings = request.image_settings if request.image_settings is not None else {}

        # Handle img2img if requested
        source_image_filename = None
        if is_img2img and request.image_feedback:
            # Upload the source image to ComfyUI
            source_image_filename = await upload_image_to_comfy(request.image_feedback)

        # Generate image
        image_data = await call_comfyui(positive_prompt, negative_prompt, settings, source_image_filename, denoise)
        response["image"] = image_data
        response["prompt"] = positive_prompt
        response["negative"] = negative_prompt

    return response


@app.post("/chat/simple")
async def chat_simple(message: str, history_json: Optional[str] = None):
    """Simplified chat endpoint for mobile - uses query parameters instead of complex JSON body"""
    try:
        history = []
        if history_json:
            import json
            history_data = json.loads(history_json)
            history = [Message(**msg) for msg in history_data]

        request = ChatRequest(
            message=message,
            history=history,
            image_settings=None,
            image_feedback=None
        )

        return await chat(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")


@app.get("/status")
async def status():
    """Quick status check - simplified version of /health for mobile"""
    gpu = get_gpu_stats()
    try:
        comfy = await get_comfy_queue()
    except:
        comfy = {"queue": -1, "running": 0, "eta": 0}

    return {
        "online": True,
        "gpu_temp": gpu.get("gpu_temp", 0),
        "gpu_load": gpu.get("gpu_load", 0),
        "vram_used_mb": gpu.get("vram_used", 0),
        "vram_total_mb": gpu.get("vram_total", 0),
        "vram_percent": round((gpu.get("vram_used", 0) / gpu.get("vram_total", 1)) * 100, 1),
        "queue": comfy.get("queue", -1),
        "generating": comfy.get("running", 0) > 0
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Use richer stats from GPU collector
    gpu = get_gpu_stats()
    comfy = {"queue": -1, "running": 0, "eta": 0}
    try:
        comfy = await get_comfy_queue()
    except Exception:
        pass

    return {
        "status": "healthy",
        "vram": {"used": gpu.get("vram_used", 0), "total": gpu.get("vram_total", 0)},
        "gpu": gpu,
        "comfy_queue": comfy,
        "services": {
            "text_gen": TEXT_GEN_URL,
            "comfyui": COMFY_URL
        }
    }


@app.post("/chats/save")
async def save_chat(request: SaveChatRequest):
    """Save chat history to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.json"
    filepath = CHATS_DIR / filename

    # Save messages with all fields including image data
    messages_data = []
    for m in request.messages:
        msg_dict = {"role": m.role, "content": m.content}
        if m.image:
            msg_dict["image"] = m.image
        if m.prompt:
            msg_dict["prompt"] = m.prompt
        if m.negative:
            msg_dict["negative"] = m.negative
        messages_data.append(msg_dict)

    chat_data = {
        "timestamp": timestamp,
        "title": request.title or f"Chat {timestamp}",
        "messages": messages_data
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

    return {"filename": filename, "timestamp": timestamp}


@app.get("/chats/list")
async def list_chats():
    """List all saved chats"""
    chats = []
    for filepath in sorted(CHATS_DIR.glob("*.json"), reverse=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chats.append({
                    "filename": filepath.name,
                    "timestamp": data.get("timestamp"),
                    "title": data.get("title", filepath.stem),
                    "message_count": len(data.get("messages", []))
                })
        except Exception:
            continue
    return {"chats": chats}


@app.get("/chats/load/{filename}")
async def load_chat(filename: str):
    """Load a specific chat by filename"""
    filepath = CHATS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat: {str(e)}")


@app.delete("/chats/delete/{filename}")
async def delete_chat(filename: str):
    """Delete a specific chat by filename"""
    filepath = CHATS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")

    try:
        filepath.unlink()
        return {"success": True, "message": "Chat deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")


@app.websocket("/ws/gpu")
async def gpu_socket(websocket: WebSocket):
    """WebSocket endpoint that pushes gpu stats + comfy queue every second."""
    await websocket.accept()
    try:
        while True:
            stats = get_gpu_stats()
            queue = await get_comfy_queue()
            # Merge the two dicts to send a single payload
            payload = {
                "vram_used": stats.get("vram_used", 0),
                "vram_total": stats.get("vram_total", 0),
                "gpu_load": stats.get("gpu_load", 0),
                "gpu_temp": stats.get("gpu_temp", 0),
                "queue": queue.get("queue", -1),
                "running": queue.get("running", 0),
                "eta": queue.get("eta", 0)
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        # client disconnected cleanly
        pass
    except Exception:
        # swallow errors to keep the server running; client will reconnect if needed
        pass


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend (Single Page Application)"""

    # Consolidated the React/Babel code and the HTML structure into one route for simplicity and stability.
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Modal AI Orchestrator</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        .markdown-content {
            line-height: 1.6;
        }
        .markdown-content p {
            margin-bottom: 0.75rem;
        }
        .markdown-content p:last-child {
            margin-bottom: 0;
        }
        .markdown-content pre {
            background-color: #1e1e1e;
            border-radius: 0.5rem;
            padding: 1rem;
            overflow-x: auto;
            margin: 0.75rem 0;
        }
        .markdown-content code {
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9em;
        }
        .markdown-content pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }
        .markdown-content :not(pre) > code {
            background-color: rgba(110, 118, 129, 0.4);
            padding: 0.2em 0.4em;
            border-radius: 0.25rem;
            font-size: 0.85em;
        }
        .markdown-content ul, .markdown-content ol {
            margin-left: 1.5rem;
            margin-bottom: 0.75rem;
        }
        .markdown-content li {
            margin-bottom: 0.25rem;
        }
        .markdown-content h1, .markdown-content h2, .markdown-content h3,
        .markdown-content h4, .markdown-content h5, .markdown-content h6 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .markdown-content h1 { font-size: 1.5em; }
        .markdown-content h2 { font-size: 1.3em; }
        .markdown-content h3 { font-size: 1.1em; }
        .markdown-content blockquote {
            border-left: 3px solid #4b5563;
            padding-left: 1rem;
            margin: 0.75rem 0;
            color: #9ca3af;
        }
        .markdown-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.75rem 0;
        }
        .markdown-content th, .markdown-content td {
            border: 1px solid #4b5563;
            padding: 0.5rem;
        }
        .markdown-content th {
            background-color: #374151;
        }
        .markdown-content a {
            color: #60a5fa;
            text-decoration: underline;
        }
        .markdown-content a:hover {
            color: #93c5fd;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;

        // Configure marked with syntax highlighting
        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {}
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true
        });

        // Component to render markdown content
        const MarkdownContent = ({ content }) => {
            const contentRef = useRef(null);

            useEffect(() => {
                if (contentRef.current) {
                    // Apply syntax highlighting to any code blocks that weren't caught by marked
                    contentRef.current.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }
            }, [content]);

            const htmlContent = marked.parse(content || '');

            return (
                <div
                    ref={contentRef}
                    className="markdown-content"
                    dangerouslySetInnerHTML={{ __html: htmlContent }}
                />
            );
        };

        const Send = () => (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
        );

        const Cpu = ({ className = "", width = 24, height = 24 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
                <rect x="9" y="9" width="6" height="6"></rect>
                <line x1="9" y1="1" x2="9" y2="4"></line>
                <line x1="15" y1="1" x2="15" y2="4"></line>
                <line x1="9" y1="20" x2="9" y2="23"></line>
                <line x1="15" y1="20" x2="15" y2="23"></line>
                <line x1="20" y1="9" x2="23" y2="9"></line>
                <line x1="20" y1="14" x2="23" y2="14"></line>
                <line x1="1" y1="9" x2="4" y2="9"></line>
                <line x1="1" y1="14" x2="4" y2="14"></line>
            </svg>
        );

        const Settings = ({ className = "", width = 20, height = 20 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M12 1v6m0 6v6m-9-9h6m6 0h6"></path>
                <path d="m19.07 4.93-4.24 4.24m0 5.66 4.24 4.24M4.93 4.93l4.24 4.24m0 5.66-4.24 4.24"></path>
            </svg>
        );

        const Sliders = ({ className = "", width = 20, height = 20 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="4" y1="21" x2="4" y2="14"></line>
                <line x1="4" y1="10" x2="4" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12" y2="3"></line>
                <line x1="20" y1="21" x2="20" y2="16"></line>
                <line x1="20" y1="12" x2="20" y2="3"></line>
                <line x1="1" y1="14" x2="7" y2="14"></line>
                <line x1="9" y1="8" x2="15" y2="8"></line>
                <line x1="17" y1="16" x2="23" y2="16"></line>
            </svg>
        );

        const History = ({ className = "", width = 20, height = 20 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v5h5"></path>
                <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"></path>
                <path d="M12 7v5l4 2"></path>
            </svg>
        );

        const Save = ({ className = "", width = 20, height = 20 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path>
                <polyline points="17 21 17 13 7 13 7 21"></polyline>
                <polyline points="7 3 7 8 15 8"></polyline>
            </svg>
        );

        const Trash = ({ className = "", width = 16, height = 16 }) => (
            <svg className={className} width={width} height={height} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                <line x1="10" y1="11" x2="10" y2="17"></line>
                <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>
        );

        // Right Sidebar component for Chat History
        const ChatHistorySidebar = ({ show, chats, onLoadChat, onDeleteChat, onSaveChat, messages }) => {
            if (!show) return null;

            const handleDelete = (e, filename) => {
                e.stopPropagation(); // Prevent triggering the load action
                if (confirm('Are you sure you want to delete this chat?')) {
                    onDeleteChat(filename);
                }
            };

            return (
                <div className="w-80 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-blue-400 flex items-center gap-2">
                            <History width={20} height={20} />
                            Chat History
                        </h3>
                        <button
                            onClick={onSaveChat}
                            disabled={messages.length === 0}
                            className="p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors"
                            title="Save Current Chat"
                        >
                            <Save width={16} height={16} />
                        </button>
                    </div>
                    <div className="space-y-2">
                        {chats.length === 0 ? (
                            <p className="text-sm text-gray-500">No saved chats yet</p>
                        ) : (
                            chats.map((chat) => (
                                <div
                                    key={chat.filename}
                                    className="relative group"
                                >
                                    <button
                                        onClick={() => onLoadChat(chat.filename)}
                                        className="w-full text-left p-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                                    >
                                        <div className="text-sm font-medium text-gray-200 truncate pr-8">{chat.title}</div>
                                        <div className="text-xs text-gray-400 mt-1">
                                            {chat.message_count} messages
                                        </div>
                                        <div className="text-xs text-gray-500 mt-1">
                                            {chat.timestamp}
                                        </div>
                                    </button>
                                    <button
                                        onClick={(e) => handleDelete(e, chat.filename)}
                                        className="absolute top-2 right-2 p-1.5 bg-red-600 hover:bg-red-700 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                                        title="Delete Chat"
                                    >
                                        <Trash width={14} height={14} />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            );
        };

        // Sidebar component for Image Generation Settings
        const Sidebar = ({ show, imageSettings, setImageSettings, availableModels }) => {
            if (!show) return null;

            const handleSettingChange = (e) => {
                const { name, value, type, checked } = e.target;
                setImageSettings(prev => ({
                    ...prev,
                    [name]: type === 'number' ? parseFloat(value) : value,
                }));
            };

            return (
                <div className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto">
                    <h3 className="text-lg font-semibold mb-4 text-blue-400 flex items-center gap-2">
                        <Sliders width={20} height={20} />
                        Image Settings
                    </h3>
                    <div className="space-y-4">
                        {/* Model Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Model (Checkpoint)</label>
                            <select
                                name="model"
                                value={imageSettings.model}
                                onChange={handleSettingChange}
                                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-sm focus:ring-blue-500 focus:border-blue-500"
                            >
                                {availableModels.map(model => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                            </select>
                        </div>

                        {/* Steps */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Steps: {imageSettings.steps}</label>
                            <input
                                type="range"
                                name="steps"
                                min="10"
                                max="150"
                                step="1"
                                value={imageSettings.steps}
                                onChange={handleSettingChange}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-lg dark:bg-gray-700"
                            />
                        </div>

                        {/* CFG Scale */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">CFG Scale: {imageSettings.cfg}</label>
                            <input
                                type="range"
                                name="cfg"
                                min="1"
                                max="20"
                                step="0.5"
                                value={imageSettings.cfg}
                                onChange={handleSettingChange}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-lg dark:bg-gray-700"
                            />
                        </div>

                        {/* Sampler */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Sampler</label>
                            <select
                                name="sampler"
                                value={imageSettings.sampler}
                                onChange={handleSettingChange}
                                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-sm focus:ring-blue-500 focus:border-blue-500"
                            >
                                {/* Common samplers, can be expanded */}
                                <option value="euler">euler</option>
                                <option value="euler_ancestral">euler_ancestral</option>
                                <option value="dpm_2">dpm_2</option>
                                <option value="dpm_2_ancestral">dpm_2_ancestral</option>
                                <option value="dpm_fast">dpm_fast</option>
                                <option value="dpm_adaptive">dpm_adaptive</option>
                                <option value="dpmpp_2s_a">dpmpp_2s_a</option>
                                <option value="dpmpp_sde">dpmpp_sde</option>
                                <option value="dpmpp_sde_gpu">dpmpp_sde_gpu</option>
                                <option value="dpmpp_2m">dpmpp_2m</option>
                                <option value="dpmpp_2m_sde">dpmpp_2m_sde</option>
                                <option value="dpmpp_2m_sde_gpu">dpmpp_2m_sde_gpu</option>
                                <option value="uni_pc">uni_pc</option>
                                <option value="uni_pc_bh2">uni_pc_bh2</option>
                            </select>
                        </div>

                        {/* Width */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Width: {imageSettings.width}</label>
                            <input
                                type="range"
                                name="width"
                                min="512"
                                max="2048"
                                step="64"
                                value={imageSettings.width}
                                onChange={handleSettingChange}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-lg dark:bg-gray-700"
                            />
                        </div>

                        {/* Height */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-1">Height: {imageSettings.height}</label>
                            <input
                                type="range"
                                name="height"
                                min="512"
                                max="2048"
                                step="64"
                                value={imageSettings.height}
                                onChange={handleSettingChange}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer range-lg dark:bg-gray-700"
                            />
                        </div>

                    </div>
                </div>
            );
        };

        const MultiModalOrchestrator = () => {
            const [messages, setMessages] = useState([]);
            const [input, setInput] = useState('');
            const [loading, setLoading] = useState(false);
            const [vramUsage, setVramUsage] = useState({ used: 0, total: 12288, load: 0, temp: 0, queue: -1, eta: 0 });
            const [config, setConfig] = useState({
                textGenUrl: 'http://tripz-lab:5000',
                comfyUrl: 'http://tripz-lab:8188',
                orchestratorUrl: window.location.origin
            });
            const [showConfig, setShowConfig] = useState(false);
            const [showSidebar, setShowSidebar] = useState(false);
            const [showHistorySidebar, setShowHistorySidebar] = useState(false);
            const [savedChats, setSavedChats] = useState([]);
            const [availableModels, setAvailableModels] = useState([]);
            const [imageSettings, setImageSettings] = useState({
                model: 'electricDreams_electricDreamsV04.safetensors',
                steps: 50,
                cfg: 3.0,
                sampler: 'euler_ancestral',
                width: 1000,
                height: 1000
            });
            const [attachedImage, setAttachedImage] = useState(null);
            const messagesEndRef = useRef(null);

            useEffect(() => {
                // Fetch available models
                fetch(`${config.orchestratorUrl}/models`)
                    .then(res => res.json())
                    .then(data => {
                        setAvailableModels(data.models);
                        if (data.models && data.models.length > 0) {
                             setImageSettings(prev => ({...prev, model: data.models[0]}));
                        }
                    })
                    .catch(err => console.error('Failed to fetch models:', err));

                // Fetch saved chats
                loadChatsList();
            }, [config.orchestratorUrl]);

            const loadChatsList = async () => {
                try {
                    const res = await fetch(`${config.orchestratorUrl}/chats/list`);
                    const data = await res.json();
                    setSavedChats(data.chats);
                } catch (err) {
                    console.error('Failed to fetch chats:', err);
                }
            };

            const saveCurrentChat = async () => {
                if (messages.length === 0) return;

                try {
                    await fetch(`${config.orchestratorUrl}/chats/save`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ messages })
                    });
                    await loadChatsList();
                } catch (err) {
                    console.error('Failed to save chat:', err);
                }
            };

            const loadChat = async (filename) => {
                try {
                    const res = await fetch(`${config.orchestratorUrl}/chats/load/${filename}`);
                    const data = await res.json();
                    setMessages(data.messages || []);
                } catch (err) {
                    console.error('Failed to load chat:', err);
                }
            };

            const deleteChat = async (filename) => {
                try {
                    await fetch(`${config.orchestratorUrl}/chats/delete/${filename}`, {
                        method: 'DELETE'
                    });
                    await loadChatsList();
                } catch (err) {
                    console.error('Failed to delete chat:', err);
                }
            };

            useEffect(() => {
                // WebSocket for realtime GPU stats
                const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
                const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/gpu`);

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        setVramUsage({
                            used: data.vram_used || 0,
                            total: data.vram_total || 0,
                            load: data.gpu_load || 0,
                            temp: data.gpu_temp || 0,
                            queue: data.queue ?? -1,
                            eta: data.eta || 0
                        });
                    } catch (e) {
                        console.warn("Invalid GPU WS payload", e);
                    }
                };

                ws.onerror = (e) => console.warn("WebSocket GPU stats error", e);
                ws.onclose = () => console.warn("WebSocket GPU stats closed");

                return () => ws.close();
            }, []);

            const scrollToBottom = () => {
                messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            };

            useEffect(() => {
                scrollToBottom();
            }, [messages]);

            const sendMessage = async () => {
                if (!input.trim() || loading) return;

                const userMessage = { role: 'user', content: input };
                setMessages(prev => [...prev, userMessage]);
                setInput('');
                setLoading(true);

                try {
                    const response = await fetch(`${config.orchestratorUrl}/chat`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: input,
                            history: messages,
                            image_settings: imageSettings,
                            image_feedback: attachedImage
                        })
                    });

                    // Clear attached image after sending
                    setAttachedImage(null);

                    const data = await response.json();

                    if (data.vram) {
                        setVramUsage(prev => ({ ...prev, used: data.vram.used, total: data.vram.total }));
                    }

                    if (response.status !== 200) {
                         throw new Error(data.detail || `HTTP Error ${response.status}`);
                    }

                    if (data.text) {
                        setMessages(prev => [...prev, { role: 'assistant', content: data.text }]);
                    }

                    if (data.image) {
                        setMessages(prev => [...prev, {
                            role: 'assistant',
                            content: '[Image Generated]',
                            image: data.image,
                            prompt: data.prompt,
                            negative: data.negative
                        }]);
                    }

                    if (data.command) {
                        setMessages(prev => [...prev, {
                            role: 'assistant',
                            content: '[Command Proposed]',
                            command: data.command,
                            command_desc: data.command_desc
                        }]);
                    }

                    if (data.web_url) {
                        setMessages(prev => [...prev, {
                            role: 'assistant',
                            content: '[Web Access Requested]',
                            web_url: data.web_url,
                            web_reason: data.web_reason
                        }]);
                    }
                } catch (error) {
                    setAttachedImage(null); // Clear on error too
                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: `Error: ${error.message}. Check your console and services (text-gen-webui, ComfyUI).`
                    }]);
                    console.error("Chat Error:", error);
                } finally {
                    setLoading(false);
                }
            };

            const attachImageForFeedback = (imageData) => {
                setAttachedImage(imageData);
                setInput("Can you refine this image? ");
            };

            const executeCommand = async (command, messageIndex) => {
                try {
                    setLoading(true);

                    // Execute the command
                    const response = await fetch(`${config.orchestratorUrl}/execute_command?command=${encodeURIComponent(command)}`, {
                        method: 'POST'
                    });

                    const data = await response.json();

                    // Update the message with command output
                    setMessages(prev => prev.map((msg, idx) =>
                        idx === messageIndex
                            ? { ...msg, command_output: data.output, content: '[Command Executed]' }
                            : msg
                    ));

                    // Send the output back to the LLM for analysis
                    const analysisMessages = [
                        ...messages.slice(0, messageIndex + 1),
                        { role: 'user', content: `Command output:\n\`\`\`\n${data.output}\n\`\`\`\n\nPlease analyze this output and answer my original question.` }
                    ];

                    const analysisResponse = await fetch(`${config.orchestratorUrl}/chat`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: `Command output:\n\`\`\`\n${data.output}\n\`\`\`\n\nPlease analyze this output and answer my original question.`,
                            history: messages.slice(0, messageIndex)
                        })
                    });

                    const analysisData = await analysisResponse.json();

                    if (analysisData.text) {
                        setMessages(prev => [...prev, { role: 'assistant', content: analysisData.text }]);
                    }

                } catch (error) {
                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: `Error executing command: ${error.message}`
                    }]);
                } finally {
                    setLoading(false);
                }
            };

            const fetchWebUrl = async (url, messageIndex) => {
                try {
                    setLoading(true);

                    // Fetch the URL content
                    const response = await fetch(`${config.orchestratorUrl}/fetch_url?url=${encodeURIComponent(url)}`, {
                        method: 'POST'
                    });

                    const data = await response.json();

                    // Update the message with web content
                    setMessages(prev => prev.map((msg, idx) =>
                        idx === messageIndex
                            ? { ...msg, web_content: data.content, content: '[Web Content Fetched]' }
                            : msg
                    ));

                    // Send the content back to the LLM for analysis
                    const analysisResponse = await fetch(`${config.orchestratorUrl}/chat`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            message: `Web content from ${url}:\n\n${data.content}\n\nPlease analyze this content and answer my original question.`,
                            history: messages.slice(0, messageIndex)
                        })
                    });

                    const analysisData = await analysisResponse.json();

                    if (analysisData.text) {
                        setMessages(prev => [...prev, { role: 'assistant', content: analysisData.text }]);
                    }

                } catch (error) {
                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: `Error fetching URL: ${error.message}`
                    }]);
                } finally {
                    setLoading(false);
                }
            };

            const vramPercentage = vramUsage.total ? (vramUsage.used / vramUsage.total) * 100 : 0;
            const vramColor = vramPercentage > 90 ? 'bg-red-500' : vramPercentage > 75 ? 'bg-yellow-500' : 'bg-green-500';

            return (
                <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
                    <div className="bg-gray-800 border-b border-gray-700 p-4 sticky top-0 z-10">
                        <div className="flex items-center justify-between max-w-7xl mx-auto">
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => setShowSidebar(!showSidebar)}
                                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                                    title="Toggle Image Settings Sidebar"
                                >
                                    <Sliders width={20} height={20} />
                                </button>
                                <Cpu className="text-blue-400" width={24} height={24} />
                                <h1 className="text-xl font-bold">AITripz MMAI Router</h1>
                                <button
                                    onClick={() => setShowHistorySidebar(!showHistorySidebar)}
                                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                                    title="Toggle Chat History Sidebar"
                                >
                                    <History width={20} height={20} />
                                </button>
                            </div>

                            <div className="flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">VRAM</span>
                                    <div className="w-32 h-2 bg-gray-700 rounded-full overflow-hidden">
                                        <div
                                            className={`${vramColor} h-full transition-all duration-300`}
                                            style={{ width: `${vramPercentage}%` }}
                                        />
                                    </div>
                                    <span className="text-xs text-gray-400 whitespace-nowrap">
                                        {vramUsage.used}MB / {vramUsage.total}MB
                                    </span>
                                </div>

                                <div className="flex items-center gap-4">
                                    <span className="text-sm text-gray-300">🧪 Load: {vramUsage.load}%</span>
                                    <span className="text-sm text-gray-300">🌡 {vramUsage.temp}°C</span>
                                    <span className="text-sm text-gray-300">📦 Queue: {vramUsage.queue}</span>
                                    <span className="text-sm text-gray-300">⏳ ETA: {vramUsage.eta}s</span>
                                </div>

                                <button
                                    onClick={() => setShowConfig(!showConfig)}
                                    className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
                                    title="Toggle Service Configuration"
                                >
                                    <Settings />
                                </button>
                            </div>
                        </div>

                        {showConfig && (
                            <div className="max-w-7xl mx-auto mt-4 p-4 bg-gray-700 rounded-lg">
                                <h3 className="font-semibold mb-3">Service Configuration</h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                    <div>
                                        <label className="text-sm text-gray-400">Text-Gen URL</label>
                                        <input
                                            type="text"
                                            value={config.textGenUrl}
                                            onChange={(e) => setConfig({...config, textGenUrl: e.target.value})}
                                            className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-sm text-gray-400">ComfyUI URL</label>
                                        <input
                                            type="text"
                                            value={config.comfyUrl}
                                            onChange={(e) => setConfig({...config, comfyUrl: e.target.value})}
                                            className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-sm text-gray-400">Orchestrator URL (Self)</label>
                                        <input
                                            type="text"
                                            value={config.orchestratorUrl}
                                            onChange={(e) => setConfig({...config, orchestratorUrl: e.target.value})}
                                            className="w-full mt-1 px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-sm"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Main content area (Sidebar + Chat + History Sidebar) */}
                    <div className="flex flex-1 overflow-hidden">

                        <Sidebar
                            show={showSidebar}
                            imageSettings={imageSettings}
                            setImageSettings={setImageSettings}
                            availableModels={availableModels}
                        />

                        <div className="flex-1 overflow-y-auto p-4">
                            <div className="max-w-4xl mx-auto space-y-4">
                                {messages.length === 0 && (
                                    <div className="text-center text-gray-500 mt-8">
                                        <Cpu className="mx-auto mb-4 opacity-50 text-gray-500" width={48} height={48} />
                                        <p>Start a conversation or request an image generation</p>
                                        <p className="text-sm mt-2">The LLM will automatically trigger ComfyUI when needed. (Try "Generate a cat in a spacesuit")</p>
                                    </div>
                                )}

                                {messages.map((msg, idx) => (
                                    <div
                                        key={idx}
                                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div
                                            className={`max-w-2xl rounded-lg p-4 ${
                                                msg.role === 'user'
                                                    ? 'bg-blue-600 text-white'
                                                    : msg.role === 'system'
                                                    ? 'bg-red-600 text-white'
                                                    : 'bg-gray-800 text-gray-100'
                                            }`}
                                        >
                                            {msg.web_url && !msg.web_content ? (
                                                <div>
                                                    <p className="mb-3">{msg.web_reason || 'I need to access this URL:'}</p>
                                                    <div className="bg-gray-900 p-3 rounded-lg mb-3 font-mono text-sm">
                                                        <code className="text-blue-400">🌐 {msg.web_url}</code>
                                                    </div>
                                                    <button
                                                        onClick={() => fetchWebUrl(msg.web_url, idx)}
                                                        disabled={loading}
                                                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
                                                    >
                                                        🌐 Fetch URL
                                                    </button>
                                                </div>
                                            ) : msg.web_content ? (
                                                <div>
                                                    <p className="mb-2 text-sm text-gray-400">Fetched from web:</p>
                                                    <div className="bg-gray-900 p-3 rounded-lg mb-2 font-mono text-xs overflow-x-auto">
                                                        <code className="text-blue-400">🌐 {msg.web_url}</code>
                                                    </div>
                                                    <div className="bg-black p-3 rounded-lg font-mono text-xs overflow-x-auto max-h-64">
                                                        <pre className="text-gray-300">{msg.web_content.substring(0, 500)}...</pre>
                                                    </div>
                                                </div>
                                            ) : msg.command && !msg.command_output ? (
                                                <div>
                                                    <p className="mb-3">{msg.command_desc || 'I can run a command to check this:'}</p>
                                                    <div className="bg-gray-900 p-3 rounded-lg mb-3 font-mono text-sm">
                                                        <code className="text-green-400">$ {msg.command}</code>
                                                    </div>
                                                    <button
                                                        onClick={() => executeCommand(msg.command, idx)}
                                                        disabled={loading}
                                                        className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors"
                                                    >
                                                        ▶ Execute Command
                                                    </button>
                                                </div>
                                            ) : msg.command_output ? (
                                                <div>
                                                    <p className="mb-2 text-sm text-gray-400">Command executed:</p>
                                                    <div className="bg-gray-900 p-3 rounded-lg mb-2 font-mono text-xs overflow-x-auto">
                                                        <code className="text-green-400">$ {msg.command}</code>
                                                    </div>
                                                    <div className="bg-black p-3 rounded-lg font-mono text-xs overflow-x-auto max-h-64">
                                                        <pre className="text-gray-300">{msg.command_output}</pre>
                                                    </div>
                                                </div>
                                            ) : msg.image ? (
                                                <div>
                                                    <a href={msg.image} target="_blank" rel="noopener noreferrer">
                                                        <img
                                                            src={msg.image}
                                                            alt="Generated"
                                                            className="rounded-lg mb-2 max-w-full cursor-pointer hover:opacity-80 transition-opacity"
                                                        />
                                                    </a>

                                                    <button
                                                        onClick={() => attachImageForFeedback(msg.image)}
                                                        className="mt-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                                                    >
                                                        💬 Give Feedback
                                                    </button>

                                                    {msg.prompt && (
                                                        <div className="text-xs text-gray-400 mt-2 p-2 bg-gray-900 rounded">
                                                            <div><strong>Prompt:</strong> <span className="text-gray-200">{msg.prompt}</span></div>
                                                            {msg.negative && <div><strong>Negative:</strong> <span className="text-gray-200">{msg.negative}</span></div>}
                                                        </div>
                                                    )}
                                                </div>
                                            ) : (
                                                <MarkdownContent content={msg.content} />
                                            )}
                                        </div>
                                    </div>
                                ))}

                                {loading && (
                                    <div className="flex justify-start">
                                        <div className="bg-gray-800 rounded-lg p-4">
                                            <div className="flex items-center gap-2">
                                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                                                <span className="text-gray-400">Processing... (Check terminal for ComfyUI status)</span>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div ref={messagesEndRef} />
                            </div>
                        </div>

                        <ChatHistorySidebar
                            show={showHistorySidebar}
                            chats={savedChats}
                            onLoadChat={loadChat}
                            onDeleteChat={deleteChat}
                            onSaveChat={saveCurrentChat}
                            messages={messages}
                        />
                    </div>


                    <div className="bg-gray-800 border-t border-gray-700 p-4 sticky bottom-0 z-10">
                        <div className="max-w-4xl mx-auto">
                            {attachedImage && (
                                <div className="mb-2 p-2 bg-gray-700 rounded-lg flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <img src={attachedImage} alt="Attached" className="w-12 h-12 object-cover rounded" />
                                        <span className="text-sm text-gray-300">Image attached for feedback</span>
                                    </div>
                                    <button
                                        onClick={() => setAttachedImage(null)}
                                        className="px-2 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                                    >
                                        Remove
                                    </button>
                                </div>
                            )}
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                                    placeholder="Type your message or request an image..."
                                    className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
                                    disabled={loading}
                                />
                                <button
                                    onClick={sendMessage}
                                    disabled={loading || !input.trim()}
                                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg transition-colors flex items-center gap-2"
                                >
                                    <Send />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            );
        };

        ReactDOM.render(<MultiModalOrchestrator />, document.getElementById('root'));
    </script>
</body>
</html>"""

    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
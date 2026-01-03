# MMAI Router API Documentation

Base URL: `http://your-server:8765`

All endpoints return JSON unless otherwise specified.

---

## 📱 Chat Endpoints

### POST /chat/simple 🆕 (Mobile Friendly)
Simplified chat endpoint using query parameters - easier for mobile apps.

**Query Parameters:**
- `message` (string, required): The message to send
- `history_json` (string, optional): JSON-encoded array of previous messages

**Response:** Same as `/chat`

**Example:**
```bash
curl -X POST "http://localhost:8765/chat/simple?message=Hello%20there"
```

**Android Example:**
```kotlin
val url = "http://your-server:8765/chat/simple?message=${URLEncoder.encode("Hello!", "UTF-8")}"
val request = Request.Builder().url(url).post("".toRequestBody()).build()
```

---

### POST /chat
Send a message and get AI response with optional image/video generation or command execution.

**Request Body:**
```json
{
  "message": "string",
  "history": [
    {
      "role": "user|assistant",
      "content": "string",
      "image": "base64_string (optional)",
      "command": "string (optional)",
      "command_output": "string (optional)"
    }
  ],
  "image_settings": {
    "model": "string",
    "steps": 50,
    "cfg": 3.0,
    "sampler": "euler_ancestral",
    "width": 1000,
    "height": 1000
  },
  "image_feedback": "base64_image_string (optional)"
}
```

**Response:**
```json
{
  "text": "AI response text",
  "vram": {
    "used": 1234,
    "total": 12288
  },
  "image": "base64_image (if generated)",
  "prompt": "image prompt (if generated)",
  "negative": "negative prompt (if generated)",
  "command": "shell command (if proposed)",
  "command_desc": "command description (if proposed)"
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "history": []
  }'
```

---

## 💾 Chat History Endpoints

### GET /chats/list
Get list of all saved chats.

**Response:**
```json
{
  "chats": [
    {
      "filename": "20250101_120000.json",
      "timestamp": "20250101_120000",
      "title": "Chat 20250101_120000",
      "message_count": 10
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8765/chats/list
```

---

### GET /chats/load/{filename}
Load a specific chat by filename.

**Response:**
```json
{
  "timestamp": "20250101_120000",
  "title": "Chat Title",
  "messages": [
    {
      "role": "user",
      "content": "message text",
      "image": "base64_string (optional)"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8765/chats/load/20250101_120000.json
```

---

### POST /chats/save
Save current chat history.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user|assistant",
      "content": "message text",
      "image": "base64 (optional)",
      "prompt": "string (optional)",
      "negative": "string (optional)"
    }
  ],
  "title": "Chat Title (optional)"
}
```

**Response:**
```json
{
  "filename": "20250101_120000.json",
  "timestamp": "20250101_120000"
}
```

---

### DELETE /chats/delete/{filename}
Delete a specific chat.

**Response:**
```json
{
  "success": true,
  "message": "Chat deleted successfully"
}
```

---

## 🎨 Image Generation Endpoints

### GET /models
Get list of available ComfyUI checkpoints.

**Response:**
```json
{
  "models": [
    "model1.safetensors",
    "model2.safetensors"
  ]
}
```

**Example:**
```bash
curl http://localhost:8765/models
```

---

## 💻 System Command Endpoints

### POST /execute_command
Execute a shell command on the server.

**Query Parameters:**
- `command` (string, required): Shell command to execute

**Response:**
```json
{
  "output": "command output text",
  "returncode": 0,
  "success": true
}
```

**Example:**
```bash
curl -X POST "http://localhost:8765/execute_command?command=df%20-h"
```

**Note:** Command has 30-second timeout.

---

## 📊 System Health Endpoints

### GET /status 🆕 (Mobile Friendly)
Quick status check - simplified and optimized for mobile polling.

**Response:**
```json
{
  "online": true,
  "gpu_temp": 65,
  "gpu_load": 45,
  "vram_used_mb": 1234,
  "vram_total_mb": 12288,
  "vram_percent": 10.0,
  "queue": 0,
  "generating": false
}
```

**Example:**
```bash
curl http://localhost:8765/status
```

**Android Example (Polling):**
```kotlin
// Poll every 5 seconds
val handler = Handler(Looper.getMainLooper())
val runnable = object : Runnable {
    override fun run() {
        val request = Request.Builder()
            .url("http://your-server:8765/status")
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                val data = JSONObject(response.body?.string())
                val temp = data.getInt("gpu_temp")
                val generating = data.getBoolean("generating")
                // Update UI
            }
        })

        handler.postDelayed(this, 5000) // Repeat every 5s
    }
}
handler.post(runnable)
```

---

### GET /health
Get system health status including GPU, VRAM, and ComfyUI queue.

**Response:**
```json
{
  "status": "healthy",
  "vram": {
    "used": 1234,
    "total": 12288
  },
  "gpu": {
    "vram_used": 1234,
    "vram_total": 12288,
    "gpu_load": 45,
    "gpu_temp": 65,
    "queue": 0,
    "running": 0,
    "eta": 0
  },
  "comfy_queue": {
    "queue": 0,
    "running": 0,
    "eta": 0
  },
  "services": {
    "text_gen": "http://tripz-lab:5000",
    "comfyui": "http://tripz-lab:8188"
  }
}
```

**Example:**
```bash
curl http://localhost:8765/health
```

---

## 🔌 WebSocket Endpoints

### WS /ws/gpu
Real-time GPU statistics (updates every second).

**Message Format:**
```json
{
  "vram_used": 1234,
  "vram_total": 12288,
  "gpu_load": 45,
  "gpu_temp": 65,
  "queue": 0,
  "running": 0,
  "eta": 0
}
```

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8765/ws/gpu');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('GPU Stats:', data);
};
```

---

## 📝 Message Object Structure

Messages in chat history and requests follow this structure:

```json
{
  "role": "user|assistant|system",
  "content": "text content",
  "image": "base64_string (optional)",
  "video": "video_url (optional)",
  "prompt": "generation prompt (optional)",
  "negative": "negative prompt (optional)",
  "command": "shell command (optional)",
  "command_desc": "command description (optional)",
  "command_output": "command output (optional)"
}
```

---

## 🚀 Quick Start for Android

### 1. Simple Chat Request
```kotlin
val client = OkHttpClient()
val json = JSONObject().apply {
    put("message", "Hello!")
    put("history", JSONArray())
}

val request = Request.Builder()
    .url("http://your-server:8765/chat")
    .post(json.toString().toRequestBody("application/json".toMediaType()))
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val data = JSONObject(response.body?.string())
        val text = data.getString("text")
        // Handle response
    }
})
```

### 2. Get System Health
```kotlin
val request = Request.Builder()
    .url("http://your-server:8765/health")
    .get()
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val data = JSONObject(response.body?.string())
        val gpuTemp = data.getJSONObject("gpu").getInt("gpu_temp")
        // Update UI
    }
})
```

### 3. Load Chat History
```kotlin
val request = Request.Builder()
    .url("http://your-server:8765/chats/list")
    .get()
    .build()

client.newCall(request).enqueue(object : Callback {
    override fun onResponse(call: Call, response: Response) {
        val data = JSONObject(response.body?.string())
        val chats = data.getJSONArray("chats")
        // Display chat list
    }
})
```

---

## 🔒 Security Notes

- No authentication required (LAN/Tailscale only)
- Commands execute with server user permissions
- 30-second timeout on command execution
- CORS enabled for all origins

---

## ⚡ Rate Limits

None currently implemented.

---

## 🐛 Error Responses

All errors return standard HTTP status codes with JSON:

```json
{
  "detail": "Error message"
}
```

Common status codes:
- `400` - Bad Request
- `404` - Not Found
- `408` - Timeout
- `500` - Internal Server Error

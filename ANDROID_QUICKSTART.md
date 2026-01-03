# Android App Quick Start Guide

## Prerequisites

1. Android Studio
2. Kotlin knowledge
3. Your server IP/hostname (e.g., `192.168.1.100:8765` or `server.tail-scale.ts.net:8765`)

---

## 📱 Minimal Android App Structure

### 1. Add Dependencies (build.gradle.kts)

```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.2")
}
```

### 2. Add Internet Permission (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

---

## 🔌 API Client Class

Create `MMAIApiClient.kt`:

```kotlin
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.json.JSONArray
import java.io.IOException
import java.net.URLEncoder

class MMAIApiClient(private val baseUrl: String) {
    private val client = OkHttpClient()

    // Simple chat - no history
    fun sendMessage(
        message: String,
        callback: (Result<ChatResponse>) -> Unit
    ) {
        val encodedMessage = URLEncoder.encode(message, "UTF-8")
        val url = "$baseUrl/chat/simple?message=$encodedMessage"

        val request = Request.Builder()
            .url(url)
            .post("".toRequestBody())
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val body = response.body?.string() ?: ""
                    val json = JSONObject(body)

                    val chatResponse = ChatResponse(
                        text = json.optString("text", ""),
                        image = json.optString("image", null),
                        command = json.optString("command", null),
                        commandDesc = json.optString("command_desc", null)
                    )

                    callback(Result.success(chatResponse))
                } catch (e: Exception) {
                    callback(Result.failure(e))
                }
            }
        })
    }

    // Get system status
    fun getStatus(callback: (Result<SystemStatus>) -> Unit) {
        val request = Request.Builder()
            .url("$baseUrl/status")
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val body = response.body?.string() ?: ""
                    val json = JSONObject(body)

                    val status = SystemStatus(
                        online = json.getBoolean("online"),
                        gpuTemp = json.getInt("gpu_temp"),
                        gpuLoad = json.getInt("gpu_load"),
                        vramPercent = json.getDouble("vram_percent"),
                        queue = json.getInt("queue"),
                        generating = json.getBoolean("generating")
                    )

                    callback(Result.success(status))
                } catch (e: Exception) {
                    callback(Result.failure(e))
                }
            }
        })
    }

    // Execute system command
    fun executeCommand(
        command: String,
        callback: (Result<CommandResult>) -> Unit
    ) {
        val encodedCommand = URLEncoder.encode(command, "UTF-8")
        val url = "$baseUrl/execute_command?command=$encodedCommand"

        val request = Request.Builder()
            .url(url)
            .post("".toRequestBody())
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val body = response.body?.string() ?: ""
                    val json = JSONObject(body)

                    val result = CommandResult(
                        output = json.getString("output"),
                        success = json.getBoolean("success"),
                        returncode = json.getInt("returncode")
                    )

                    callback(Result.success(result))
                } catch (e: Exception) {
                    callback(Result.failure(e))
                }
            }
        })
    }

    // Get chat history list
    fun getChatsList(callback: (Result<List<ChatInfo>>) -> Unit) {
        val request = Request.Builder()
            .url("$baseUrl/chats/list")
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(e))
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val body = response.body?.string() ?: ""
                    val json = JSONObject(body)
                    val chatsArray = json.getJSONArray("chats")

                    val chats = mutableListOf<ChatInfo>()
                    for (i in 0 until chatsArray.length()) {
                        val chat = chatsArray.getJSONObject(i)
                        chats.add(ChatInfo(
                            filename = chat.getString("filename"),
                            title = chat.getString("title"),
                            messageCount = chat.getInt("message_count"),
                            timestamp = chat.getString("timestamp")
                        ))
                    }

                    callback(Result.success(chats))
                } catch (e: Exception) {
                    callback(Result.failure(e))
                }
            }
        })
    }
}

// Data classes
data class ChatResponse(
    val text: String,
    val image: String?,
    val command: String?,
    val commandDesc: String?
)

data class SystemStatus(
    val online: Boolean,
    val gpuTemp: Int,
    val gpuLoad: Int,
    val vramPercent: Double,
    val queue: Int,
    val generating: Boolean
)

data class CommandResult(
    val output: String,
    val success: Boolean,
    val returncode: Int
)

data class ChatInfo(
    val filename: String,
    val title: String,
    val messageCount: Int,
    val timestamp: String
)
```

---

## 🎨 Example MainActivity

```kotlin
import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {
    private lateinit var api: MMAIApiClient
    private lateinit var messageInput: EditText
    private lateinit var sendButton: Button
    private lateinit var chatDisplay: TextView
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize API client
        api = MMAIApiClient("http://YOUR_SERVER_IP:8765")

        // Find views
        messageInput = findViewById(R.id.messageInput)
        sendButton = findViewById(R.id.sendButton)
        chatDisplay = findViewById(R.id.chatDisplay)
        statusText = findViewById(R.id.statusText)

        // Setup listeners
        sendButton.setOnClickListener {
            sendMessage()
        }

        // Start status polling
        startStatusPolling()
    }

    private fun sendMessage() {
        val message = messageInput.text.toString()
        if (message.isBlank()) return

        // Disable input while processing
        sendButton.isEnabled = false
        chatDisplay.append("\nYou: $message\n")

        api.sendMessage(message) { result ->
            runOnUiThread {
                result.onSuccess { response ->
                    chatDisplay.append("AI: ${response.text}\n")

                    // Handle command if proposed
                    if (response.command != null) {
                        showCommandDialog(response.command, response.commandDesc ?: "")
                    }

                    // Handle image if generated
                    if (response.image != null) {
                        chatDisplay.append("[Image Generated]\n")
                        // TODO: Display image
                    }
                }

                result.onFailure { error ->
                    chatDisplay.append("Error: ${error.message}\n")
                }

                sendButton.isEnabled = true
                messageInput.text.clear()
            }
        }
    }

    private fun showCommandDialog(command: String, description: String) {
        AlertDialog.Builder(this)
            .setTitle("Execute Command?")
            .setMessage("$description\n\n$ $command")
            .setPositiveButton("Execute") { _, _ ->
                executeCommand(command)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun executeCommand(command: String) {
        api.executeCommand(command) { result ->
            runOnUiThread {
                result.onSuccess { cmdResult ->
                    chatDisplay.append("\n[Command Output]\n${cmdResult.output}\n")
                }

                result.onFailure { error ->
                    chatDisplay.append("Command error: ${error.message}\n")
                }
            }
        }
    }

    private fun startStatusPolling() {
        val scope = CoroutineScope(Dispatchers.Main)
        scope.launch {
            while (true) {
                api.getStatus { result ->
                    runOnUiThread {
                        result.onSuccess { status ->
                            statusText.text = "GPU: ${status.gpuTemp}°C | " +
                                    "Load: ${status.gpuLoad}% | " +
                                    "VRAM: ${status.vramPercent.toInt()}% | " +
                                    if (status.generating) "⚡ Generating" else "✓ Ready"
                        }
                    }
                }
                delay(5000) // Poll every 5 seconds
            }
        }
    }
}
```

---

## 📐 Example Layout (activity_main.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <!-- Status Bar -->
    <TextView
        android:id="@+id/statusText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:padding="8dp"
        android:background="#333333"
        android:textColor="#FFFFFF"
        android:text="Status: Connecting..." />

    <!-- Chat Display -->
    <ScrollView
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:layout_marginTop="16dp">

        <TextView
            android:id="@+id/chatDisplay"
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:textSize="14sp"
            android:padding="8dp"
            android:background="#F5F5F5" />
    </ScrollView>

    <!-- Input Area -->
    <LinearLayout
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:orientation="horizontal"
        android:layout_marginTop="16dp">

        <EditText
            android:id="@+id/messageInput"
            android:layout_width="0dp"
            android:layout_height="wrap_content"
            android:layout_weight="1"
            android:hint="Type your message..."
            android:padding="12dp" />

        <Button
            android:id="@+id/sendButton"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Send"
            android:layout_marginStart="8dp" />
    </LinearLayout>
</LinearLayout>
```

---

## 🚀 Quick Setup Steps

1. **Create new Android project** in Android Studio
2. **Copy the API client class** above
3. **Update server URL** in MainActivity: `"http://YOUR_SERVER_IP:8765"`
4. **Add dependencies** to build.gradle.kts
5. **Add internet permission** to AndroidManifest.xml
6. **Create layout** XML
7. **Run on device** (emulator won't work with local network)

---

## 🔒 For Tailscale Access

If using Tailscale, replace server URL with your Tailscale hostname:

```kotlin
api = MMAIApiClient("http://server-name.tail-scale.ts.net:8765")
```

---

## 🎯 Features to Add

- [ ] Image display (use Glide or Coil for base64 images)
- [ ] Chat history persistence
- [ ] Markdown rendering
- [ ] Push notifications when generation completes
- [ ] Voice input
- [ ] Dark theme
- [ ] Swipe to delete chats
- [ ] Image generation settings UI

---

## 📚 Additional Resources

- OkHttp docs: https://square.github.io/okhttp/
- Full API docs: See `API.md` in project root
- Kotlin coroutines: https://kotlinlang.org/docs/coroutines-overview.html

# Getting Started

<cite>
**Referenced Files in This Document**   
- [main.py](file://main.py)
- [bartending_agent.py](file://bartending_agent.py)
- [requirements.txt](file://requirements.txt)
- [README.md](file://README.md)
- [submission_notebook.ipynb](file://notebooks/submission_notebook.ipynb) - *Updated in recent commit*
</cite>

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Cloning the Repository](#cloning-the-repository)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Installing Dependencies](#installing-dependencies)
5. [Configuring API Keys](#configuring-api-keys)
6. [Running the Application](#running-the-application)
7. [Interacting with the AI Bartender](#interacting-with-the-ai-bartender)
8. [Quick Start Example](#quick-start-example)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Prerequisites

Before setting up the Maya Bartending Agent, ensure your system meets the following requirements:

- **Python 3.8 or higher**: The application is built using Python and requires a compatible version. Verify your installation with:
  ```bash
  python --version
  ```
- **Internet Connection**: Required to access external APIs (Google AI Studio and Cartesia) for AI processing and text-to-speech functionality.
- **Terminal Access**: You will need command-line access to execute setup and run commands.

**Section sources**
- [requirements.txt](file://requirements.txt#L1-L9)
- [README.md](file://README.md#L1-L5)

## Cloning the Repository

To begin, clone the repository from its source (e.g., GitHub or another version control platform). Replace the URL with the actual repository location:

```bash
git clone https://github.com/your-username/MayaBartendingAgent.git
cd MayaBartendingAgent
```

This creates a local copy of the project, including all source files and configuration.

**Section sources**
- [README.md](file://README.md#L6-L18)

## Setting Up the Environment

It is recommended to use a virtual environment to isolate the project’s dependencies from your system-wide Python packages.

### Create a Virtual Environment

Run the following command to create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

### Activate the Virtual Environment

Depending on your operating system, use the appropriate command:

- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

Once activated, your terminal prompt should display `(.venv)` indicating the virtual environment is active.

**Section sources**
- [README.md](file://README.md#L6-L18)

## Installing Dependencies

The project depends on several Python packages listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

This command installs all required libraries, including:
- `google-generativeai`: For interacting with Google's Gemini API
- `cartesia`: For text-to-speech synthesis
- `gradio`: For the web-based user interface
- `python-dotenv`: For loading environment variables from a `.env` file
- `tenacity`: For retry logic on API calls

Ensure the installation completes without errors. If you encounter permission issues, make sure your virtual environment is activated.

**Section sources**
- [requirements.txt](file://requirements.txt#L1-L9)
- [README.md](file://README.md#L20-L21)

## Configuring API Keys

The application requires API keys from two services: **Google AI Studio** (for Gemini) and **Cartesia** (for voice synthesis).

### Step 1: Obtain Google AI Studio API Key

1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click on **Get API Key** or navigate to the API keys section.
4. Create a new API key or copy an existing one.
5. Save the key securely.

### Step 2: Obtain Cartesia API Key

1. Visit [Cartesia AI](https://cartesia.ai/).
2. Sign up or log in to your account.
3. Navigate to the **API Keys** section.
4. Generate a new API key and copy it.

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory of the project:

```bash
touch .env
```

Open the file in a text editor and add the following lines:

```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here
```

Replace the placeholders with your actual API keys.

> **Note**: The application uses `python-dotenv` to load these variables. If no `.env` file is found, it will fall back to system environment variables.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L45-L52)

## Running the Application

Once dependencies are installed and API keys are configured, start the application:

```bash
python main.py
```

Upon successful launch, you should see output similar to:

```
INFO - Successfully imported agent logic functions.
INFO - Successfully initialized Gemini model: gemini-2.0-flash
INFO - Successfully initialized Cartesia client.
INFO - Launching Gradio interface locally...
Running on local URL:  http://127.0.0.1:7860
```

The Gradio interface will be accessible via the provided local URL.

**Section sources**
- [main.py](file://main.py#L138-L142)
- [README.md](file://README.md#L23-L24)

## Interacting with the AI Bartender

1. Open your web browser and navigate to `http://127.0.0.1:7860`.
2. You will see a chat interface with a bartender avatar on the left and a message input box on the right.
3. Type your request, such as:
   - `"I'd like a Margarita"`
   - `"Show me the menu"`
   - `"What's MOK 5-ha?"`
4. Press **Send** or hit Enter.
5. The AI bartender will respond with a text message and an audio response (if TTS is working).
6. Your order items are tracked in the background and can be reviewed by asking, `"What's in my order?"`

The conversation history is preserved within the session until you click **Clear Conversation**.

## Quick Start Example

Follow these steps for a complete first-time experience:

```bash
# Clone the repo
git clone https://github.com/your-username/MayaBartendingAgent.git
cd MayaBartendingAgent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Use appropriate command for your OS

# Install dependencies
pip install -r requirements.txt

# Create .env file with your keys
echo "GEMINI_API_KEY=your_gemini_key" >> .env
echo "CARTESIA_API_KEY=your_cartesia_key" >> .env

# Run the app
python main.py
```

After launching:
1. Open `http://127.0.0.1:7860` in your browser.
2. Type: `"Hi, I'd like a Mojito."`
3. Observe the AI response: `"Coming right up! I've added Mojito to your order."`
4. Audio will play automatically.
5. Ask: `"What's my current order?"`
6. The AI replies: `"You have 1 Mojito in your order."`

You’ve successfully placed your first drink order!

## Troubleshooting Common Issues

### Issue: Missing Dependencies or Module Not Found
**Symptom**: `ModuleNotFoundError: No module named 'google.generativeai'`  
**Solution**: Ensure you ran `pip install -r requirements.txt` and your virtual environment is activated.

### Issue: API Key Not Found
**Symptom**: `FATAL: GEMINI_API_KEY not found in environment variables or .env file.`  
**Solution**:  
- Verify the `.env` file exists in the project root.  
- Check for typos in variable names (`GEMINI_API_KEY`, `CARTESIA_API_KEY`).  
- Restart the application after editing the `.env` file.

### Issue: Cartesia Voice ID Not Set
**Symptom**: `CARTESIA_VOICE_ID is not set to a valid ID.`  
**Solution**:  
- Visit the [Cartesia dashboard](https://cartesia.ai/dashboard), select a voice, and copy its ID.  
- Update the `CARTESIA_VOICE_ID` variable in `bartending_agent.py`.

### Issue: Gradio Interface Not Loading
**Symptom**: `Connection refused` or browser timeout  
**Solution**:  
- Ensure the app is running (`python main.py`).  
- Check if port 7860 is in use: `lsof -i :7860` (macOS/Linux).  
- Try accessing `http://localhost:7860` instead.

### Issue: TTS Audio Not Playing
**Symptom**: No sound despite successful text response  
**Solution**:  
- Confirm `autoplay` is enabled in your browser.  
- Check console logs for TTS errors.  
- Validate your Cartesia API key has sufficient credits.

### Issue: Gemini API Call Fails
**Symptom**: `Failed to initialize Gemini model`  
**Solution**:  
- Confirm the model name `gemini-2.0-flash` is valid and accessible in your Google AI Studio plan.  
- Ensure your API key has permissions enabled for the Gemini API.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L54-L65)
- [main.py](file://main.py#L10-L18)
- [bartending_agent.py](file://bartending_agent.py#L100-L110)
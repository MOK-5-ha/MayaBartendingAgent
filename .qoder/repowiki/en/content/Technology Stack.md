# Technology Stack

<cite>
**Referenced Files in This Document**   
- [main.py](file://main.py#L1-L142)
- [bartending_agent.py](file://bartending_agent.py#L1-L374)
- [requirements.txt](file://requirements.txt#L1-L9)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Frameworks and Libraries](#core-frameworks-and-libraries)
3. [Gradio for UI](#gradio-for-ui)
4. [Google Generative AI for LLM-Powered Responses](#google-generative-ai-for-llm-powered-responses)
5. [Cartesia for Text-to-Speech](#cartesia-for-text-to-speech)
6. [Tenacity for Retry Logic](#tenacity-for-retry-logic)
7. [Python-Dotenv for Environment Management](#python-dotenv-for-environment-management)
8. [Integration and System Architecture](#integration-and-system-architecture)
9. [Performance and Security Considerations](#performance-and-security-considerations)

## Introduction
The Maya Bartending Agent is a conversational AI application that enables users to place drink orders via natural language interaction. The system combines a web-based user interface with large language model (LLM) processing and real-time text-to-speech (TTS) synthesis. This document details the core technologies used in the stack, their roles, integration patterns, and performance characteristics. The architecture emphasizes stateless processing, modular design, and robust error handling to ensure reliability and scalability.

**Section sources**
- [main.py](file://main.py#L1-L142)
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Core Frameworks and Libraries
The Maya Bartending Agent leverages several key Python libraries to deliver its functionality. These include Gradio for the user interface, Google Generative AI for natural language understanding and response generation, Cartesia for voice synthesis, tenacity for fault tolerance, and python-dotenv for secure configuration management. Each component plays a distinct role in the system, contributing to a seamless conversational experience.

```mermaid
graph TB
A[User] --> B[Gradio UI]
B --> C[main.py]
C --> D[bartending_agent.py]
D --> E[Google Generative AI]
D --> F[Cartesia TTS]
D --> G[Environment Variables]
G --> H[.env file]
E --> D
F --> D
D --> C
C --> B
```

**Diagram sources**
- [main.py](file://main.py#L1-L142)
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Gradio for UI
Gradio provides the web-based user interface for the Maya Bartending Agent, enabling interactive chat and audio playback. It is used to create a responsive, real-time conversational interface with minimal frontend development overhead.

### Role in Architecture
Gradio acts as the presentation layer, handling user input, displaying chat history, and playing synthesized audio. It manages session state through `gr.State` variables, which store conversation history and order details across interactions.

### Use Case and Implementation
In `main.py`, Gradio is initialized with a structured layout including an avatar image, chatbot display, text input, and audio output components. The `handle_gradio_input` function processes user messages by calling backend logic and returning updated UI elements.

```python
import gradio as gr

def handle_gradio_input(user_input, session_history_state, session_order_state):
    response_text, updated_history, updated_order = process_order(
        user_input, session_history_state, session_order_state
    )
    audio_data = get_voice_audio(response_text) if response_text.strip() else None
    return "", updated_history, updated_history, updated_order, audio_data
```

The interface uses `msg_input.submit()` and `click()` events to trigger processing, ensuring responsiveness and real-time feedback.

### Version and Compatibility
The project requires `gradio>=4.0.0`, which supports modern features like message-type chatbots, improved theming, and enhanced audio handling. The `type="messages"` parameter in `gr.Chatbot` ensures proper formatting of conversation turns.

### Performance Characteristics
Gradio is lightweight and efficient for prototyping and deployment. However, it runs synchronously by default, which can block the UI during long-running operations like API calls. To mitigate this, the application uses stateless backend functions and pre-initializes external clients.

### Limitations
- No native support for asynchronous processing in the main event loop
- Limited customization compared to full-stack frameworks like React
- Audio streaming is disabled (`streaming=False`), requiring full synthesis before playback

### Why Chosen
Gradio was selected for its rapid development capabilities, built-in support for AI applications, and ease of integration with Python-based LLMs and TTS systems. It enables quick iteration and deployment without requiring frontend expertise.

**Section sources**
- [main.py](file://main.py#L1-L142)

## Google Generative AI for LLM-Powered Responses
Google Generative AI powers the conversational intelligence of the bartending agent, generating natural, context-aware responses based on user input and session history.

### Role in Architecture
The Gemini model processes user queries, maintains conversational context, and generates responses that reflect the agent's personality and menu knowledge. It is invoked through the `google.generativeai` SDK with retry-enhanced reliability.

### Use Case and Implementation
In `bartending_agent.py`, the model is initialized once at module load using the API key from environment variables. The `_call_gemini_api` function is decorated with tenacity retry logic to handle transient failures.

```python
import google.generativeai as ggenai

ggenai.configure(api_key=GOOGLE_API_KEY)
model = ggenai.GenerativeModel('gemini-2.0-flash')

@tenacity_retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_gemini_api(prompt_content: List[str], config: Dict):
    return model.generate_content(contents=prompt_content, generation_config=config)
```

The `process_order` function constructs a dynamic prompt including the menu, current order, and conversation history, then sends it to Gemini for response generation.

### Version and Compatibility
The application uses `google-generativeai>=0.3.0`, which supports the Gemini 1.5/2.0 models and structured content generation. The `gemini-2.0-flash` model is specified, though availability should be verified.

### Performance Characteristics
- Low-latency responses suitable for real-time interaction
- Context window supports up to 1M tokens (depending on model)
- Pricing based on input/output tokens encourages concise prompts

### Error Handling
The system checks for blocked prompts, safety filters, and truncation due to token limits. If a response is blocked, the agent provides a user-friendly explanation.

### Security Implications
API keys are loaded from environment variables or `.env` files, preventing hardcoding. The model enforces safety filters to prevent harmful content generation.

### Why Chosen
Gemini was selected for its strong natural language understanding, Google's infrastructure reliability, and seamless integration with other AI services. Its ability to handle multi-turn conversations with context makes it ideal for bartending scenarios.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Cartesia for Text-to-Speech
Cartesia enables voice output for the bartending agent, converting text responses into natural-sounding speech for an immersive user experience.

### Role in Architecture
Cartesia acts as the audio generation layer, transforming LLM responses into WAV audio streams. It is called synchronously after text generation to produce voice feedback.

### Use Case and Implementation
The `get_voice_audio` function in `bartending_agent.py` uses the Cartesia client to synthesize speech. It applies regex to replace "MOK 5-ha" with "Moksha" for correct pronunciation.

```python
from cartesia import Cartesia

cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)

@tenacity_retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
def get_voice_audio(text_to_speak: str) -> bytes | None:
    text_for_tts = re.sub(r'MOK 5-ha', 'Moksha', text_to_speak, flags=re.IGNORECASE)
    audio_generator = cartesia_client.tts.bytes(
        model_id="sonic-2",
        transcript=text_for_tts,
        voice={"mode": "id", "id": CARTESIA_VOICE_ID},
        output_format={"container": "wav", "sample_rate": 24000}
    )
    return b"".join(chunk for chunk in audio_generator)
```

The audio is returned as raw bytes and played in the Gradio interface with `autoplay=True`.

### Version and Compatibility
The project requires `cartesia>=2.0.0`, which supports streaming TTS and high-fidelity voice models like "sonic-2". The voice ID is hardcoded but should be validated.

### Performance Characteristics
- High-quality 24kHz audio output
- Low-latency synthesis suitable for real-time use
- Generator-based API allows memory-efficient streaming

### Limitations
- No fallback voice if the specified ID is invalid
- Synchronous processing may block the UI during long responses
- Requires stable internet connection for API calls

### Security Implications
The API key is securely loaded from environment variables. The system fails gracefully if TTS is unavailable, continuing with text-only responses.

### Why Chosen
Cartesia was selected for its high-quality, expressive voices and developer-friendly API. Its support for real-time audio generation aligns with the conversational nature of the application.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Tenacity for Retry Logic
Tenacity provides robust retry mechanisms for external API calls, improving system resilience against transient network and service failures.

### Role in Architecture
Tenacity wraps calls to Gemini and Cartesia APIs, automatically retrying failed requests with exponential backoff. This reduces error rates and improves user experience.

### Use Case and Implementation
Two retry-decorated functions are defined: `_call_gemini_api` and `get_voice_audio`. Both use exponential backoff with a maximum of three attempts.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_gemini_api(prompt_content: List[str], config: Dict):
    # API call
    pass
```

A fallback decorator is provided if tenacity is not installed, ensuring graceful degradation.

### Version and Compatibility
The project requires `tenacity>=8.2.3`, which includes support for logging retries and custom exception handling. The syntax is stable and widely used.

### Performance Characteristics
- Reduces failure rate during network instability
- Exponential backoff prevents overwhelming services
- Minimal overhead when calls succeed

### Limitations
- Retries increase latency for failed requests
- Not all errors are retryable (e.g., authentication failures)
- Configuration is hardcoded, not configurable at runtime

### Why Chosen
Tenacity was selected for its simplicity, reliability, and extensive configuration options. It integrates seamlessly with Python functions and provides clear logging for debugging.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Python-Dotenv for Environment Management
Python-dotenv manages configuration by loading environment variables from a `.env` file, keeping sensitive data out of the codebase.

### Role in Architecture
It enables secure storage of API keys (Gemini and Cartesia) and other configuration values. The system attempts to load `.env` at startup, falling back to system environment variables.

### Use Case and Implementation
In `bartending_agent.py`, `load_dotenv()` is called to populate `os.environ`:

```python
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
```

If keys are missing, the application raises a fatal error, ensuring secure operation.

### Version and Compatibility
The project requires `python-dotenv>=1.0.0`, which supports modern `.env` parsing and encoding. It is a lightweight, dependency-free library.

### Security Implications
- Prevents accidental exposure of API keys in code or version control
- Encourages separation of configuration and code
- `.env` file should be added to `.gitignore` for production use

### Why Chosen
Python-dotenv is the de facto standard for environment management in Python. It is simple, reliable, and widely supported across development and deployment environments.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Integration and System Architecture
The system follows a modular, stateless design where Gradio handles UI, business logic resides in `bartending_agent.py`, and external APIs provide AI capabilities.

```mermaid
sequenceDiagram
participant User
participant Gradio
participant Logic
participant Gemini
participant Cartesia
User->>Gradio : Submit message
Gradio->>Logic : process_order(user_input, history, order)
Logic->>Gemini : generate_content(prompt)
Gemini-->>Logic : response
Logic->>Cartesia : tts.bytes(transcript)
Cartesia-->>Logic : audio bytes
Logic-->>Gradio : response, history, order, audio
Gradio-->>User : Update chat and play audio
```

**Diagram sources**
- [main.py](file://main.py#L1-L142)
- [bartending_agent.py](file://bartending_agent.py#L1-L374)

## Performance and Security Considerations
The application prioritizes reliability through retry logic, secure credential handling, and graceful error recovery. Performance is optimized by pre-initializing clients and limiting prompt context. Security is enforced via environment variables, safety filters in Gemini, and input validation. External API dependencies are managed through version-pinned requirements and fallback mechanisms.

**Section sources**
- [main.py](file://main.py#L1-L142)
- [bartending_agent.py](file://bartending_agent.py#L1-L374)
- [requirements.txt](file://requirements.txt#L1-L9)
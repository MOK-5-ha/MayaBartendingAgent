# Configuration

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
- [requirements.txt](file://requirements.txt)
</cite>

## Table of Contents
1. [Configuration](#configuration)
2. [Environment Variables and API Keys](#environment-variables-and-api-keys)
3. [Loading Environment Variables with python-dotenv](#loading-environment-variables-with-python-dotenv)
4. [Configuration Parameters for AI and TTS Services](#configuration-parameters-for-ai-and-tts-services)
5. [Accessing Configuration Values in Code](#accessing-configuration-values-in-code)
6. [Setup Instructions for Environment File](#setup-instructions-for-environment-file)
7. [Configuration Across Environments](#configuration-across-environments)
8. [Configuration Validation and Error Handling](#configuration-validation-and-error-handling)

## Environment Variables and API Keys

The application uses environment variables to securely store sensitive credentials such as API keys for external services. This approach prevents hardcoding secrets in source code and allows flexible configuration across different deployment environments.

Two primary API keys are required:

- **GEMINI_API_KEY**: Authentication token for accessing Google's Gemini generative AI service.
- **CARTESIA_API_KEY**: Authentication token for accessing Cartesia's text-to-speech (TTS) service.

These keys are retrieved at runtime using `os.getenv()` and are essential for initializing the respective service clients.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L77)

## Loading Environment Variables with python-dotenv

The application attempts to load environment variables from a `.env` file using the `python-dotenv` library. This enables developers to define environment-specific configurations locally without modifying system-wide settings.

The loading process is implemented as follows:

```python
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("Loaded environment variables from .env file.")
    else:
        print("No .env file found or it is empty.")
except ImportError:
    print("Info: python-dotenv not found. Skipping .env file loading. Relying on system environment variables.")
```

If the `python-dotenv` package is installed, the `load_dotenv()` function reads the `.env` file from the project root and populates the environment. If the package is not available, the application gracefully falls back to relying on system-level environment variables.

The dependency is declared in `requirements.txt`:
```
python-dotenv>=1.0.0
```

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L48-L56)
- [requirements.txt](file://requirements.txt#L3)

## Configuration Parameters for AI and TTS Services

### Gemini AI Model Configuration

The application configures the Gemini generative model with the following parameters:

- **Model Name**: `'gemini-2.0-flash'` — Specifies the version of the Gemini model used for generating responses.
- **Temperature**: `0.7` — Controls response randomness; higher values increase creativity.
- **Max Output Tokens**: `2048` — Limits the length of generated responses.

These settings are passed via a configuration dictionary during the API call:
```python
config_dict = {
    'temperature': 0.7,
    'max_output_tokens': 2048,
}
```

### Cartesia TTS Configuration

The text-to-speech functionality uses Cartesia with the following settings:

- **Model ID**: `"sonic-2"` — Specifies the voice synthesis model.
- **Voice ID**: `"6f84f4b8-58a2-430c-8c79-688dad597532"` — Unique identifier for the selected voice.
- **Language**: `"en"` — Output language for speech synthesis.
- **Output Format**: WAV container with 24kHz sample rate and 32-bit floating-point PCM encoding.

The voice ID is hardcoded but should be replaced with a valid value in production.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L77)
- [bartending_agent.py](file://bartending_agent.py#L110)
- [bartending_agent.py](file://bartending_agent.py#L299-L305)

## Accessing Configuration Values in Code

Configuration values are accessed using Python’s built-in `os.getenv()` function, which retrieves environment variables by name. This method returns `None` if a variable is not set, allowing for safe checks before use.

Examples from the codebase:

```python
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
```

Additionally, default values can be provided:
```python
GEMINI_MODEL_VERSION = os.getenv("GEMINI_MODEL_VERSION", "gemini-2.5-flash-preview-04-17")
```

This pattern is used throughout the application to ensure fallback behavior when optional variables are missing.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L77)
- [bartending_agent.py](file://bartending_agent.py#L439)

## Setup Instructions for Environment File

To configure the application locally, create a `.env` file in the project root directory (``) with the following content:

```
GEMINI_API_KEY=your_gemini_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here
GEMINI_MODEL_VERSION=gemini-2.5-flash-preview-04-17
```

Replace the placeholder values with actual credentials obtained from the respective service providers.

Ensure that the `.env` file is included in `.gitignore` to prevent accidental exposure of secrets in version control.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L48-L56)

## Configuration Across Environments

The configuration strategy supports multiple environments through variable scoping:

- **Development**: Variables are loaded from the local `.env` file.
- **Testing**: CI/CD pipelines should inject environment variables directly into the execution context.
- **Production**: Secrets are managed via platform-specific mechanisms (e.g., Kubernetes secrets, cloud environment variables).

The fallback mechanism ensures that even without `python-dotenv`, the application can operate if environment variables are set at the system level.

No separate configuration files exist for different environments; instead, environment-specific values are injected externally, promoting consistency and security.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L48-L56)

## Configuration Validation and Error Handling

The application performs strict validation of required environment variables at startup. If critical keys are missing, it raises a fatal error and terminates execution.

Validation logic:
```python
if not GOOGLE_API_KEY:
    logger.error("FATAL: GEMINI_API_KEY not found in environment variables or .env file.")
    raise EnvironmentError("GEMINI_API_KEY is required but not found.")

if not CARTESIA_API_KEY:
    logger.error("FATAL: CARTESIA_API_KEY not found in environment variables or .env file.")
    raise EnvironmentError("CARTESIA_API_KEY is required but not found.")
```

This ensures that service clients are only initialized when valid credentials are present, preventing runtime failures due to authentication issues.

Additionally, client initialization is wrapped in try-except blocks to catch and log configuration or connectivity problems early.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L87)
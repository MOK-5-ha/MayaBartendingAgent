# Environment Variables

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
- [requirements.txt](file://requirements.txt)
- [notebooks/mvp_notebook_kaggle.py](file://notebooks/mvp_notebook_kaggle.py)
- [notebooks/submission_notebook.py](file://notebooks/submission_notebook.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose of Environment Variables](#purpose-of-environment-variables)
3. [Key Environment Variables](#key-environment-variables)
4. [Setting Up the .env File](#setting-up-the-env-file)
5. [Loading Environment Variables at Runtime](#loading-environment-variables-at-runtime)
6. [Error Handling for Missing or Invalid Variables](#error-handling-for-missing-or-invalid-variables)
7. [Security Best Practices](#security-best-practices)
8. [Integration with Development and Deployment Workflows](#integration-with-development-and-deployment-workflows)

## Introduction
This document provides a comprehensive overview of environment variable management in the MayaBartendingAgent application. It details how sensitive credentials such as API keys are securely stored and accessed using a `.env` file and the `python-dotenv` library. The configuration supports secure authentication with external services like Google Generative AI (Gemini) and Cartesia Text-to-Speech (TTS), ensuring that secrets are not hardcoded into the source code.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L50-L52)
- [requirements.txt](file://requirements.txt#L3)

## Purpose of Environment Variables
Environment variables are used to store configuration data that may vary between environments (development, testing, production) or contain sensitive information such as API keys. In the MayaBartendingAgent project, they enable secure access to third-party services without exposing credentials in version control.

By externalizing configuration, developers can:
- Keep secrets out of source code
- Easily switch between different configurations
- Support multiple deployment environments
- Comply with security best practices

The application relies on environment variables primarily for service authentication and model configuration.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L77)

## Key Environment Variables
The following environment variables are defined and used within the codebase:

**: GEMINI_API_KEY**
- **Purpose**: Authenticates requests to the Google Generative AI (Gemini) API.
- **Usage**: Passed to `google.generativeai.configure()` to initialize the Gemini client.
- **Source**: `os.getenv("GEMINI_API_KEY")` in `bartending_agent.py`

**: CARTESIA_API_KEY**
- **Purpose**: Authenticates requests to the Cartesia TTS service for voice synthesis.
- **Usage**: Provided when initializing the `Cartesia` client instance.
- **Source**: `os.getenv("CARTESIA_API_KEY")` in `bartending_agent.py`

**: GEMINI_MODEL_VERSION (Optional)**
- **Purpose**: Specifies which Gemini model version to use (e.g., `gemini-2.5-flash-preview-04-17`).
- **Default**: If not set, defaults to `"gemini-2.5-flash-preview-04-17"` in notebook contexts.
- **Source**: Used in Jupyter notebooks like `mvp_notebook_kaggle.py`

These variables are accessed via `os.getenv()` calls throughout the application, allowing runtime flexibility and secure credential handling.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L76-L77)
- [notebooks/mvp_notebook_kaggle.py](file://notebooks/mvp_notebook_kaggle.py#L246-L251)
- [notebooks/mvp_notebook_kaggle.py](file://notebooks/mvp_notebook_kaggle.py#L439)
- [notebooks/submission_notebook.py](file://notebooks/submission_notebook.py#L468)

## Setting Up the .env File
To configure the application locally, create a `.env` file in the root directory of the project (``). This file should contain the required API keys in the following format:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
CARTESIA_API_KEY=your_actual_cartesia_api_key_here
GEMINI_MODEL_VERSION=gemini-2.5-flash-preview-04-17
```

### Step-by-Step Setup Instructions:
1. Navigate to your project root:  
   `cd `
2. Create a new `.env` file:  
   `touch .env`
3. Open the file in a text editor and add your credentials:
   ```env
   GEMINI_API_KEY=AIzaSyABC123...
   CARTESIA_API_KEY=cst_987xyz...
   ```
4. Save and close the file.

Ensure that the `.env` file is never committed to version control by verifying it is listed in `.gitignore`.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L50-L52)

## Loading Environment Variables at Runtime
The application uses the `python-dotenv` library to load environment variables from the `.env` file during startup. The process occurs in `bartending_agent.py` as follows:

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

Once loaded, the values are retrieved using `os.getenv()`:

```python
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
```

These values are then used to configure the respective clients:
- `ggenai.configure(api_key=GOOGLE_API_KEY)`
- `Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))`

This approach ensures that credentials are available at runtime while remaining isolated from the source code.

```mermaid
flowchart TD
A[Application Start] --> B{Is python-dotenv installed?}
B --> |Yes| C[Load .env file with load_dotenv()]
B --> |No| D[Proceed with system environment only]
C --> E{Is .env file present?}
E --> |Yes| F[Load variables into environment]
E --> |No| G[Use system environment variables]
F --> H[Retrieve with os.getenv()]
G --> H
H --> I[Initialize Gemini and Cartesia Clients]
```

**Diagram sources**
- [bartending_agent.py](file://bartending_agent.py#L50-L52)
- [bartending_agent.py](file://bartending_agent.py#L76-L77)

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L50-L52)
- [bartending_agent.py](file://bartending_agent.py#L76-L77)

## Error Handling for Missing or Invalid Variables
The application implements strict validation for required environment variables. If critical keys are missing, the application raises a fatal error and exits.

### Behavior on Missing Variables:
- **GEMINI_API_KEY**:  
  Logs an error and raises `EnvironmentError` if not found.
  ```python
  if not GOOGLE_API_KEY:
      logger.error("FATAL: GEMINI_API_KEY not found...")
      raise EnvironmentError("GEMINI_API_KEY is required but not found.")
  ```

- **CARTESIA_API_KEY**:  
  Similarly checked and raises an error if missing, as TTS functionality is considered essential.

- **Model Initialization Failure**:  
  If the Gemini model fails to initialize (due to invalid key or model name), a `RuntimeError` is raised with diagnostic details.

This strict handling prevents the application from running in an improperly configured state, reducing the risk of runtime failures during user interaction.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L78-L85)

## Security Best Practices
To maintain the security and integrity of API credentials, the following best practices are implemented:

**: Add .env to .gitignore**
Ensure the `.env` file is excluded from version control:
```gitignore
.env
*.env
.env.local
```

**: Use Strong, Unique API Keys**
Generate long-lived, scoped API keys from the respective provider consoles (Google Cloud Console, Cartesia Dashboard).

**: Never Hardcode Secrets**
Avoid embedding API keys directly in Python files or notebooks.

**: Validate Installation of python-dotenv**
The `requirements.txt` includes `python-dotenv>=1.0.0`, ensuring consistent behavior across environments.

**: Provide Clear Setup Instructions**
Document the need for a `.env` file in README or setup guides to assist new developers.

**: Use Environment-Specific Files (Optional)**
For advanced use, consider `.env.development`, `.env.testing`, `.env.production`.

Following these practices ensures that credentials remain secure and the application remains portable across environments.

**Section sources**
- [requirements.txt](file://requirements.txt#L3)
- [bartending_agent.py](file://bartending_agent.py#L50-L52)

## Integration with Development and Deployment Workflows
The environment variable strategy integrates seamlessly across different stages of the software lifecycle:

### Local Development
Developers use a local `.env` file for testing. Gradio UI runs via `main.py`, loading credentials automatically.

### Notebook Testing
Jupyter notebooks (`mvp_notebook_kaggle.ipynb`, etc.) also use `load_dotenv()` and `os.getenv()` to access the same credentials, enabling consistent behavior during prototyping.

### Deployment
In production (e.g., cloud platforms like Vertex AI, Kaggle Kernels, or Hugging Face Spaces), environment variables are set through platform-specific configuration interfaces instead of `.env` files. This maintains security without changing code.

### CI/CD Pipelines
Automated pipelines should inject secrets via secure environment variable injection mechanisms rather than committing `.env` files.

This unified configuration model allows smooth transitions between development, testing, and deployment while maintaining security and consistency.

**Section sources**
- [main.py](file://main.py#L10-L15)
- [notebooks/mvp_notebook_kaggle.py](file://notebooks/mvp_notebook_kaggle.py#L243-L244)
- [bartending_agent.py](file://bartending_agent.py#L50-L52)
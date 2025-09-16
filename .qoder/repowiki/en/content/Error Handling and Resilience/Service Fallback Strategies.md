# Service Fallback Strategies

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Fallback Mechanisms for LLM Responses](#core-fallback-mechanisms-for-llm-responses)
3. [Handling TTS Failures with Graceful Degradation](#handling-tts-failures-with-graceful-degradation)
4. [Maintaining User Experience During Technical Issues](#maintaining-user-experience-during-technical-issues)
5. [Monitoring and Debugging Fallback Triggers](#monitoring-and-debugging-fallback-triggers)

## Introduction
This document details the service fallback strategies implemented in the MayaBartendingAgent system, focusing on resilience when external APIs fail. The system interacts with two critical third-party services: the Gemini Large Language Model (LLM) for generating conversational responses and the Cartesia API for Text-to-Speech (TTS) synthesis. Failures in either service could disrupt the user experience. To ensure robustness, the system employs a layered approach to error handling, including retry mechanisms, fallback messages, and graceful degradation. This documentation explains how the system detects and responds to various failure modes, ensuring that core functionality remains available even under adverse conditions.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L0-L374)
- [main.py](file://main.py#L0-L142)

## Core Fallback Mechanisms for LLM Responses

The `process_order` function in `bartending_agent.py` is responsible for generating responses using the Gemini LLM. It implements comprehensive error handling to manage scenarios where the LLM fails to return a valid response. The function uses the `tenacity` library to automatically retry the API call up to three times with exponential backoff in case of transient network errors or server issues.

When the Gemini API returns a response, the system first checks for the presence of valid candidates. If no candidates are returned, the system logs an error and generates a fallback message. This can occur due to a blocked prompt, which is indicated by the `prompt_feedback` field in the response. In such cases, the system constructs a user-facing message that explains the response was blocked, incorporating the specific reason if available.

```python
if not response.candidates:
     logger.error("Gemini response has no candidates.")
     if response.prompt_feedback and response.prompt_feedback.block_reason:
         logger.error(f"Prompt Blocked: {response.prompt_feedback.block_reason_message}")
         agent_response_text = f"I'm sorry, my ability to respond was blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
     else:
         agent_response_text = "Sorry, I couldn't generate a response. Please try again."
```

If a candidate is present but the content is empty or has no parts, the system examines the `finish_reason` code to determine the cause of the failure. The system interprets several key `finish_reason` codes and triggers appropriate fallback messages:

- **SAFETY**: The response was blocked due to safety filters. The system returns a polite message: "I'm sorry, I can't provide that response due to safety reasons."
- **RECITATION**: The model detected potential issues with reciting copyrighted or sensitive material. The system responds with: "My response couldn't be completed due to potential recitation issues."
- **MAX_TOKENS**: The response was truncated because it reached the maximum token limit. The system attempts to retrieve any partial text and appends an ellipsis and a truncation notice. If no partial text is available, it returns: "My response was cut short as it reached the maximum length."

This structured interpretation of `finish_reason` codes ensures that the system provides meaningful feedback to the user rather than a generic error, maintaining transparency and trust.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L280-L320)

## Handling TTS Failures with Graceful Degradation

The system implements a graceful degradation strategy for its Text-to-Speech functionality. The `get_voice_audio` function in `bartending_agent.py` is responsible for converting the text response into audio using the Cartesia API. This function is also decorated with `tenacity.retry` to handle transient failures, retrying up to three times with exponential backoff.

The key design principle is that a failure in TTS should not break the main conversation flow. The `handle_gradio_input` function in `main.py` calls `get_voice_audio` after receiving a text response from `process_order`. It explicitly checks the return value of `get_voice_audio` and handles the case where it returns `None`.

```python
audio_data = None # Default to None
if response_text and response_text.strip():
     audio_data = get_voice_audio(response_text)
     if audio_data is None:
         logger.warning("Failed to get audio data from get_voice_audio.")
```

If `get_voice_audio` fails—due to an uninitialized client, empty input, or an exception during the API call—it returns `None`. The main flow in `handle_gradio_input` simply logs a warning and continues. The Gradio interface will not play any audio, but the text response is still displayed in the chatbot. This ensures that the core functionality of the conversation is preserved. The user can still read the bartender's response, and the interaction can continue without interruption. This conditional logic is crucial for system resilience, as it decouples the availability of the voice feature from the availability of the text-based service.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L330-L374)
- [main.py](file://main.py#L30-L45)

## Maintaining User Experience During Technical Issues

A critical aspect of the fallback strategy is maintaining the bartender persona and user engagement, even when technical issues occur. All fallback messages are crafted to be polite, helpful, and consistent with the character of a friendly bartender at "MOK 5-ha" (Moksha).

For example, instead of a technical message like "API Error 500," the system uses empathetic language such as "I'm sorry, an unexpected error occurred. Please try again later." This message is returned from the `process_order` function's outer exception handler, which catches any unhandled errors during processing. This ensures that even a catastrophic failure in the logic does not expose raw exceptions to the user.

The fallback messages for LLM-specific issues are also designed to be informative without being overly technical. By mentioning "safety reasons" or "maximum length," the system acknowledges the limitation in a way that is understandable to a non-technical user. The use of phrases like "Could you rephrase?" encourages the user to continue the conversation, preserving engagement.

Furthermore, the system maintains state integrity during failures. In the event of a critical error in `process_order`, the function returns a safe error message while preserving the original session history and order state. This prevents the error from corrupting the user's session, allowing them to recover seamlessly once the issue is resolved.

```python
safe_history = current_session_history[:]
safe_history.append({'role': 'user', 'content': user_input_text})
safe_history.append({'role': 'assistant', 'content': error_message})
return error_message, safe_history, current_session_order
```

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L300-L315)

## Monitoring and Debugging Fallback Triggers

The system is instrumented with comprehensive logging to facilitate monitoring and debugging of fallback scenarios. The `logging` module is configured at the beginning of both `main.py` and `bartending_agent.py` to capture events at the INFO level and higher, with detailed timestamps and source information.

Every potential failure point is logged with an appropriate severity level:
- **INFO**: Successful operations, such as initializing the Gemini model or receiving a valid response.
- **WARNING**: Recoverable failures, such as an empty text input to TTS or a failed TTS call.
- **ERROR**: More serious issues, like a missing API key or a response with no candidates.
- **EXCEPTION**: Critical errors with a full stack trace, such as an unhandled exception in `process_order`.

This logging strategy allows developers to monitor the system's health and identify when fallbacks are being triggered frequently. For instance, a high volume of "Gemini response has no candidates" ERROR logs could indicate a problem with the prompt structure or the LLM's safety filters. Similarly, frequent WARNING logs from `get_voice_audio` could point to issues with the Cartesia API or network connectivity.

By analyzing these logs, the team can proactively address underlying issues, tune retry parameters, or refine fallback messages to improve the overall user experience and system reliability.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L50-L75)
- [bartending_agent.py](file://bartending_agent.py#L280-L320)
- [bartending_agent.py](file://bartending_agent.py#L350-L370)
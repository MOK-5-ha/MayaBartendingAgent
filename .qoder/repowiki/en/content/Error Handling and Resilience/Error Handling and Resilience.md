# Error Handling and Resilience

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
</cite>

## Table of Contents
1. [Error Handling and Resilience](#error-handling-and-resilience)
2. [Retry Mechanisms with Tenacity](#retry-mechanisms-with-tenacity)
3. [Exception Handling Patterns](#exception-handling-patterns)
4. [Fallback Strategies and Graceful Degradation](#fallback-strategies-and-graceful-degradation)
5. [User-Facing Error Messaging](#user-facing-error-messaging)
6. [Monitoring and Retry Configuration](#monitoring-and-retry-configuration)
7. [Conclusion](#conclusion)

## Retry Mechanisms with Tenacity

The bartending agent employs the **tenacity** library to implement robust retry logic for external API calls, specifically targeting interactions with the Gemini and Cartesia services. This ensures the system can recover from transient failures such as network hiccups, temporary service outages, or rate limiting.

Two primary functions are decorated with retry logic:
- `_call_gemini_api`: Handles communication with the Google Gemini LLM service.
- `get_voice_audio`: Manages text-to-speech synthesis via the Cartesia API.

The retry configuration uses an exponential backoff strategy, which increases the wait time between attempts to avoid overwhelming the service during periods of instability. The specific parameters are:
- **Maximum Attempts**: 3 retries before giving up.
- **Wait Strategy**: `wait_exponential` with a multiplier of 1, a minimum wait of 2 seconds, and a maximum of 10 seconds for Gemini, and 1 to 5 seconds for Cartesia.

```python
@tenacity_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_gemini_api(prompt_content: List[str], config: Dict) -> ggenai.types.GenerateContentResponse:
    # API call implementation
    pass
```

A defensive programming pattern is used to handle the case where the `tenacity` library is not installed. If the import fails, a dummy `tenacity_retry` decorator is defined, allowing the code to run without retry functionality while logging a warning. This prevents a hard failure of the entire application due to a missing dependency.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L21-L38)
- [bartending_agent.py](file://bartending_agent.py#L147-L157)
- [bartending_agent.py](file://bartending_agent.py#L312-L320)

## Exception Handling Patterns

The agent implements a layered exception handling strategy to manage various failure modes, including network failures, authentication errors, and invalid responses.

### Network and API Failures
For network-related issues, the agent defines sets of retryable exceptions. For Cartesia TTS, it explicitly retries on `ConnectionError` and `TimeoutError`. For Gemini, the code includes a commented-out section suggesting the use of specific Google API exceptions like `ResourceExhaustedError` and `ServiceUnavailableError`, indicating a planned, more granular approach to error handling.

### Authentication and Initialization Errors
Critical errors are handled during the application's startup phase in `main.py`. If the `bartending_agent` module cannot be imported, or if the required API keys (`GEMINI_API_KEY`, `CARTESIA_API_KEY`) are missing from the environment, the application logs a fatal error and exits using `SystemExit`. This ensures the agent does not run in a misconfigured state.

```python
if not GOOGLE_API_KEY:
    logger.error("FATAL: GEMINI_API_KEY not found in environment variables or .env file.")
    raise EnvironmentError("GEMINI_API_KEY is required but not found.")
```

### Invalid and Blocked Responses
The `_call_gemini_api` function includes logic to handle cases where the LLM returns an invalid or blocked response. If the response has no candidates, the agent checks the `prompt_feedback.block_reason` to determine if the input was blocked for safety reasons and crafts an appropriate user message. It also handles specific `finish_reason` codes like `SAFETY`, `RECITATION`, and `MAX_TOKENS` to provide meaningful feedback when the model cannot generate a response.

```python
if not response.candidates:
    logger.error("Gemini response has no candidates.")
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        agent_response_text = f"I'm sorry, my ability to respond was blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
```

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L90-L115)
- [bartending_agent.py](file://bartending_agent.py#L265-L285)
- [main.py](file://main.py#L15-L25)

## Fallback Strategies and Graceful Degradation

The system is designed with multiple fallback mechanisms to maintain core functionality even when components fail.

### Default Responses and State Preservation
When an unexpected error occurs within the `process_order` function, a broad `except Exception as e` block catches the error, logs it, and returns a safe, generic error message to the user. Crucially, it preserves the current session state by reverting to the original `current_session_history` and `current_session_order`, preventing the corruption of the conversation flow.

```python
except Exception as e:
    logger.exception(f"Critical error in process_order: {str(e)}")
    error_message = "I'm sorry, an unexpected error occurred. Please try again later."
    safe_history = current_session_history[:]
    safe_history.append({'role': 'user', 'content': user_input_text})
    safe_history.append({'role': 'assistant', 'content': error_message})
    return error_message, safe_history, current_session_order
```

### Graceful TTS Degradation
The system gracefully degrades when the TTS service fails. In the `handle_gradio_input` function in `main.py`, if `get_voice_audio` returns `None`, a warning is logged, but the text response is still delivered to the user. The core conversation functionality remains fully operational, ensuring the user experience is not completely broken by a single component failure.

```python
if response_text and response_text.strip():
     audio_data = get_voice_audio(response_text)
     if audio_data is None:
         logger.warning("Failed to get audio data from get_voice_audio.")
```

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L289-L307)
- [main.py](file://main.py#L48-L52)

## User-Facing Error Messaging

The agent prioritizes maintaining its bartender persona even when communicating technical issues. Error messages are crafted to be polite, helpful, and on-brand.

- **Technical Failures**: Messages like "I'm sorry, an unexpected error occurred. Please try again later." are professional and apologetic.
- **Safety Blocks**: When a prompt is blocked, the agent explains the reason in a neutral way: "I'm sorry, my ability to respond was blocked. Reason: [block reason]."
- **Service Unavailability**: For TTS failures, while the user is not directly notified, the system continues to provide the text response, implying the core service is still available.

This approach ensures that the user is informed of a problem without being exposed to raw technical details, preserving the illusion of a competent and friendly bartender.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L270-L275)
- [bartending_agent.py](file://bartending_agent.py#L300-L302)

## Monitoring and Retry Configuration

The system is instrumented with comprehensive logging to facilitate monitoring and debugging.

### Logging Practices
The `logging` module is used extensively throughout the codebase. The `before_sleep_log` parameter in the `tenacity_retry` decorator automatically logs a warning message before each retry attempt, providing visibility into API instability. Other log levels are used appropriately:
- `INFO`: For general operational milestones (e.g., "Successfully initialized Gemini model").
- `WARNING`: For non-critical failures (e.g., TTS failure, empty input).
- `ERROR` and `EXCEPTION`: For critical failures, with `exception` used to log the full stack trace.

### Configuring Retry Budgets
The retry budget is currently hardcoded with a maximum of 3 attempts. This can be adjusted by modifying the `stop_after_attempt(3)` parameter in the `@tenacity_retry` decorators. The exponential backoff parameters (`min`, `max`) can also be tuned based on the specific requirements and tolerance for latency. For production use, these values could be moved to a configuration file for easier management.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L154)
- [bartending_agent.py](file://bartending_agent.py#L1-L10)

## Conclusion

The Maya Bartending Agent demonstrates a well-structured approach to error handling and resilience. By leveraging the `tenacity` library for retries, implementing comprehensive exception handling for various failure modes, and employing graceful degradation strategies, the system ensures high availability and a smooth user experience. The focus on user-friendly error messaging maintains the agent's persona, while detailed logging provides the necessary observability for monitoring and maintenance. This robust error handling framework is critical for a reliable AI-powered conversational agent.
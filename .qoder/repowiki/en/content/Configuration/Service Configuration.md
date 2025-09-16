# Service Configuration

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
</cite>

## Table of Contents
1. [Service Configuration](#service-configuration)
2. [Gemini LLM Configuration](#gemini-llm-configuration)
3. [TTS Configuration with Cartesia API](#tts-configuration-with-cartesia-api)
4. [Configuration Validation and Error Handling](#configuration-validation-and-error-handling)
5. [Performance Implications and Tuning Recommendations](#performance-implications-and-tuning-recommendations)

## Gemini LLM Configuration

The MayaBartendingAgent integrates with Google's Gemini LLM through the `google.generativeai` library. The configuration for the generative model is defined within the `process_order` function in `bartending_agent.py`. This configuration directly influences the behavior of the AI bartender in terms of creativity, response length, and safety.

The primary configuration parameters for the Gemini model are set in a dictionary passed to the `generate_content` method. The key settings are:

- **Model Selection**: The agent uses the `gemini-2.0-flash` model, which is configured at the module level. This model is chosen for its balance of speed and capability, suitable for real-time conversational interactions.
- **Temperature**: Set to `0.7`, this value controls the randomness of the model's output. A higher temperature leads to more creative and varied responses, while a lower value makes the output more deterministic and focused. The value of `0.7` provides a conversational tone that is friendly and engaging without being overly random.
- **Max Output Tokens**: Set to `2048`, this parameter limits the length of the generated response. This ensures that responses are concise and prevents the model from generating excessively long outputs, which could increase latency and cost.

```python
config_dict = {
    'temperature': 0.7,
    'max_output_tokens': 2048,
}
response = _call_gemini_api(prompt_content=[full_prompt], config=config_dict)
```

These parameters are injected directly into the `generate_content` call. The configuration has a direct impact on response quality, latency, and cost. A higher `max_output_tokens` value can improve the completeness of responses but increases processing time and token usage, which affects cost. The `temperature` setting affects the perceived personality of the agent; a value too high might lead to irrelevant or off-topic responses, while a value too low might make the agent seem robotic.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L250-L255)

## TTS Configuration with Cartesia API

The text-to-speech (TTS) functionality is powered by the Cartesia API, which converts the agent's textual responses into spoken audio. The TTS configuration is managed within the `get_voice_audio` function in `bartending_agent.py`.

Key TTS-specific settings include:

- **Voice ID**: The voice used for audio synthesis is specified by the `CARTESIA_VOICE_ID` constant, which is set to `6f84f4b8-58a2-430c-8c79-688dad597532`. This ID corresponds to a specific voice model available in the Cartesia service. The voice ID can be customized to change the agent's vocal characteristics.
- **Model ID**: The TTS model used is `sonic-2`, which is specified in the `tts.bytes` method call. This model is optimized for high-quality, natural-sounding speech synthesis.
- **Output Audio Format**: The audio output is configured to be in WAV format with a sample rate of `24000` Hz and `pcm_f32le` encoding. This ensures high audio quality suitable for web playback.

```python
audio_generator = cartesia_client.tts.bytes(
    model_id="sonic-2",
    transcript=text_for_tts,
    voice={"mode":"id", "id": CARTESIA_VOICE_ID},
    language="en",
    output_format={"container":"wav", "sample_rate": 24000, "encoding": "pcm_f32le"},
)
```

The configuration is passed directly to the Cartesia client's `tts.bytes` method. The choice of audio format and sample rate affects both the quality and size of the generated audio. Higher sample rates improve audio fidelity but increase file size and bandwidth usage. The `pcm_f32le` encoding provides high-quality audio suitable for professional applications.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L335-L350)

## Configuration Validation and Error Handling

The system implements robust configuration validation and error handling to ensure reliable operation. Configuration parameters are validated at both initialization and runtime.

During initialization, the presence of required API keys is checked. If `GEMINI_API_KEY` or `CARTESIA_API_KEY` is missing from environment variables or a `.env` file, the application raises a `RuntimeError` with a descriptive message, preventing startup with incomplete configuration.

```python
if not GOOGLE_API_KEY:
    logger.error("FATAL: GEMINI_API_KEY not found in environment variables or .env file.")
    raise EnvironmentError("GEMINI_API_KEY is required but not found.")
```

For the Gemini API, the response is validated for candidates and content. If the response is blocked due to safety reasons or lacks content, the agent provides a user-friendly error message. The `finish_reason` is checked to handle cases like `MAX_TOKENS`, where the response is truncated, allowing the system to inform the user appropriately.

For TTS, the `get_voice_audio` function includes retry logic using the `tenacity` library, configured to retry up to three times with exponential backoff for network-related exceptions. This improves resilience against transient API failures.

```python
@tenacity_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(CARTESIA_RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING) if callable(before_sleep_log) else None,
    reraise=True
)
```

If TTS generation fails, the function returns `None`, and the Gradio interface handles this by not playing any audio, ensuring a graceful degradation of service.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L65-L85)
- [bartending_agent.py](file://bartending_agent.py#L265-L295)
- [bartending_agent.py](file://bartending_agent.py#L320-L330)

## Performance Implications and Tuning Recommendations

The configuration parameters have significant performance implications. For the Gemini LLM, a higher `temperature` increases response variability but may reduce coherence, while a higher `max_output_tokens` increases latency and cost. For TTS, a higher sample rate improves audio quality but increases bandwidth and processing time.

**Tuning Recommendations:**
- For **low-latency applications**, consider reducing `max_output_tokens` to `1024` and using a faster model like `gemini-1.5-flash`.
- For **high-quality voice output**, maintain the current `24000` Hz sample rate, but consider compressing the audio to `mp3` format for reduced bandwidth if needed.
- For **cost-sensitive deployments**, monitor token usage and adjust `temperature` and `max_output_tokens` to minimize unnecessary generation.
- For **reliability**, ensure retry logic is enabled for both LLM and TTS calls to handle transient network issues.

The current default values provide a balanced experience suitable for a conversational bartender agent, prioritizing natural interaction and audio quality.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L250-L255)
- [bartending_agent.py](file://bartending_agent.py#L335-L350)
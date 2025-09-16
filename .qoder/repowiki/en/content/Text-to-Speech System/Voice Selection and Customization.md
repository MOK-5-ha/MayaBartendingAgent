# Voice Selection and Customization

<cite>
**Referenced Files in This Document**   
- [bartending_agent.py](file://bartending_agent.py)
- [main.py](file://main.py)
</cite>

## Table of Contents
1. [Voice Selection and Customization](#voice-selection-and-customization)
2. [Voice Configuration in the Cartesia API](#voice-configuration-in-the-cartesia-api)
3. [Current Voice Implementation in Maya's Bartender Persona](#current-voice-implementation-in-mayas-bartender-persona)
4. [Code Analysis: Voice Handling in bartending_agent.py](#code-analysis-voice-handling-in-bartending_agentpy)
5. [Strategies for Contextual Voice Adaptation](#strategies-for-contextual-voice-adaptation)
6. [Pronunciation Handling for Complex Drink Names](#pronunciation-handling-for-complex-drink-names)
7. [Testing and Selecting Optimal Voices](#testing-and-selecting-optimal-voices)
8. [Maintaining Brand Consistency Through Vocal Identity](#maintaining-brand-consistency-through-vocal-identity)

## Voice Configuration in the Cartesia API

The text-to-speech (TTS) system in the Maya bartending agent leverages the **Cartesia API** to generate natural-sounding voice output. Voice characteristics such as tone, accent, and speed are controlled through specific API parameters.

Key configuration parameters include:

- **voice_id**: A unique identifier specifying the vocal profile (timbre, gender, age, accent).
- **model_id**: The underlying AI model used for synthesis (e.g., "sonic-2").
- **language**: Language code (e.g., "en" for English).
- **output_format**: Specifies audio container, sample rate, and encoding.
  - container: "wav"
  - sample_rate: 24000 Hz
  - encoding: "pcm_f32le"

The Cartesia API allows fine-grained control over voice characteristics, enabling customization to match brand personality. For example, a warm, mid-pitched female voice with a neutral American accent can be selected to convey approachability and professionalism in a bar setting.

```python
voice={"mode":"id", "id": CARTESIA_VOICE_ID}
```

This configuration ensures consistent vocal identity across interactions.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L342-L345)

## Current Voice Implementation in Maya's Bartender Persona

The current implementation uses a hardcoded voice configuration designed to reflect Maya’s persona as a friendly, efficient, and knowledgeable bartender at **MOK 5-ha** (pronounced "Moksha").

### Voice Characteristics
- **Voice ID**: `6f84f4b8-58a2-430c-8c79-688dad597532`
- **Model**: "sonic-2"
- **Language**: English (en)
- **Output Format**: WAV at 24kHz, 32-bit float PCM

This voice profile contributes to the bartender persona by:
- Using a **warm, conversational tone** to create a welcoming atmosphere.
- Maintaining **clear articulation** for accurate drink name pronunciation.
- Delivering responses with **moderate pacing**, balancing friendliness with efficiency.

The voice is designed to be engaging without being overly theatrical, aligning with a modern, upscale bar environment.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L104-L107)
- [bartending_agent.py](file://bartending_agent.py#L335-L347)

## Code Analysis: Voice Handling in bartending_agent.py

The voice synthesis functionality is encapsulated in the `get_voice_audio()` function within `bartending_agent.py`. This function is responsible for converting the agent’s text response into audio using the Cartesia TTS service.

### Key Implementation Details

```python
@tenacity_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(CARTESIA_RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def get_voice_audio(text_to_speak: str) -> bytes | None:
    global cartesia_client, CARTESIA_VOICE_ID

    if not text_to_speak or not text_to_speak.strip():
        return None
    if not cartesia_client or not CARTESIA_VOICE_ID:
        return None

    # Replace 'MOK 5-ha' with 'Moksha' for correct pronunciation
    text_for_tts = re.sub(r'MOK 5-ha', 'Moksha', text_to_speak, flags=re.IGNORECASE)

    audio_generator = cartesia_client.tts.bytes(
        model_id="sonic-2",
        transcript=text_for_tts,
        voice={"mode":"id", "id": CARTESIA_VOICE_ID},
        language="en",
        output_format={
            "container":"wav",
            "sample_rate": 24000,
            "encoding": "pcm_f32le"
        },
    )

    audio_data = b"".join(chunk for chunk in audio_generator)
    return audio_data
```

### Observations
- **Hardcoded Voice ID**: The `CARTESIA_VOICE_ID` is currently hardcoded, limiting dynamic voice selection.
- **Retry Logic**: The function uses `tenacity` to retry failed TTS requests up to 3 times with exponential backoff.
- **Pronunciation Preprocessing**: The function replaces "MOK 5-ha" with "Moksha" to ensure correct pronunciation.
- **Synchronous Processing**: Audio is generated synchronously, blocking until complete.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L320-L355)

## Strategies for Contextual Voice Adaptation

To enhance user experience, the voice style can be dynamically adapted based on conversation context. While not currently implemented, potential strategies include:

### 1. **Tone Modulation by Context**
| Context | Recommended Tone | Rationale |
|-------|------------------|---------|
| Drink Recommendations | Upbeat, enthusiastic | Encourages exploration and engagement |
| Order Confirmation | Clear, neutral | Ensures accuracy and professionalism |
| Billing/Checkout | Formal, precise | Conveys trust and reliability |
| Greetings | Warm, friendly | Creates welcoming first impression |

### 2. **Speed Adjustment**
- **Faster speech**: For routine confirmations (e.g., "Got it, one Margarita coming up!")
- **Slower speech**: For complex explanations (e.g., describing Moksha philosophy)

### 3. **Implementation Approach**
A context-aware voice selector could be implemented as:

```python
def select_voice_profile(context: str) -> dict:
    profiles = {
        "greeting": {"voice_id": "warm-female-1", "speed": 1.0},
        "recommendation": {"voice_id": "energetic-female-2", "speed": 1.1},
        "confirmation": {"voice_id": "clear-neutral-3", "speed": 1.0},
        "billing": {"voice_id": "formal-male-4", "speed": 0.9}
    }
    return profiles.get(context, profiles["confirmation"])
```

This would require modifying `get_voice_audio()` to accept a context parameter.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L320-L355)

## Pronunciation Handling for Complex Drink Names

Accurate pronunciation of cocktail names is critical for user trust and brand credibility. The system currently handles this through **text preprocessing** before TTS synthesis.

### Current Implementation
- **Regex Substitution**: The name "MOK 5-ha" is replaced with "Moksha" to ensure correct pronunciation.
- **Case-Insensitive Matching**: The replacement is performed regardless of input case.

```python
text_for_tts = re.sub(r'MOK 5-ha', 'Moksha', text_to_speak, flags=re.IGNORECASE)
```

### Enhancement Opportunities
1. **Phonetic Hints via SSML** (if supported by Cartesia):
   ```xml
   <speak>
     One <phoneme alphabet="ipa" ph="mɒkˈʃɑː">Moksha</phoneme> cocktail.
   </speak>
   ```
2. **Pronunciation Dictionary**:
   ```python
   PRONUNCIATION_MAP = {
       "Negroni": "nəˈɡroʊni",
       "Daiquiri": "daɪˈkɪri",
       "Manhattan": "mænˈhætən"
   }
   ```
3. **Pre-recorded Audio Clips** for frequently ordered drinks.

These enhancements would improve clarity, especially for non-native speakers or complex names.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L334-L336)

## Testing and Selecting Optimal Voices

Selecting the right voice involves both technical evaluation and user experience testing.

### Evaluation Criteria
- **Clarity**: Can users understand every word?
- **Naturalness**: Does the voice sound human-like?
- **Engagement**: Does the voice hold attention?
- **Brand Alignment**: Does the voice match MOK 5-ha’s upscale, spiritual theme?

### Testing Methodology
1. **A/B Testing**: Present users with two voice variants and collect preference feedback.
2. **Task Success Rate**: Measure how often users correctly understand drink names and prices.
3. **Perceived Friendliness**: Use surveys to rate the bartender’s warmth and approachability.
4. **Audio Quality Metrics**: Evaluate signal-to-noise ratio, prosody, and intonation.

### Recommended Workflow
1. Generate samples using multiple Cartesia voices.
2. Conduct internal review with stakeholders.
3. Perform user testing in a controlled environment.
4. Deploy the winning voice with monitoring.

The current voice ID (`6f84f4b8-58a2-430c-8c79-688dad597532`) should be evaluated against alternatives to ensure it remains optimal.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L104)
- [bartending_agent.py](file://bartending_agent.py#L335-L347)

## Maintaining Brand Consistency Through Vocal Identity

A consistent vocal identity reinforces brand recognition and trust.

### Brand Voice Guidelines for Maya
- **Tone**: Friendly, knowledgeable, slightly formal
- **Pace**: Moderate (not rushed, not slow)
- **Pitch**: Mid-range, warm timbre
- **Accent**: Neutral American English
- **Personality**: Attentive, professional, with subtle warmth

### Implementation Recommendations
1. **Document the Voice Profile**: Create a style guide for the bartender’s vocal characteristics.
2. **Centralize Voice Configuration**: Move `CARTESIA_VOICE_ID` to a config file or environment variable.
3. **Version Control**: Track voice changes as part of deployment.
4. **Fallback Mechanism**: Define a default voice if the primary fails.

### Configuration Example
```env
# .env
CARTESIA_VOICE_ID=6f84f4b8-58a2-430c-8c79-688dad597532
CARTESIA_MODEL_ID=sonic-2
CARTESIA_SAMPLE_RATE=24000
```

This ensures consistency across environments and simplifies updates.

**Section sources**
- [bartending_agent.py](file://bartending_agent.py#L104)
- [bartending_agent.py](file://bartending_agent.py#L335-L347)
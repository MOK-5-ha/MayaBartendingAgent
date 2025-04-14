# bartending_agent.py (Stateless Version for Gradio Session State)
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

# Gemini - Frontier LLM
try:
    # Using 'ggenai' alias consistent with user's snippets
    import google.generativeai as ggenai
    from google.api_core import retry as core_retry # For potential core retries
    from google.generativeai import types as genai_types # For specific types if needed later
except ImportError:
    print("Error: google.generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)

# Tenacity for retries on specific functions
try:
    from tenacity import (
        retry as tenacity_retry, # Alias to avoid confusion with google.api_core.retry
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log
    )
except ImportError:
    print("Warning: tenacity library not found. Retries on API calls will not be enabled.")
    print("Install it using: pip install tenacity")
    # Define a dummy decorator if tenacity is missing
    def tenacity_retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    RETRYABLE_EXCEPTIONS = (Exception,) # Fallback to generic exception
    before_sleep_log = lambda logger, level: None # Dummy function
#else:
    # Define specific exceptions for tenacity retry relevant to API calls
    #RETRYABLE_EXCEPTIONS = (
        #genai_errors.ResourceExhaustedError,
        #genai_errors.InternalServerError,
        #genai_errors.ServiceUnavailableError,
        #Add other potentially transient network errors if needed, e.g., ConnectionError
        #ConnectionError, # Be cautious with retrying generic connection errors
#    )


# Attempt to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("Loaded environment variables from .env file.")
    else:
        print("No .env file found or it is empty.")
except ImportError:
    print("Info: python-dotenv not found. Skipping .env file loading. Relying on system environment variables.")

try:
    from cartesia import Cartesia
    from cartesia.tts import OutputFormat_Raw, TtsRequestIdSpecifier
except ImportError:
    print("Error: Cartesia library not found.")
    print("Please ensure it's listed in requirements.txt and installed.")
    # Decide if this should be fatal - probably yes if TTS is required
    sys.exit(1)


# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API Key (Ensure this is set in your .env file or system environment)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY") # Load Cartesia key

if not GOOGLE_API_KEY:
    logger.error("FATAL: GEMINI_API_KEY not found in environment variables or .env file.")
    raise EnvironmentError("GEMINI_API_KEY is required but not found.")

if not CARTESIA_API_KEY:
    logger.error("FATAL: CARTESIA_API_KEY not found in environment variables or .env file.")
    # Decide if TTS is optional or required. Assuming required for now.
    raise EnvironmentError("CARTESIA_API_KEY is required but not found.")

# Configure Gemini Client and Model (Initialized ONCE at module load)
try:
    ggenai.configure(api_key=GOOGLE_API_KEY)
    # Use a valid and available model name, e.g., 'gemini-1.5-flash' or 'gemini-pro'
    MODEL_NAME = 'gemini-2.0-flash' # Verify this model name is correct and accessible
    model = ggenai.GenerativeModel(MODEL_NAME)
    logger.info(f"Successfully initialized Gemini model: {MODEL_NAME}")
except Exception as e:
    logger.exception(f"Fatal: Failed to initialize Gemini model: {str(e)}")
    raise RuntimeError(
        f"Failed to initialize Gemini model. Check API key and model name ('{MODEL_NAME}')."
    ) from e

# Initialize Cartesia Client (ONCE at module load)
try:
    # Replace "your-chosen-voice-id" with an actual valid ID from Cartesia
    CARTESIA_VOICE_ID = "6f84f4b8-58a2-430c-8c79-688dad597532" # Example placeholder ID - CHANGE THIS
    if not CARTESIA_VOICE_ID or "your-chosen-voice-id" in CARTESIA_VOICE_ID:
         logger.warning("CARTESIA_VOICE_ID is not set to a valid ID. Please edit bartending_agent.py.")
         # Decide if this is fatal. Maybe proceed without voice for now?

    cartesia_client = Cartesia(
        api_key=os.getenv("CARTESIA_API_KEY"),
        )
    logger.info("Successfully initialized Cartesia client.")
    # Optional: Could add a check here to verify the voice ID exists using the client if possible
except Exception as e:
     logger.exception("Fatal: Failed to initialize Cartesia client.")
     raise RuntimeError("Cartesia client initialization failed.") from e


# --- Static Data ---
# Define the Menu (Doesn't change per session)
menu: Dict[str, Dict[str, float]] = {
    "1": {"name": "Old Fashioned", "price": 12.00},
    "2": {"name": "Margarita", "price": 10.00},
    "3": {"name": "Mojito", "price": 11.00},
    "4": {"name": "Martini", "price": 13.00},
    "5": {"name": "Whiskey Sour", "price": 11.00},
    "6": {"name": "Gin and Tonic", "price": 9.00},
    "7": {"name": "Manhattan", "price": 12.00},
    "8": {"name": "Daiquiri", "price": 10.00},
    "9": {"name": "Negroni", "price": 11.00},
    "10": {"name": "Cosmopolitan", "price": 12.00}
}
# NO global history/order variables needed here - state is passed in/out


# --- Core Agent Logic (Stateless Functions) ---

def get_menu_text() -> str:
    """Generates the menu text (Stateless)."""
    global menu # Access the global menu variable
    menu_text = "Menu:\n" + "-"*5 + "\n"
    for item_id, item in menu.items():
        menu_text += f"{item_id}. {item['name']} - ${item['price']:.2f}\n"
    return menu_text


@tenacity_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    #retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING) if callable(before_sleep_log) else None, # Check if callable
    reraise=True # Re-raise the exception if all retries fail
)
def _call_gemini_api(prompt_content: List[str], config: Dict) -> ggenai.types.GenerateContentResponse:
    """Internal function to call the Gemini API with retry logic (Stateless)."""
    logger.debug("Calling Gemini API...")
    # Uses the globally initialized 'model'
    response = model.generate_content(
        contents=prompt_content, # Correct parameter name is 'contents'
        generation_config=config,
        # safety_settings can be added here if needed
    )
    logger.debug("Gemini API call successful.")
    return response


def process_order(
    user_input_text: str,
    current_session_history: List[Dict[str, str]],
    current_session_order: List[Dict[str, float]]
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, float]]]:
    """
    Processes user input using Gemini, updates state for the CURRENT SESSION.
    Accepts session history and order, returns (response_text, updated_history, updated_order).
    """
    global menu # Allow access to the global menu

    if not user_input_text:
        logger.warning("Received empty user input.")
        # Return current state unchanged with a message
        return "Please tell me what you'd like to order.", current_session_history, current_session_order

    # Local copies for modification within this function call - ensures statelessness
    # Regarding the input arguments. The returned values become the new state.
    updated_history = current_session_history[:]
    updated_order = current_session_order[:]

    try:
        # --- Construct the prompt using session-specific history/order ---
        prompt_context = [
            "You are a friendly and helpful bartender taking drink orders.",
            "Be conversational. Ask clarifying questions if the order is unclear.",
            "If the user asks for something not on the menu, politely tell them and show the menu again.",
            "If the user asks to see their current order, list the items and their prices.",
            "\nHere is the menu:",
            get_menu_text(), # Call the stateless menu function
            "\nCurrent order:",
        ]
        if updated_order: # Use the passed-in order state copy
            order_text = "\n".join([f"- {item['name']} (${item['price']:.2f})" for item in updated_order])
            prompt_context.append(order_text)
        else:
            prompt_context.append("No items ordered yet.")

        prompt_context.append("\nConversation History (latest turns):")
        history_limit = 10 # Keep the last ~5 pairs of interactions
        limited_history_for_prompt = updated_history[-history_limit:] # Use passed-in history state copy

        for entry in limited_history_for_prompt:
             role = entry.get("role", "unknown").capitalize()
             content = entry.get("content", "")
             prompt_context.append(f"{role}: {content}")

        # Add the current user input to the prompt context
        prompt_context.append(f"\nUser: {user_input_text}")
        prompt_context.append("\nBartender:") # Ask the model to reply as the bartender

        full_prompt = "\n".join(prompt_context)
        logger.info(f"Processing user input for session: {user_input_text}")
        logger.debug(f"Full prompt for Gemini:\n------\n{full_prompt}\n------")

        # --- Call Gemini API via the retry wrapper ---
        config_dict = {
            'temperature': 0.7,
            'max_output_tokens': 2048,
            # 'candidate_count': 1 # Usually defaults to 1
        }
        response = _call_gemini_api(prompt_content=[full_prompt], config=config_dict)

        # --- Process the response ---
        agent_response_text = "" # Default empty response

        # Check response validity and safety
        if not response.candidates:
             logger.error("Gemini response has no candidates.")
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 logger.error(f"Prompt Blocked: {response.prompt_feedback.block_reason_message}")
                 agent_response_text = f"I'm sorry, my ability to respond was blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
             else:
                 agent_response_text = "Sorry, I couldn't generate a response. Please try again."

        elif not response.candidates[0].content or not response.candidates[0].content.parts:
             logger.error("Gemini response candidate is empty or has no parts.")
             finish_reason = response.candidates[0].finish_reason
             finish_reason_name = finish_reason.name if finish_reason else 'UNKNOWN'
             logger.error(f"Finish Reason: {finish_reason_name}")

             if finish_reason_name == "SAFETY":
                 agent_response_text = "I'm sorry, I can't provide that response due to safety reasons."
             elif finish_reason_name == "RECITATION":
                 agent_response_text = "My response couldn't be completed due to potential recitation issues."
             elif finish_reason_name == "MAX_TOKENS":
                 try: # Attempt to get partial text if stopped due to length
                     agent_response_text = response.candidates[0].content.parts[0].text + "... (response truncated)"
                     logger.warning("Response truncated due to max_tokens.")
                 except (AttributeError, IndexError):
                     agent_response_text = "My response was cut short as it reached the maximum length."
             else:
                agent_response_text = f"Sorry, I had trouble generating a complete response (Finish Reason: {finish_reason_name}). Could you rephrase?"
        else:
             # Successfully got response text
             agent_response_text = response.candidates[0].content.parts[0].text
             logger.info(f"Gemini response received: {agent_response_text}")

             # --- Update Order Based on Response (Heuristic) ---
             # Modifies the 'updated_order' local variable
             for item_id, item in menu.items():
                 item_name_lower = item["name"].lower()
                 response_lower = agent_response_text.lower()
                 if item_name_lower in response_lower and \
                    any(add_word in response_lower for add_word in ["added", "adding", "got it", "sure thing", "order up", "coming right up"]):
                      # Avoid adding duplicates if it's already the *last* item added
                      if not updated_order or item["name"] != updated_order[-1]["name"]:
                          updated_order.append(item) # Append to local copy
                          logger.info(f"Heuristic: Added '{item['name']}' to session order.")
                          break # Only add the first match found

        # --- Update Session History ---
        # Append user input and assistant response to the local history copy
        # This prepares the history to be returned as the new state
        updated_history.append({'role': 'user', 'content': user_input_text})
        updated_history.append({'role': 'assistant', 'content': agent_response_text})

        # --- Return updated state for this session ---
        return agent_response_text, updated_history, updated_order

    except Exception as e:
        # Catch exceptions not handled by tenacity retry
        logger.exception(f"Critical error in process_order: {str(e)}")
        # Provide a safe fallback response and state
        error_message = "I'm sorry, an unexpected error occurred. Please try again later."
        # Append only the error message to history to inform the user
        # Make sure not to corrupt the state further
        safe_history = current_session_history[:] # Revert to original history for this turn
        safe_history.append({'role': 'user', 'content': user_input_text}) # Keep user msg
        safe_history.append({'role': 'assistant', 'content': error_message})
        return error_message, safe_history, current_session_order # Return original order state on error

# Note: No reset_order function is needed as state is reset in main.py callbacks.
# Note: No get_voice_response function included as it wasn't part of the stateless refactoring focus.
#       If needed, it would require passing the cartesia_api_key or client. 

# --- New TTS Function ---

# Define retryable exceptions for Cartesia if known, otherwise use generic ones
# Example: CARTESIA_RETRYABLE_EXCEPTIONS = (cartesia.errors.ServerError, cartesia.errors.RateLimitError, ConnectionError)
# Using generic exceptions for now as specific Cartesia ones aren't known here.
CARTESIA_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError) # Add more specific Cartesia errors if documented

@tenacity_retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(CARTESIA_RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING) if callable(before_sleep_log) else None,
    reraise=True
)
def get_voice_audio(text_to_speak: str) -> bytes | None:
    """Calls Cartesia API synchronously to synthesize speech and returns WAV bytes."""
    global cartesia_client, CARTESIA_VOICE_ID # Access the global client and voice ID

    if not text_to_speak or not text_to_speak.strip():
        logger.warning("get_voice_audio received empty text.")
        return None
    if not cartesia_client or not CARTESIA_VOICE_ID:
         logger.error("Cartesia client or voice ID not initialized, cannot generate audio.")
         return None

    try:
        logger.info(f"Requesting TTS from Cartesia (Voice ID: {CARTESIA_VOICE_ID}) for: '{text_to_speak[:50]}...'")

        # --- Check Cartesia Documentation for the exact method call ---
        # This is a plausible synchronous implementation pattern:
        audio_generator = cartesia_client.tts.bytes(
            model_id="sonic-2",
            transcript=text_to_speak,
            voice={"mode":"id",
                   "id": CARTESIA_VOICE_ID,
            },
            language="en",
            # Specify desired output format and sample rate
            output_format={"container":"wav",
                           "sample_rate": 24000,
                           "encoding": "pcm_f32le",
            },
        )

        # Concatenate chunks from the generator for a blocking result
        audio_data = b"".join(chunk for chunk in audio_generator)
        # --- End of section requiring Cartesia documentation check ---

        if not audio_data:
            logger.warning("Cartesia TTS returned empty audio data.")
            return None

        logger.info(f"Received {len(audio_data)} bytes of WAV audio data from Cartesia.")
        return audio_data

    # Catch specific Cartesia errors if they exist and are imported
    # except cartesia.errors.CartesiaError as e:
    #    logger.exception(f"Cartesia API error during TTS generation: {e}")
    #    return None
    except Exception as e:
        # Catch any other unexpected error during TTS
        logger.exception(f"Unexpected error generating voice audio with Cartesia: {e}")
        return None

# --- End of bartending_agent.py --- 
import gradio as gr
import logging
from typing import List, Dict, Tuple, Any

# --- Configure logging FIRST ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import functions from the refactored agent logic module ---
# Ensure this module exists and contains the necessary functions
try:
    # Assuming the refactored logic is in 'bartending_agent_logic.py'
    from bartending_agent import (
        process_order,
        get_menu_text,
        get_voice_audio,
        # No need to import reset_order if clearing state is handled here
    )
    # Initialization (like model loading, API key checks) should occur
    # within bartending_agent_logic.py upon import.
    logger.info("Successfully imported agent logic functions.")
except ImportError as e:
    logger.exception("Failed to import agent functions. Ensure 'bartending_agent.py' exists and is correctly structured.")
    raise SystemExit(f"Import Error: {e}") from e
except Exception as e:
    # Catch potential errors during module-level initialization in the logic module
    logger.exception(f"Error during agent module initialization: {e}")
    raise SystemExit(f"Initialization Error: {e}") from e

# --- Gradio Interface Callbacks (Using Session State) ---

def handle_gradio_input(
    user_input: str,
    session_history_state: List[Dict[str, str]],
    session_order_state: List[Dict[str, float]]
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, float]], Any]:
    """
    Gradio callback: Takes input/state, calls logic & TTS, returns updates.
    """
    logger.info(f"Gradio input: '{user_input}'")
    logger.debug(f"Received session history state (len {len(session_history_state)}): {session_history_state}")
    logger.debug(f"Received session order state (len {len(session_order_state)}): {session_order_state}")

    # Call text processing logic first
    response_text, updated_history, updated_order = process_order(
        user_input,
        session_history_state,
        session_order_state
    )

    # --- Get Voice Audio ---
    audio_data = None # Default to None
    # Check if there is a non-empty response text to synthesize
    if response_text and response_text.strip():
         audio_data = get_voice_audio(response_text) # Call the imported function
         if audio_data is None:
             logger.warning("Failed to get audio data from get_voice_audio.")
             # Optional: Add indication to user? E.g., append "[Audio failed]" to response_text
    else:
        logger.info("No response text generated, skipping TTS.")
    # --- End Get Voice Audio ---

    # Return updates including audio data (which might be None)
    return "", updated_history, updated_history, updated_order, audio_data

def clear_chat_state() -> Tuple[List, List, List, None]:
    """Clears UI/session state including audio."""
    logger.info("Clear button clicked - Resetting session state.")
    # Return empty lists for Chatbot/history/order, and None for the audio component
    return [], [], [], None

# --- Gradio UI Definition (with gr.State) ---

theme = gr.themes.Citrus()

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Bartending Agent")
    gr.Markdown("Welcome to MOK 5-ha! Ask me for a drink or check your order.")

    # --- Define Session State Variables ---
    history_state = gr.State([])
    order_state = gr.State([])

    # --- Restructured Main Row with 2 Columns (Equal Scaling) ---
    with gr.Row():

        # --- Column 1: Avatar Image ---
        # Scale is relative to other columns in the same row
        with gr.Column(scale=1, min_width=200): # Keep scale=1
            gr.Image(
                value="assets/bartender_avatar_ai_studio.jpeg",
                label="Bartender Avatar",
                show_label=False,
                interactive=False,
                height=600, # Adjust as desired
                elem_classes=["avatar-image"]
            )

        # --- Column 2: Chat Interface ---
        with gr.Column(scale=1): # <-- Changed scale from 3 to 1
            chatbot_display = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Conversation",
                bubble_full_width=False,
                height=450, # Keep or adjust height for rectangular shape
                type="messages"
            )
            agent_audio_output = gr.Audio(
                label="Agent Voice",
                autoplay=True,
                streaming=False,
                format="wav",
                show_label=True,
                interactive=False
            )
            msg_input = gr.Textbox(
                label="Your Order / Message",
                placeholder="What can I get for you? (e.g., 'I'd like a Margarita', 'Show my order')"
            )
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation")
                submit_btn = gr.Button("Send", variant="primary")

    # --- Event Handlers (Remain the same) ---
    submit_inputs = [msg_input, history_state, order_state]
    submit_outputs = [msg_input, chatbot_display, history_state, order_state, agent_audio_output]
    msg_input.submit(handle_gradio_input, submit_inputs, submit_outputs)
    submit_btn.click(handle_gradio_input, submit_inputs, submit_outputs)

    clear_outputs = [chatbot_display, history_state, order_state, agent_audio_output]
    clear_btn.click(clear_chat_state, None, clear_outputs)

# --- Launch the Gradio Interface (Remains the same) ---
if __name__ == "__main__":
    logger.info("Launching Gradio interface locally...")
    # For local development (VSCode):
    # - debug=True enables auto-reloading on code changes and more verbose logs.
    # - share=False keeps the app accessible only on your local machine/network.
    # - You might want to specify server_name="0.0.0.0" to access from other devices
    #   on your local network.
    demo.launch(debug=True, share=False) # server_name="0.0.0.0"
    logger.info("Gradio interface closed.") 
import gradio as gr
from api.bartending_agent import BartendingAgent
import logging

# Configure logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Bartending Agent
try:
    agent = BartendingAgent()
    logger.info("Bartending Agent initialized successfully.")
except EnvironmentError as e:
    logger.error(f"EnvironmentError initializing agent: {e}")
    # Exit or provide a dummy agent if API keys are missing
    # For now, we'll re-raise to make the issue clear
    raise 
except RuntimeError as e:
    logger.error(f"RuntimeError initializing agent: {e}")
    # Handle other initialization errors (e.g., Gemini connection)
    raise

# Define the function that Gradio will call
def handle_order(user_input, chat_history):
    logger.info(f"Received user input: {user_input}")
    
    # Process the order using the agent
    response_text = agent.process_order(user_input)
    logger.info(f"Agent response text: {response_text}")
    
    # Optionally, generate voice response (if Cartesia API is configured and needed)
    # voice_response = agent.get_voice_response(response_text) 
    # logger.info(f"Agent voice response generated (placeholder): {voice_response}")
    
    # Update chat history
    chat_history.append((user_input, response_text))
    
    # Return updated chat history and potentially the voice response
    # For now, just returning text response updates
    return "", chat_history # Return empty string to clear input box

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Bartending Agent")
    gr.Markdown(agent.get_menu_text()) # Display the menu

    chatbot = gr.Chatbot(label="Conversation", value=[]) # Initialize chatbot display
    
    msg = gr.Textbox(label="Your Order", placeholder="What can I get for you?")
    
    clear = gr.Button("Clear Conversation")

    msg.submit(handle_order, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (agent.reset_order(), []), None, [chatbot], queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    # Set share=True to create a public link (optional)
    demo.launch(share=True) 
    logger.info("Gradio interface launched.") 
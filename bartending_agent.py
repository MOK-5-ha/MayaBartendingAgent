from typing import Dict, List, Optional
import google.generativeai as genai
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
import os
import logging
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
if not load_dotenv():
    raise EnvironmentError("Could not load .env file. Please ensure it exists in the project root.")

class BartendingAgent:
    def __init__(self):
        # Check for required environment variables
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )

        self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        if not self.cartesia_api_key:
            raise EnvironmentError(
                "CARTESIA_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise RuntimeError(
                f"Failed to initialize Gemini model: {str(e)}. "
                "Please check if your GEMINI_API_KEY is valid."
            )
        
        self.menu = {
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
        
        self.current_order = []
        self.conversation_history = []
        
    def get_menu_text(self) -> str:
        menu_text = "Here's our menu:\n"
        for item_id, item in self.menu.items():
            menu_text += f"{item_id}. {item['name']} - ${item['price']:.2f}\n"
        return menu_text
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def process_order(self, text: str) -> str:
        """Processes user input using the Gemini model for conversational interaction."""
        try:
            # Add the user's input to conversation history
            self.conversation_history.append({"role": "user", "parts": [text]})
            logger.info(f"Processing user input: {text}")

            # --- Construct the prompt for Gemini ---
            prompt_parts = [
                "You are a friendly and helpful bartender taking drink orders.",
                "Be conversational. Ask clarifying questions if the order is unclear.",
                "If the user asks for something not on the menu, politely tell them and show the menu again.",
                "If the user asks to see their current order, list the items and their prices.",
                "\nHere is the menu:",
                self.get_menu_text(),
                "\nCurrent order:",
            ]
            if self.current_order:
                order_text = "\n".join([f"- {item['name']} (${item['price']:.2f})" for item in self.current_order])
                prompt_parts.append(order_text)
            else:
                prompt_parts.append("No items ordered yet.")

            prompt_parts.append("\nConversation History:")
            # Add previous turns, limiting history length if necessary
            history_limit = 10 # Keep the last 10 turns
            limited_history = self.conversation_history[-(history_limit + 1):-1] # Exclude the latest user input already added

            for entry in limited_history:
                 # Ensure history parts are strings. Handle potential list/dict structures if they exist.
                 content = entry.get("parts", [""])[0] # Assuming parts is a list with one string
                 if isinstance(content, dict): # Handle cases where parts might be structured differently
                     content = str(content) # Fallback to string representation
                 role = entry.get("role", "unknown")
                 prompt_parts.append(f"{role.capitalize()}: {content}")


            prompt_parts.append(f"\nUser: {text}")
            prompt_parts.append("\nBartender:")

            full_prompt = "\n".join(prompt_parts)
            logger.debug(f"Full prompt for Gemini:\n{full_prompt}")

            # --- Call the Gemini model ---
            # Use stream=False for a single response object
            # Safety settings can be configured here if needed
            generation_config = genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                # stop_sequences=None,
                # max_output_tokens=2048, # Already set in .env? Agent could load this.
                temperature=0.7 # Already set in .env? Agent could load this.
            )

            response = self.model.generate_content(
                contents=[full_prompt], # Send as a list for single-turn
                generation_config=generation_config,
                # safety_settings='HARM_BLOCK_THRESHOLD_UNSPECIFIED' # Adjust safety if needed
            )

            # --- Process the response ---
            if not response.candidates or not response.candidates[0].content.parts:
                 logger.error("Gemini response was empty or invalid.")
                 agent_response_text = "Sorry, I had trouble understanding that. Could you please rephrase?"
            else:
                 agent_response_text = response.candidates[0].content.parts[0].text
                 logger.info(f"Gemini response: {agent_response_text}")

                 # Simple check if the *model's response* mentions adding an item
                 # This is basic; a more robust approach might involve asking the model
                 # to output structured data or using function calling.
                 for item_id, item in self.menu.items():
                     if item["name"].lower() in agent_response_text.lower() and \
                        any(add_word in agent_response_text.lower() for add_word in ["added", "adding", "got it", "sure thing"]):
                          # Avoid adding duplicates if already in order from this turn?
                          # This logic might need refinement based on model behavior.
                          if not self.current_order or item["name"] != self.current_order[-1]["name"]:
                              self.current_order.append(item)
                              logger.info(f"Added '{item['name']}' to order based on Gemini response.")
                              # Only add the first match found in the response?
                              break

            # Add the agent's response to conversation history
            self.conversation_history.append({"role": "assistant", "parts": [agent_response_text]})

            return agent_response_text

        # except genai.types.BlockedPromptError as e: # Requires specific import if needed
        #     logger.error(f"Gemini prompt blocked: {e}")
        #     return "I'm sorry, I can't respond to that request due to safety guidelines."
        # except genai.types.StopCandidateException as e: # Requires specific import if needed
        #     logger.error(f"Gemini response stopped: {e}")
        #     # The partial response might be in e.response
        #     return response.candidates[0].content.parts[0].text if response and response.candidates else "My response was cut short."
        except Exception as e:
            logger.exception(f"Error processing order with Gemini: {str(e)}") # Use logger.exception for traceback
            # Fallback response
            return "I'm sorry, there was an internal error processing your request. Please try again."
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def get_voice_response(self, text: str) -> str:
        try:
            if not self.cartesia_api_key:
                raise RuntimeError(
                    "Cannot generate voice response: CARTESIA_API_KEY is not set"
                )
            
            logger.info(f"Generating voice response for text: {text}")
            
            # This would be where you'd call the Cartesia API
            # For now, we'll just return the text
            return text
        except Exception as e:
            logger.error(f"Error generating voice response: {str(e)}")
            raise RuntimeError(
                f"Failed to generate voice response: {str(e)}. "
                "Please check your CARTESIA_API_KEY and network connection."
            )
    
    def reset_order(self):
        try:
            self.current_order = []
            self.conversation_history = []
            logger.info("Order and conversation history reset successfully")
        except Exception as e:
            logger.error(f"Error resetting order: {str(e)}")
            raise RuntimeError(f"Failed to reset order: {str(e)}") 
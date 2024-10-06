#!/usr/bin/env python3
"""
Advanced chatbot interface using Gradio with color-coded messages and history maintenance.
"""

import gradio as gr
import logging
from typing import List, Dict, Optional, Tuple
from g4f.client import Client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Chatbot implementation that retains user history during interactions,
providing context-aware responses based on past conversations.
"""



class Chatbot:
    """
    A chatbot that retains conversation history and provides context-aware responses.
    """

    def __init__(self, model: str = "gpt-4o", max_history: Optional[int] = None) -> None:
        """
        Initialize the Chatbot with a specified model and optional max_history limit.

        Args:
            model (str): The model name to use for generating responses.
            max_history (Optional[int]): The maximum number of exchanges (user and assistant messages)
                                         to retain in history. If None, retains all messages.
        """
        self.client: Client = Client()
        self.model: str = model
        self.messages: List[Dict[str, str]] = []
        self.max_history: Optional[int] = max_history

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger: logging.Logger = logging.getLogger(__name__)

    def send_message(self, content: str) -> str:
        """
        Send a message to the chatbot and receive a response.

        Args:
            content (str): The user's message content.

        Returns:
            str: The chatbot's response.
        """
        # Append user's message to the conversation history
        self.messages.append({"role": "user", "content": content})

        # Trim conversation history if necessary
        if self.max_history is not None and len(self.messages) > self.max_history * 2:
            # Retain only the last max_history exchanges (each exchange is 2 messages)
            self.messages = self.messages[-self.max_history * 2 :]
            self.logger.debug("Trimmed conversation history to the last %s exchanges.", self.max_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            assistant_response: str = response.choices[0].message.content.strip()

            # Append assistant's response to the conversation history
            self.messages.append({"role": "assistant", "content": assistant_response})
            return assistant_response

        except Exception as e:
            # Handle exceptions gracefully and log the error
            self.logger.error("An error occurred during chatbot interaction.", exc_info=True)
            return "I'm sorry, but I couldn't process your request at this time."



# Custom CSS for color-coding and layout
custom_css = """
.user-message { color: #1E90FF; text-align: right; }
.bot-message { color: #32CD32; text-align: left; }
.chatbot-container { height: 400px; overflow-y: auto; }
"""

def format_message(role: str, content: str) -> str:
    """
    Format a message with appropriate CSS class based on the role.

    Args:
        role (str): The role of the message sender ('user' or 'assistant').
        content (str): The content of the message.

    Returns:
        str: HTML-formatted message with appropriate CSS class.
    """
    css_class = "user-message" if role == "user" else "bot-message"
    return f'<div class="{css_class}">{content}</div>'

def chat_interface(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Process user input and generate chatbot response.

    Args:
        message (str): User's input message.
        history (List[Tuple[str, str]]): Conversation history.

    Returns:
        Tuple[str, List[Tuple[str, str]]]: Chatbot's response and updated history.
    """
    try:
        chatbot = Chatbot()  # Initialize chatbot
        response = chatbot.send_message(message)
        history.append((message, response))
        return "", history
    except Exception as e:
        logger.error("An error occurred in the chat interface.", exc_info=True)
        return "I apologize, but I encountered an error. Please try again.", history

def format_history(history: List[Tuple[str, str]]) -> str:
    """
    Format the conversation history with color-coded messages.

    Args:
        history (List[Tuple[str, str]]): Conversation history.

    Returns:
        str: HTML-formatted conversation history.
    """
    formatted_history = ""
    for user_msg, bot_msg in history:
        formatted_history += format_message("user", user_msg)
        formatted_history += format_message("assistant", bot_msg)
    return formatted_history

def launch_interface() -> None:
    """
    Launch the Gradio interface for the chatbot.
    """
    with gr.Blocks(css=custom_css) as demo:
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            height=400,
        )
        msg = gr.Textbox(placeholder="Type your message here...")
        clear = gr.Button("Clear")

        def user(user_message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
            """
            Process user input and update the interface.

            Args:
                user_message (str): User's input message.
                history (List[Tuple[str, str]]): Conversation history.

            Returns:
                Tuple[str, List[Tuple[str, str]]]: Empty string and updated history.
            """
            return chat_interface(user_message, history)

        msg.submit(user, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()

if __name__ == "__main__":
    launch_interface()
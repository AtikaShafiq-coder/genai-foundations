from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
import logging

# Set up logging for production-readiness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    """
    Backend engine for AI interactions using LangChain and Ollama.
    Separates the AI logic from the UI layer.
    """
    def __init__(self, model_name: str = "gemma3:1b", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self._init_llm()

    def _init_llm(self):
        """Initializes the ChatOllama model with current parameters."""
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
            )
            logger.info(f"Initialized ChatOllama with model {self.model_name} and temperature {self.temperature}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise e

    def update_params(self, temperature: float):
        """Updates model parameters without needing to re-instantiate the whole class."""
        if self.temperature != temperature:
            self.temperature = temperature
            self._init_llm()

    def get_response(self, messages: list[BaseMessage], stream: bool = True):
        """
        Fetches a response from the LLM.

        Args:
            messages: A list of LangChain message objects (HumanMessage, AIMessage).
            stream: Whether to stream the response or return it as a single block.

        Returns:
            A generator of content chunks if stream=True, otherwise the full response string.
        """
        try:
            if stream:
                # llm.stream returns a generator of message chunks
                for chunk in self.llm.stream(messages):
                    yield chunk.content
            else:
                response = self.llm.invoke(messages)
                return response.content
        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            # Raise the exception to be caught by the UI layer (app.py)
            raise e

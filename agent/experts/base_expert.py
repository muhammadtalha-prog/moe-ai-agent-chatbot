from abc import ABC, abstractmethod
from typing import List, Dict, Any
from agent.memory import VectorStore

class BaseExpert(ABC):
    @abstractmethod
    def respond(self, user_input: str, history: List[Dict[str, str]], memory: VectorStore, **kwargs) -> str:
        """
        Process the user input and return a string response.
        
        :param user_input: The raw input text from the user.
        :param history: List of past messages (e.g. [{"role": "user", "content": "..."}]).
        :param memory: The VectorStore memory instance.
        :param kwargs: Additional metadata or tools parameters.
        :return: Response string.
        """
        pass

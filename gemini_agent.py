from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import google.generativeai as genai
import os
from typing import List, Dict
import warnings

class GeminiChatAgent:
    def __init__(self, api_key: str, system_prompt: str = None):
        """
        Initialize the Gemini chat agent with API key and optional system prompt.
        
        Args:
            api_key: Google API key for Gemini
            system_prompt: Custom system prompt to define agent behavior
        """
        # Set up API key
        print("Initialized GeminiChatAgent: ")

        os.environ['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=api_key)
        
        # Default system prompt if none provided
        self.system_prompt = system_prompt or """You are a helpful AI assistant. 
        You provide clear, accurate, and helpful responses while maintaining context 
        of the conversation history."""
        
        # Initialize the Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # Initialize chat history
        self.chat_history = []
        
        # Add system message to history
        self.chat_history.append(SystemMessage(content=self.system_prompt))
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the chain
        self.chain = (
            {"history": lambda x: self.chat_history[1:], "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def chat(self, user_input: str) -> str:
        """
        Process user input and return agent response while maintaining conversation history.
        
        Args:
            user_input: User's message
            
        Returns:
            str: Agent's response
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Add user message to history
                self.chat_history.append(HumanMessage(content=user_input))
                
                # Get response from chain
                response = self.chain.invoke(user_input)
                
                # Add AI response to history
                self.chat_history.append(AIMessage(content=response))
                
                return response
            except Exception as e:
                return f"Error processing request: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history.
        
        Returns:
            List of dictionaries containing messages and their types
        """
        history = []
        for message in self.chat_history:
            if isinstance(message, SystemMessage):
                history.append({"type": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})
        return str(history)
    
    def modify_system_prompt(self, new_prompt: str):
        """
        Update the system prompt for the agent.
        
        Args:
            new_prompt: New system prompt to use
        """
        self.system_prompt = new_prompt
        # Clear existing history
        self.chat_history = [SystemMessage(content=new_prompt)]
        # Update prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        # Update chain
        self.chain = (
            {"history": lambda x: self.chat_history[1:], "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

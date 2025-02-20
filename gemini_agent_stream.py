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
import asyncio
from typing import List, Dict, AsyncGenerator, Generator


class GeminiChatAgentStream:
    def __init__(self, api_key: str, system_prompt: str = None):
        print("Initialized GeminiChatAgent: ")

        os.environ['GOOGLE_API_KEY'] = api_key
        genai.configure(api_key=api_key)
        
        self.system_prompt = system_prompt or """You are a helpful AI assistant. 
        You provide clear, accurate, and helpful responses while maintaining context 
        of the conversation history."""
        
        # print(self.system_prompt)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            stream=True,  # Enable streaming
            convert_system_message_to_human=True
        )
        
        self.chat_history = []
        self.chat_history.append(SystemMessage(content=self.system_prompt))
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self.chain = (
            {"history": lambda x: self.chat_history[1:], "input": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
    
    def chat_stream(self, user_input: str) -> Generator[str, None, None]:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Add user message to history
                self.chat_history.append(HumanMessage(content=user_input))
                
                # Stream response
                full_response = ""
                for chunk in self.chain.stream(user_input):
                    # print(chunk)
                    text = str(chunk.content)
                    full_response += text
                    yield text
                
                # Add full AI response to history
                self.chat_history.append(AIMessage(content=full_response))
            
            except Exception as e:
                yield f"Error processing request: {str(e)}"
    
    async def chat_stream_for_fastApi(self, user_input: str) -> AsyncGenerator[str, None]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Add user message to history
                self.chat_history.append(HumanMessage(content=user_input))

                # Stream response
                full_response = ""
                for chunk in self.chain.stream(user_input):
                    # Process each chunk
                    text = str(chunk.content)
                    full_response += text
                    yield text  # Send chunk to the client
                    await asyncio.sleep(0)  # Allow event loop to run other tasks

                # Add full AI response to history
                self.chat_history.append(AIMessage(content=full_response))

            except Exception as e:
                yield f"Error processing request: {str(e)}"

    def chat(self, user_input: str) -> str:
        """
        Process user input and return agent response while maintaining conversation history.
        
        Args:
            user_input: User's message
            
        Returns:
            str: Agent's response
        """
        full_response = ""
        for chunk in self.chat_stream(user_input):
            full_response += str(chunk.content)
            # chunk += '\n'
            
        return full_response
    
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
        )
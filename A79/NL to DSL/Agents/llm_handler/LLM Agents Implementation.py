import os
from typing import Any, List, Dict, TypeVar, Generic, Optional
from pydantic import BaseModel, Field
import litellm
from opentelemetry import trace
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from enum import Enum
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent_benchmark")

# Type variable for generic type hinting
T = TypeVar('T')

# Mock classes for compatibility with the provided code
class RateLimitError(Exception):
    pass

class APIConnectionError(Exception):
    pass

class ServiceUnavailableError(Exception):
    pass

class Constants:
    FALLBACK_LIST = []

constants = Constants()

# Tracer decorator mock
def ot_tracer():
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Classes for structured output
class AgentAction(BaseModel):
    action: str = Field(description="The action to take")
    action_input: str = Field(description="The input to the action")
    
class AgentResponse(BaseModel):
    thoughts: str = Field(description="Agent's reasoning process")
    action: AgentAction = Field(description="The action the agent decides to take")
    final_answer: Optional[str] = Field(description="The final answer if the agent has completed the task", default=None)

# Base Agent class
class BaseAgent:
    def __init__(
        self,
        model_name: str,
        system_message: str = "You are a helpful assistant.",
        max_output_tokens: int = 4096,
        max_retries: int = 3,
        retry_min_wait: int = 1,
        retry_max_wait: int = 60,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        self.model_name = model_name
        self.system_message = system_message
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.base_url = base_url
        self.api_version = api_version
    
    def set_langfuse_call_metadata(self):
        # Mock implementation
        return {"source": "agent_benchmark"}
    
    def attempt_on_error(self, retry_state):
        logger.warning(f"Retrying LLM call. Attempt {retry_state.attempt_number} of {self.max_retries}")
        return None

# Standard LLM Agent implementing the provided call_llm method
class StandardLLMAgent(BaseAgent):
    def call_llm(self, text: str, *args: Any, **kwargs: Any) -> str:
        """
        This function calls the LLM with a given text and returns the response.
        Args:
        - text (str): The text to be sent to the LLM.
        - *args (Any): Additional positional arguments to be passed to the LLM.
        - **kwargs (Any): Additional keyword arguments to be passed to the LLM.
        Returns:
        - str: The response from the LLM.
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"{text}"},
        ]
        
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(
                min=self.retry_min_wait, max=self.retry_max_wait
            ),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, ServiceUnavailableError)
            ),
            after=self.attempt_on_error,
        )
        def completion_with_retry(
            messages: List[Dict[str, str]], *args: Any, **kwargs: Any
        ) -> str:
            metadata = self.set_langfuse_call_metadata()
            
            # For benchmark purposes, we'll simulate API call delay
            time.sleep(0.5)
            
            # Mock response or actual call depending on environment
            if os.environ.get("BENCHMARK_MOCK", "true").lower() == "true":
                # Create a mock response
                resp = {
                    "choices": [
                        {
                            "message": {
                                "content": f"Simulated response to: {messages[-1]['content'][:50]}..."
                            }
                        }
                    ]
                }
            else:
                # Actual LLM call
                resp = litellm.completion(
                    messages=messages,
                    *args,
                    **kwargs,
                    model=self.model_name,
                    base_url=self.base_url,
                    api_version=self.api_version,
                    max_tokens=self.max_output_tokens,
                    metadata=metadata,
                    fallbacks=constants.FALLBACK_LIST,
                )
            
            # Simulate OpenTelemetry span
            # trace.get_current_span().add_event(str(resp))
            
            message_back = resp["choices"][0]["message"]["content"]
            logger.debug(message_back)
            logger.info(f"LLM call completed")
            
            return resp["choices"][0]["message"]["content"]
        
        return completion_with_retry(messages, *args, **kwargs)
    
    def run(self, query: str) -> str:
        """Run the agent on a query and get a text response"""
        prompt = f"""
Please help with the following query:

{query}

Think through this step by step and provide your answer.
"""
        return self.call_llm(prompt)

# Structured Output Agent using the call_llm_structured_output method
class StructuredLLMAgent(BaseAgent):
    @ot_tracer().start_as_current_span("call_llm_structured")
    def call_llm_structured_output(
        self, text: str, resp_model: type[T], *args: Any, **kwargs: Any
    ) -> T:
        """
        This function calls the LLM with a given text and returns the response.
        Args:
        - text (str): The text to be sent to the LLM.
        - resp_model (BaseModel): The model to be used for the response.
        - *args (Any): Additional positional arguments to be passed to the LLM.
        - **kwargs (Any): Additional keyword arguments to be passed to the LLM.
        Returns:
        - T: An instance of the specified Pydantic model filled with the LLM response.
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"{text}"},
        ]
        
        message_separator = "\n---\n"
        debug_message = ""
        for message in messages:
            debug_message += message_separator + message["content"]
        logger.debug(debug_message)
        
        metadata = self.set_langfuse_call_metadata()
        
        # For benchmark purposes, we'll simulate API call delay
        time.sleep(0.5)
        
        # Mock response or actual call depending on environment
        if os.environ.get("BENCHMARK_MOCK", "true").lower() == "true":
            # Create a simulated structured response based on model
            if resp_model == AgentResponse:
                mock_data = {
                    "thoughts": f"Thinking about how to respond to: {messages[-1]['content'][:50]}...",
                    "action": {
                        "action": "search",
                        "action_input": "relevant terms from query"
                    },
                    "final_answer": f"Simulated structured response to the query."
                }
                resp_obj = json.dumps(mock_data)
            else:
                resp_obj = "{}"
            
            resp = {
                "choices": [
                    {
                        "message": {
                            "content": resp_obj
                        }
                    }
                ]
            }
        else:
            # Actual LLM call
            resp = litellm.completion(
                messages=messages,
                *args,
                **kwargs,
                model=self.model_name,
                base_url=self.base_url,
                max_tokens=self.max_output_tokens,
                metadata=metadata,
                response_format=resp_model,
                fallbacks=constants.FALLBACK_LIST,
            )
        
        # Simulate OpenTelemetry span
        # trace.get_current_span().add_event(str(resp))
        
        resp_obj = resp["choices"][0]["message"]["content"]
        rep_obj_parsed = resp_model.model_validate_json(resp_obj)
        logger.debug(resp_obj)
        
        return rep_obj_parsed
    
    def run(self, query: str) -> str:
        """Run the agent on a query and get a structured response, then extract final answer"""
        prompt = f"""
Please analyze the following query and respond with your reasoning and actions.

Query: {query}

Think step by step. First analyze what's being asked, then decide on the best action, 
and if you have enough information, provide a final answer.
"""
        response = self.call_llm_structured_output(prompt, AgentResponse)
        
        if response.final_answer:
            return response.final_answer
        else:
            return f"Action: {response.action.action} with input: {response.action.action_input}\nThoughts: {response.thoughts}"
# Copyright 2025 The OpenManus-RL Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenManus Model Manager Module

This module provides a unified interface for loading and running inference with various LLM models.
It supports multiple model types including Hugging Face models, Grok, and other API-based models.
"""

import os
import logging
import json
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Generator, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
import threading

logger = logging.getLogger(__name__)

# Default system prompts for different models
DEFAULT_SYSTEM_PROMPTS = {
    "default": """You are OpenManus AI, an assistant specialized in reinforcement learning for LLM agent tuning.
You can answer questions about reinforcement learning, LLM agent tuning, and related topics.
Provide detailed and accurate information based on your training.""",
    
    "grok": """You are OpenManus AI powered by Grok, an assistant specialized in reinforcement learning for LLM agent tuning.
You can answer questions about reinforcement learning, LLM agent tuning, and related topics.
Provide detailed and accurate information based on your training.""",
    
    "developer": """You are OpenManus AI in developer mode, an assistant specialized in reinforcement learning for LLM agent tuning.
You provide detailed technical information, code examples, and implementation details.
Feel free to discuss advanced concepts, experimental features, and implementation challenges."""
}

# Simple default chat template for models that don't have one
DEFAULT_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|assistant|>
{% endif %}
"""


class BaseModelInterface(ABC):
    """Abstract base class for all model interfaces."""
    
    def __init__(self, system_prompt: str = None, max_new_tokens: int = 1024, 
                 temperature: float = 0.7, top_p: float = 0.9):
        """Initialize the model interface with common parameters."""
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS["default"]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    @abstractmethod
    def generate_response(self, user_message: str) -> str:
        """Generate a response to a user message."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], streaming: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate a response based on a conversation history."""
        pass
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        return {
            "system_prompt": self.system_prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


class HuggingFaceModelInterface(BaseModelInterface):
    """Interface for Hugging Face models."""
    
    def __init__(self, model_path: str, device: str = None, system_prompt: str = None,
                 max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9):
        """Initialize the Hugging Face model interface."""
        super().__init__(system_prompt, max_new_tokens, temperature, top_p)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Determine device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading HuggingFace model from {model_path} on {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Set chat template if not already set
            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Set generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Successfully loaded model and tokenizer")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_response(self, user_message: str) -> str:
        """Generate a response to a user message."""
        try:
            # Create messages list with system prompt and user message
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Apply chat template and tokenize
            encoded_inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )
            
            # Create input_ids and attention_mask
            input_ids = encoded_inputs.to(self.device)
            # Create proper attention mask that accounts for padding
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int() if self.tokenizer.pad_token_id is not None else torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], streaming: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate a response based on a conversation history."""
        try:
            # Ensure system prompt is included
            if not any(msg.get("role") == "system" for msg in messages):
                messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            logger.info(f"Processing chat with {len(messages)} messages, streaming={streaming}")
            
            # Signal thinking stage: Tokenizing input
            logger.info("Thinking stage: Tokenizing input")
            
            # Apply chat template and tokenize
            encoded_inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            )
            
            # Create input_ids and attention_mask
            input_ids = encoded_inputs.to(self.device)
            # Create proper attention mask that accounts for padding
            attention_mask = (input_ids != self.tokenizer.pad_token_id).int() if self.tokenizer.pad_token_id is not None else torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.device)
            
            logger.debug(f"Input prepared with {input_ids.shape[1]} tokens")
            
            # Signal thinking stage: Analyzing context
            logger.info("Thinking stage: Analyzing context")
            
            if streaming:
                # For streaming, we'll use the model's generate method with a streamer
                logger.info("Setting up streaming response generation")
                
                # Signal thinking stage: Preparing response
                logger.info("Thinking stage: Preparing response")
                
                # Create a streamer with improved settings
                streamer = TextIteratorStreamer(
                    self.tokenizer, 
                    skip_prompt=True, 
                    skip_special_tokens=True,
                    timeout=120.0  # Add timeout to prevent hanging
                )
                
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "generation_config": self.generation_config,
                    "streamer": streamer
                }
                
                # Create a thread to run the generation
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.daemon = True  # Make thread daemon so it doesn't block program exit
                thread.start()
                
                logger.info("Started streaming generation thread")
                
                # Return the streamer as a generator
                return streamer
            else:
                # Generate response without streaming
                logger.info("Generating non-streaming response")
                
                # Signal thinking stage: Generating response
                logger.info("Thinking stage: Generating response")
                
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            generation_config=self.generation_config
                        )
                        
                        # Decode response
                        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                        logger.info(f"Generated response with {len(response)} characters")
                        
                        # Signal thinking stage: Complete
                        logger.info("Thinking stage: Complete")
                        
                        return response.strip()
                    except Exception as e:
                        logger.error(f"Error during non-streaming generation: {e}")
                        raise
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        info = super().model_info
        info.update({
            "model_type": "huggingface",
            "model_path": self.model_path,
            "device": self.device
        })
        return info


class GrokModelInterface(BaseModelInterface):
    """Interface for Grok API."""
    
    def __init__(self, api_key: str = None, system_prompt: str = None,
                 max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9):
        """Initialize the Grok model interface."""
        system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPTS["grok"]
        super().__init__(system_prompt, max_new_tokens, temperature, top_p)
        
        # Get API key from environment variable if not provided
        self.api_key = api_key or os.environ.get("GROK_API_KEY")
        if not self.api_key:
            logger.warning("No Grok API key provided. Please set GROK_API_KEY environment variable or provide api_key parameter.")
        
        self.api_url = "https://api.grok.ai/v1/chat/completions"
        logger.info("Initialized Grok model interface")
    
    def generate_response(self, user_message: str) -> str:
        """Generate a response to a user message using Grok API."""
        try:
            if not self.api_key:
                return "Error: Grok API key not provided. Please set GROK_API_KEY environment variable or provide api_key parameter."
            
            # Create messages list with system prompt and user message
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Prepare request payload
            payload = {
                "model": "grok-1",
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            # Set headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Make API request
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating response from Grok API: {e}")
            return f"I'm sorry, I encountered an error with the Grok API: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]], streaming: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate a response based on a conversation history using Grok API."""
        try:
            if not self.api_key:
                return "Error: Grok API key not provided. Please set GROK_API_KEY environment variable or provide api_key parameter."
            
            # Ensure system prompt is included
            if not any(msg.get("role") == "system" for msg in messages):
                messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            logger.info(f"Processing chat with {len(messages)} messages, streaming={streaming}")
            
            # Prepare request payload
            payload = {
                "model": "grok-1",
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": streaming
            }
            
            # Set headers with API key
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            if streaming:
                # For streaming, we need to handle the response differently
                def generate_stream():
                    with requests.post(self.api_url, json=payload, headers=headers, stream=True) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith('data: '):
                                    data = line[6:]
                                    if data == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data)
                                        content = chunk["choices"][0]["delta"].get("content", "")
                                        if content:
                                            yield content
                                    except Exception as e:
                                        logger.error(f"Error parsing streaming response: {e}")
                                        yield f"[Error: {str(e)}]"
                
                return generate_stream()
            else:
                # For non-streaming, make a regular request
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error in chat with Grok API: {e}")
            return f"I'm sorry, I encountered an error with the Grok API: {str(e)}"
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        info = super().model_info
        info.update({
            "model_type": "grok",
            "api_url": self.api_url,
            "has_api_key": bool(self.api_key)
        })
        return info


class ModelManager:
    """Class for managing multiple model interfaces."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.current_model = None
        self.current_model_id = None
        logger.info("Initialized ModelManager")
    
    def add_model(self, model_id: str, model_interface: BaseModelInterface) -> None:
        """Add a model interface to the manager."""
        self.models[model_id] = model_interface
        logger.info(f"Added model with ID: {model_id}")
        
        # Set as current model if it's the first one
        if self.current_model is None:
            self.set_current_model(model_id)
    
    def set_current_model(self, model_id: str) -> None:
        """Set the current model to use for inference."""
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found")
        
        self.current_model = self.models[model_id]
        self.current_model_id = model_id
        logger.info(f"Set current model to: {model_id}")
    
    def get_model(self, model_id: str = None) -> BaseModelInterface:
        """Get a model interface by ID, or the current model if no ID is provided."""
        if model_id is None:
            if self.current_model is None:
                raise ValueError("No current model set")
            return self.current_model
        
        if model_id not in self.models:
            raise ValueError(f"Model with ID {model_id} not found")
        
        return self.models[model_id]
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        return {model_id: model.model_info for model_id, model in self.models.items()}
    
    def generate_response(self, user_message: str, model_id: str = None) -> str:
        """Generate a response using the specified model or the current model."""
        model = self.get_model(model_id)
        return model.generate_response(user_message)
    
    def chat(self, messages: List[Dict[str, str]], streaming: bool = False, model_id: str = None) -> Union[str, Generator[str, None, None]]:
        """Generate a response based on a conversation history using the specified model or the current model."""
        model = self.get_model(model_id)
        return model.chat(messages, streaming)
    
    @classmethod
    def create_huggingface_model(cls, model_id: str, model_path: str, device: str = None, 
                               system_prompt: str = None, max_new_tokens: int = 1024, 
                               temperature: float = 0.7, top_p: float = 0.9) -> "ModelManager":
        """Create a ModelManager with a HuggingFace model."""
        manager = cls()
        model = HuggingFaceModelInterface(
            model_path=model_path,
            device=device,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        manager.add_model(model_id, model)
        return manager
    
    @classmethod
    def create_grok_model(cls, model_id: str, api_key: str = None, 
                        system_prompt: str = None, max_new_tokens: int = 1024, 
                        temperature: float = 0.7, top_p: float = 0.9) -> "ModelManager":
        """Create a ModelManager with a Grok model."""
        manager = cls()
        model = GrokModelInterface(
            api_key=api_key,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        manager.add_model(model_id, model)
        return manager
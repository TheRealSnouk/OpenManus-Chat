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
OpenManus AI Model Inference Module

This module provides functionality for loading and running inference with OpenManus AI models.
Supports multiple model types including Hugging Face models, Grok, and other API-based models.
"""

import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union, Generator

from .model_manager import (
    ModelManager,
    HuggingFaceModelInterface,
    GrokModelInterface,
    DEFAULT_SYSTEM_PROMPTS
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPTS["default"]
DEVELOPER_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPTS["developer"]


class OpenManusAI:
    """Class for loading and running inference with OpenManus AI models."""
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model_type: str = "huggingface",
        api_key: str = None,
        developer_mode: bool = False
    ):
        """
        Initialize the OpenManus AI model.
        
        Args:
            model_path: Path to the model or model name on Hugging Face Hub
            device: Device to load the model on ("cuda", "cpu", etc.)
            system_prompt: System prompt to use for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            model_type: Type of model to use ("huggingface" or "grok")
            api_key: API key for API-based models like Grok
            developer_mode: Whether to use developer mode system prompt
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.system_prompt = DEVELOPER_SYSTEM_PROMPT if developer_mode else system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.api_key = api_key
        self.developer_mode = developer_mode
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Load the appropriate model based on model_type
        try:
            if self.model_type == "huggingface":
                logger.info(f"Loading HuggingFace model from {model_path} on {self.device}")
                model = HuggingFaceModelInterface(
                    model_path=model_path,
                    device=self.device,
                    system_prompt=self.system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                self.model_manager.add_model("default", model)
            elif self.model_type == "grok":
                logger.info("Initializing Grok model interface")
                model = GrokModelInterface(
                    api_key=api_key,
                    system_prompt=self.system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                self.model_manager.add_model("default", model)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Successfully initialized {model_type} model")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def add_model(self, model_id: str, model_type: str, model_path: str = None, 
                 api_key: str = None, system_prompt: str = None) -> None:
        """
        Add a new model to the model manager.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model ("huggingface" or "grok")
            model_path: Path to the model or model name (for HuggingFace models)
            api_key: API key (for API-based models like Grok)
            system_prompt: System prompt to use for the model
        """
        try:
            if model_type.lower() == "huggingface":
                if not model_path:
                    raise ValueError("model_path is required for HuggingFace models")
                
                model = HuggingFaceModelInterface(
                    model_path=model_path,
                    device=self.device,
                    system_prompt=system_prompt or self.system_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            elif model_type.lower() == "grok":
                model = GrokModelInterface(
                    api_key=api_key or self.api_key,
                    system_prompt=system_prompt or self.system_prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model_manager.add_model(model_id, model)
            logger.info(f"Added {model_type} model with ID: {model_id}")
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            raise
    
    def set_current_model(self, model_id: str) -> None:
        """
        Set the current model to use for inference.
        
        Args:
            model_id: ID of the model to use
        """
        self.model_manager.set_current_model(model_id)
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary of model IDs to model information
        """
        return self.model_manager.get_available_models()
    
    def generate_response(self, user_message: str, model_id: str = None) -> str:
        """
        Generate a response to a user message.
        
        Args:
            user_message: The user's message
            model_id: ID of the model to use (optional)
            
        Returns:
            The model's response
        """
        return self.model_manager.generate_response(user_message, model_id)
    
    def chat(self, messages: List[Dict[str, str]], streaming: bool = False, model_id: str = None) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response based on a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            streaming: Whether to stream the response token by token
            model_id: ID of the model to use (optional)
            
        Returns:
            The model's response as a string, or a generator yielding tokens if streaming=True
        """
        return self.model_manager.chat(messages, streaming, model_id)
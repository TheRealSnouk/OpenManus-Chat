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
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)

from .utils import DEFAULT_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SYSTEM_PROMPT = """You are OpenManus AI, an assistant specialized in reinforcement learning for LLM agent tuning.
You can answer questions about reinforcement learning, LLM agent tuning, and related topics.
Provide detailed and accurate information based on your training."""


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
        """
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading model from {model_path} on {self.device}")
        
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
        """
        Generate a response to a user message.
        
        Args:
            user_message: The user's message
            
        Returns:
            The model's response
        """
        try:
            # Create messages list with system prompt and user message
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Apply chat template and tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=self.generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on a conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The model's response
        """
        try:
            # Ensure system prompt is included
            if not any(msg.get("role") == "system" for msg in messages):
                messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            # Apply chat template and tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=self.generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
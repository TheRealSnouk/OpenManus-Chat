import streamlit as st
import os
import sys

# Add the project directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="OpenManus-RL UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description - simplified
st.title("OpenManus-RL")
st.markdown("**AI-powered LLM agent tuning platform**")

# Display the logo with smaller size
st.image("assets/manus.jpg", width=300)

# Sidebar for navigation with improved styling
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "🔧 SFT Training", "⚙️ GRPO Training", "📊 Dataset Viewer", "💬 Chat with AI", "🛠️ Developer Interface"],
    label_visibility="collapsed"
)

# Home page
if page == "Home":
    st.header("Welcome to OpenManus-RL")
    
    st.markdown("""
    OpenManus-RL is an extended version of the original OpenManus initiative. 
    It explores new paradigms for RL-based LLM agent tuning, particularly building upon foundations.
    
    ## Project Overview
    - Reinforcement Learning for LLM agents
    - Reasoning model exploration
    - Alternative rollout strategies
    - Environment and benchmark testing
    """)
    
    # Display the roadmap image
    st.subheader("Project Roadmap")
    st.image("assets/openmanus-roadmap.png")
    
    # Display the method overview image if it exists
    if os.path.exists("assets/method_overview.png"):
        st.subheader("Method Overview")
        st.image("assets/method_overview.png")

# SFT Training page
elif page == "SFT Training":
    st.header("Supervised Fine-Tuning")
    
    st.markdown("""
    Configure and run Supervised Fine-Tuning (SFT) for your model.
    """)
    
    # Model configuration
    st.subheader("Model Configuration")
    model_name = st.text_input("Model Name or Path", "Qwen/Qwen2.5-1.5B-Instruct")
    dataset_name = st.text_input("Dataset Name", "HuggingFaceH4/Bespoke-Stratos-17k")
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.text_input("Learning Rate", "2.0e-5")
        num_epochs = st.number_input("Number of Epochs", min_value=1, value=1)
        max_seq_length = st.number_input("Max Sequence Length", min_value=128, value=4096)
    with col2:
        batch_size = st.number_input("Per Device Train Batch Size", min_value=1, value=2)
        gradient_accumulation = st.number_input("Gradient Accumulation Steps", min_value=1, value=8)
        gradient_checkpointing = st.checkbox("Enable Gradient Checkpointing", value=True)
    
    # Output configuration
    st.subheader("Output Configuration")
    output_dir = st.text_input("Output Directory", "data/sft-output")
    
    # Run button
    if st.button("Run SFT Training"):
        command = f"""python -m openmanus_rl.sft \
            --model_name_or_path {model_name} \
            --dataset_name {dataset_name} \
            --learning_rate {learning_rate} \
            --num_train_epochs {num_epochs} \
            --packing \
            --max_seq_length {max_seq_length} \
            --per_device_train_batch_size {batch_size} \
            --gradient_accumulation_steps {gradient_accumulation} \
            {'--gradient_checkpointing' if gradient_checkpointing else ''} \
            --bf16 \
            --logging_steps 5 \
            --output_dir {output_dir}
        """
        st.code(command, language="bash")
        st.info("Copy and run this command in your terminal to start training.")

# GRPO Training page
elif page == "GRPO Training":
    st.header("Gradient-based Reinforcement for Policy Optimization")
    
    st.markdown("""
    Configure and run GRPO training for your model.
    """)
    
    # Model configuration
    st.subheader("Model Configuration")
    model_name = st.text_input("Model Name or Path", "Qwen/Qwen2.5-1.5B-Instruct")
    dataset_name = st.text_input("Dataset Name", "HuggingFaceH4/Bespoke-Stratos-17k")
    
    # Training parameters
    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.text_input("Learning Rate", "2.0e-5")
        num_epochs = st.number_input("Number of Epochs", min_value=1, value=1)
        max_seq_length = st.number_input("Max Sequence Length", min_value=128, value=4096)
    with col2:
        batch_size = st.number_input("Per Device Train Batch Size", min_value=1, value=2)
        gradient_accumulation = st.number_input("Gradient Accumulation Steps", min_value=1, value=8)
        gradient_checkpointing = st.checkbox("Enable Gradient Checkpointing", value=True)
    
    # Reward functions
    st.subheader("Reward Functions")
    reward_funcs = st.multiselect(
        "Select Reward Functions",
        ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty", "length", "tag_count", "trajectories_format"],
        default=["accuracy", "format", "tag_count"]
    )
    
    # Output configuration
    st.subheader("Output Configuration")
    output_dir = st.text_input("Output Directory", "data/grpo-output")
    
    # Run button
    if st.button("Run GRPO Training"):
        reward_funcs_str = " ".join([f"--reward_funcs {func}" for func in reward_funcs])
        command = f"""python -m openmanus_rl.grpo \
            --model_name_or_path {model_name} \
            --dataset_name {dataset_name} \
            --learning_rate {learning_rate} \
            --num_train_epochs {num_epochs} \
            --max_seq_length {max_seq_length} \
            --per_device_train_batch_size {batch_size} \
            --gradient_accumulation_steps {gradient_accumulation} \
            {'--gradient_checkpointing' if gradient_checkpointing else ''} \
            --bf16 \
            --logging_steps 5 \
            {reward_funcs_str} \
            --output_dir {output_dir}
        """
        st.code(command, language="bash")
        st.info("Copy and run this command in your terminal to start training.")

# Dataset Viewer page
elif page == "Dataset Viewer":
    st.header("Dataset Viewer")
    
    st.markdown("""
    View and explore the OpenManus-RL dataset.
    """)
    
    st.info("The dataset is available on Hugging Face: [OpenManus-RL Dataset](https://huggingface.co/datasets/CharlieDreemur/OpenManus-RL)")
    
    # Placeholder for dataset viewing functionality
    st.warning("Dataset viewing functionality will be implemented in a future update.")

# Chat with AI page
elif page == "Chat with AI":
    st.header("Chat with OpenManus AI")
    
    st.markdown("""
    Interact with the OpenManus AI model through this chat interface.
    Ask questions about reinforcement learning, LLM agent tuning, or any other topic.
    """)
    
    # Initialize model in session state if it doesn't exist
    if "openmanus_model" not in st.session_state:
        # Check for required dependencies
        missing_deps = []
        try:
            import accelerate
        except ImportError:
            missing_deps.append("accelerate")
            
        # If dependencies are missing, show installation instructions
        if missing_deps:
            deps_str = ", ".join(missing_deps)
            st.error(f"Error: The following packages are required: {deps_str}")
            st.info(f"Please install them using: pip install {deps_str}")
            st.session_state.openmanus_model = None
        else:
            # Load the model if dependencies are available
            try:
                from openmanus_rl.model import OpenManusAI
                
                # Get model settings from session state or use defaults
                model_type = st.session_state.get('model_type', 'huggingface')
                model_path = st.session_state.get('model_path', 'Qwen/Qwen2.5-1.5B-Instruct')
                api_key = st.session_state.get('api_key', None)
                max_tokens = st.session_state.get('max_tokens', 1024)
                temperature = st.session_state.get('temperature', 0.7)
                top_p = st.session_state.get('top_p', 0.9)
                developer_mode = st.session_state.get('developer_mode', False)
                
                with st.status("Loading OpenManus AI model...", expanded=True) as status:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write("⚙️ This may take a moment. Please wait...")
                    with col2:
                        st.markdown("<div style='text-align: right;'>⏳</div>", unsafe_allow_html=True)
                    progress = st.progress(0)
                    for i in range(0, 101, 25):
                        progress.progress(i)
                        if i == 25:
                            status.update(label="🔄 Initializing tokenizer...", state="running")
                        elif i == 50:
                            status.update(label="🧠 Loading model weights...", state="running")
                        elif i == 75:
                            # Initialize model with appropriate settings
                            st.session_state.openmanus_model = OpenManusAI(
                                model_path=model_path,
                                model_type=model_type,
                                api_key=api_key,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                developer_mode=developer_mode
                            )
                            status.update(label="⚡ Optimizing model...", state="running")
                    
                    # Initialize session state variables
                    st.session_state.use_streaming = True
                    st.session_state.available_models = st.session_state.openmanus_model.get_available_models()
                    st.session_state.current_model_id = "default"
                    
                    # Show model type in status message
                    model_type_display = "HuggingFace" if model_type == "huggingface" else "Grok"
                    dev_mode_indicator = " (Developer Mode)" if developer_mode else ""
                    status.update(label=f"✅ {model_type_display} model loaded successfully{dev_mode_indicator}!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.session_state.openmanus_model = None
    
    # Initialize chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.openmanus_model is not None:
                message_placeholder = st.empty()
                
                # Create an enhanced thinking status indicator with visual feedback
                with st.status("OpenManus AI is thinking...", expanded=True) as status:
                    # Initialize thinking stage
                    thinking_stage = st.empty()
                    thinking_stage.write("💭 Processing your request...")
                    
                    # Add a more visually appealing progress indicator
                    progress_bar = st.progress(0)
                    
                    try:
                        # Convert chat history to format expected by model
                        model_messages = [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.chat_history
                        ]
                        
                        # Update thinking stage based on model's processing stage
                        status.update(label="🧠 Analyzing your question...", state="running")
                        thinking_stage.write("🔍 Understanding context and formulating response...")
                        progress_bar.progress(25)
                        
                        # Initialize use_streaming if not in session state
                        if "use_streaming" not in st.session_state:
                            st.session_state.use_streaming = True
                            
                        # Check if streaming is enabled
                        if st.session_state.use_streaming:
                            # Get streaming response from model
                            response_text = ""
                            
                            # Update thinking stage for response generation
                            status.update(label="✨ Generating response...", state="running")
                            thinking_stage.write("⚡ Creating response for you...")
                            progress_bar.progress(50)
                            
                            streamer = st.session_state.openmanus_model.chat(model_messages, streaming=True)
                            
                            # Update progress as generation begins
                            progress_bar.progress(75)
                            
                            # Display the response as it's being generated with improved visual feedback
                            for token in streamer:
                                response_text += token
                                message_placeholder.markdown(response_text + "▌")
                            
                            # Final update without the cursor
                            message_placeholder.markdown(response_text)
                        else:
                            # Get non-streaming response from model
                            status.update(label="✨ Generating response...", state="running")
                            thinking_stage.write("⚡ Creating response for you...")
                            progress_bar.progress(50)
                            
                            response_text = st.session_state.openmanus_model.chat(model_messages, streaming=False)
                            progress_bar.progress(75)
                            message_placeholder.markdown(response_text)
                            
                        # Complete the progress and update status
                        progress_bar.progress(100)
                        status.update(label="✅ Response complete!", state="complete", expanded=False)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    except Exception as e:
                        error_msg = f"I'm sorry, I encountered an error: {str(e)}"
                        message_placeholder.markdown(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                        status.update(label="❌ Error occurred", state="error", expanded=True)
                        thinking_stage.write("😓 Something went wrong while generating the response.")
                        st.error("There was a problem generating the response. Please try again or check your model settings.")
                        progress_bar.progress(100)
            else:
                error_msg = "I'm sorry, I'm not available right now. Please check the error messages above."
                st.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Enhanced model settings with multi-LLM support and developer interface
    with st.sidebar.expander("⚙️ Model Settings"):
        st.markdown("### Model Configuration")
        
        # Model type selection
        model_type = st.selectbox(
            "Model Type", 
            ["HuggingFace", "Grok"], 
            index=0,
            help="Select the type of model to use"
        )
        
        # Model-specific settings
        if model_type == "HuggingFace":
            model_path = st.text_input(
                "Model Path", 
                "Qwen/Qwen2.5-1.5B-Instruct", 
                help="Enter the Hugging Face model path or local model directory"
            )
            api_key = None
        else:  # Grok
            model_path = None
            api_key = st.text_input(
                "Grok API Key", 
                type="password",
                help="Enter your Grok API key. You can also set the GROK_API_KEY environment variable."
            )
            if not api_key:
                st.info("💡 If no API key is provided, the system will try to use the GROK_API_KEY environment variable.")
        
        # Developer mode toggle
        developer_mode = st.toggle(
            "Developer Mode", 
            False,
            help="Enable developer mode for more technical responses and implementation details"
        )
        
        st.markdown("### Generation Parameters")
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature", 
                0.0, 1.0, 0.7, 0.1, 
                help="Higher values produce more diverse outputs"
            )
            use_streaming = st.toggle(
                "Enable streaming", 
                True, 
                help="Show response as it's being generated"
            )
        with col2:
            max_tokens = st.number_input(
                "Max Tokens", 
                64, 4096, 1024, 
                help="Maximum length of generated response"
            )
            top_p = st.slider(
                "Top P", 
                0.0, 1.0, 0.9, 0.1, 
                help="Nucleus sampling parameter"
            )
        
        # Model management
        if "available_models" not in st.session_state:
            st.session_state.available_models = {}
            st.session_state.current_model_id = "default"
        
        # Display available models if any
        if len(st.session_state.available_models) > 1:  # More than just the default model
            st.markdown("### Available Models")
            current_model = st.selectbox(
                "Select Model",
                list(st.session_state.available_models.keys()),
                index=list(st.session_state.available_models.keys()).index(st.session_state.current_model_id),
                help="Select which loaded model to use"
            )
            
            if current_model != st.session_state.current_model_id and st.session_state.openmanus_model is not None:
                st.session_state.current_model_id = current_model
                st.session_state.openmanus_model.set_current_model(current_model)
                st.success(f"✅ Switched to model: {current_model}")
        
        # Add model name field for adding new models
        with st.expander("Add New Model"):
            new_model_id = st.text_input("Model ID", placeholder="Enter a unique identifier for this model")
            new_model_type = st.selectbox("Model Type", ["HuggingFace", "Grok"], key="new_model_type")
            
            if new_model_type == "HuggingFace":
                new_model_path = st.text_input("Model Path", placeholder="Enter model path or name")
                new_api_key = None
            else:  # Grok
                new_model_path = None
                new_api_key = st.text_input("API Key", type="password", placeholder="Enter API key")
            
            if st.button("Add Model", use_container_width=True):
                if new_model_id and (new_model_path or new_model_type == "Grok"):
                    try:
                        if st.session_state.openmanus_model is not None:
                            st.session_state.openmanus_model.add_model(
                                model_id=new_model_id,
                                model_type=new_model_type.lower(),
                                model_path=new_model_path,
                                api_key=new_api_key
                            )
                            st.session_state.available_models = st.session_state.openmanus_model.get_available_models()
                            st.success(f"✅ Added new model: {new_model_id}")
                    except Exception as e:
                        st.error(f"Error adding model: {str(e)}")
                else:
                    st.warning("Please fill in all required fields")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Settings", use_container_width=True):
                try:
                    from openmanus_rl.model import OpenManusAI
                    with st.spinner("Loading model with new settings..."):
                        st.session_state.openmanus_model = OpenManusAI(
                            model_path=model_path,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            model_type=model_type.lower(),
                            api_key=api_key,
                            developer_mode=developer_mode
                        )
                        st.session_state.use_streaming = use_streaming
                        st.session_state.available_models = st.session_state.openmanus_model.get_available_models()
                        st.session_state.current_model_id = "default"
                        st.success("✅ Settings applied successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.session_state.openmanus_model = None
        with col2:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.experimental_rerun()
    
    # Remove the separate clear chat history button since we moved it to the settings panel

# Developer Interface page
elif page == "🛠️ Developer Interface":
    st.header("Developer Interface")
    
    st.markdown("""
    This interface provides advanced tools and information for developers working with OpenManus-RL.
    Monitor model performance, debug issues, and access detailed model information.
    """)
    
    # Check if model is loaded
    if "openmanus_model" not in st.session_state or st.session_state.openmanus_model is None:
        st.warning("⚠️ No model is currently loaded. Please load a model in the Chat interface first.")
    else:
        # Create tabs for different developer tools
        dev_tabs = st.tabs(["📊 Model Info", "🔍 Debug Console", "⚡ Performance", "🧪 Test Suite"])
        
        # Model Info Tab
        with dev_tabs[0]:
            st.subheader("Available Models")
            
            # Get available models
            available_models = st.session_state.openmanus_model.get_available_models()
            current_model_id = st.session_state.current_model_id
            
            # Display model information in an expandable table
            for model_id, model_info in available_models.items():
                with st.expander(f"{model_id}{' (Current)' if model_id == current_model_id else ''}", expanded=(model_id == current_model_id)):
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Basic Information**")
                        st.markdown(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                        if model_info.get('model_type') == 'huggingface':
                            st.markdown(f"**Model Path:** {model_info.get('model_path', 'N/A')}")
                            st.markdown(f"**Device:** {model_info.get('device', 'N/A')}")
                        elif model_info.get('model_type') == 'grok':
                            st.markdown(f"**API URL:** {model_info.get('api_url', 'N/A')}")
                            st.markdown(f"**API Key Set:** {'Yes' if model_info.get('has_api_key', False) else 'No'}")
                    
                    with col2:
                        st.markdown("**Generation Parameters**")
                        st.markdown(f"**Temperature:** {model_info.get('temperature', 'N/A')}")
                        st.markdown(f"**Top P:** {model_info.get('top_p', 'N/A')}")
                        st.markdown(f"**Max New Tokens:** {model_info.get('max_new_tokens', 'N/A')}")
                    
                    # System prompt display
                    st.markdown("**System Prompt:**")
                    st.text_area("System Prompt", model_info.get('system_prompt', 'N/A'), height=150, disabled=True, key=f"sys_prompt_{model_id}")
        
        # Debug Console Tab
        with dev_tabs[1]:
            st.subheader("Debug Console")
            
            # Create a debug message input
            debug_message = st.text_area("Test Prompt", "", height=100, placeholder="Enter a test prompt to see how the model processes it...")
            
            col1, col2 = st.columns(2)
            with col1:
                debug_model = st.selectbox("Model", list(available_models.keys()), index=list(available_models.keys()).index(current_model_id))
            with col2:
                debug_streaming = st.checkbox("Enable Streaming", True)
            
            if st.button("Run Test", use_container_width=True):
                if debug_message:
                    with st.status("Processing test prompt...", expanded=True) as status:
                        try:
                            # Create a single message for testing
                            test_messages = [{"role": "user", "content": debug_message}]
                            
                            # Display the raw input
                            st.markdown("**Raw Input:**")
                            st.json(test_messages)
                            
                            # Process with the model
                            status.update(label="Generating response...", state="running")
                            
                            if debug_streaming:
                                st.markdown("**Streaming Response:**")
                                response_container = st.empty()
                                response_text = ""
                                
                                streamer = st.session_state.openmanus_model.chat(test_messages, streaming=True, model_id=debug_model)
                                for token in streamer:
                                    response_text += token
                                    response_container.markdown(response_text + "▌")
                                response_container.markdown(response_text)
                            else:
                                response_text = st.session_state.openmanus_model.chat(test_messages, streaming=False, model_id=debug_model)
                                st.markdown("**Response:**")
                                st.markdown(response_text)
                            
                            status.update(label="Test completed", state="complete")
                        except Exception as e:
                            st.error(f"Error during test: {str(e)}")
                            status.update(label="Test failed", state="error")
                else:
                    st.warning("Please enter a test prompt")
        
        # Performance Tab
        with dev_tabs[2]:
            st.subheader("Performance Metrics")
            
            # Simulated performance metrics
            st.markdown("**Response Time Metrics**")
            
            # Create sample data for demonstration
            import numpy as np
            import pandas as pd
            import time
            
            # Generate sample response times
            if "perf_history" not in st.session_state:
                st.session_state.perf_history = []
                st.session_state.perf_last_update = time.time()
            
            # Add a new performance test button
            if st.button("Run Performance Test"):
                with st.spinner("Running performance test..."):
                    # Simple test prompt
                    test_prompt = "Explain reinforcement learning in one paragraph."
                    test_messages = [{"role": "user", "content": test_prompt}]
                    
                    # Measure response time
                    start_time = time.time()
                    _ = st.session_state.openmanus_model.chat(test_messages, streaming=False)
                    end_time = time.time()
                    
                    # Calculate metrics
                    response_time = end_time - start_time
                    tokens_per_second = len(test_prompt.split()) / response_time
                    
                    # Add to history
                    st.session_state.perf_history.append({
                        "timestamp": time.time(),
                        "response_time": response_time,
                        "tokens_per_second": tokens_per_second
                    })
                    st.session_state.perf_last_update = time.time()
                    
                    st.success(f"Test completed in {response_time:.2f} seconds")
            
            # Display performance history if available
            if st.session_state.perf_history:
                # Convert to DataFrame for display
                perf_df = pd.DataFrame(st.session_state.perf_history)
                perf_df["test_number"] = range(1, len(perf_df) + 1)
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Response Time", f"{perf_df['response_time'].mean():.2f}s")
                with col2:
                    st.metric("Average Tokens/Second", f"{perf_df['tokens_per_second'].mean():.2f}")
                
                # Display chart
                st.line_chart(perf_df, x="test_number", y="response_time")
            else:
                st.info("No performance data available. Run a test to collect metrics.")
        
        # Test Suite Tab
        with dev_tabs[3]:
            st.subheader("Model Test Suite")
            
            st.markdown("""
            Run standardized tests to evaluate model performance across different tasks.
            These tests help identify strengths and weaknesses in the model's capabilities.
            """)
            
            # Define test categories
            test_categories = {
                "Reasoning": [
                    "Explain the concept of reinforcement learning to a beginner.",
                    "Compare and contrast supervised and unsupervised learning."
                ],
                "Code Generation": [
                    "Write a Python function to calculate the Fibonacci sequence.",
                    "Create a simple neural network using PyTorch."
                ],
                "Problem Solving": [
                    "How would you implement a reward function for an LLM agent?",
                    "Describe a method to prevent reward hacking in RL systems."
                ]
            }
            
            # Test selection
            test_category = st.selectbox("Test Category", list(test_categories.keys()))
            test_prompt = st.selectbox("Test Prompt", test_categories[test_category])
            
            # Custom test option
            use_custom_test = st.checkbox("Use Custom Test")
            if use_custom_test:
                test_prompt = st.text_area("Custom Test Prompt", "", height=100)
            
            # Run test button
            if st.button("Run Test Suite", use_container_width=True):
                if test_prompt:
                    with st.status("Running test suite...", expanded=True) as status:
                        try:
                            # Create test message
                            test_messages = [{"role": "user", "content": test_prompt}]
                            
                            # Get response from model
                            response = st.session_state.openmanus_model.chat(test_messages, streaming=False)
                            
                            # Display results
                            st.markdown("**Test Results:**")
                            st.markdown(f"**Prompt:** {test_prompt}")
                            st.markdown("**Response:**")
                            st.markdown(response)
                            
                            # Simple evaluation metrics
                            response_length = len(response)
                            word_count = len(response.split())
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Response Length", f"{response_length} chars")
                            with col2:
                                st.metric("Word Count", word_count)
                            with col3:
                                st.metric("Avg Word Length", f"{response_length/max(1, word_count):.1f} chars")
                            
                            status.update(label="Test completed", state="complete")
                        except Exception as e:
                            st.error(f"Error during test: {str(e)}")
                            status.update(label="Test failed", state="error")
                else:
                    st.warning("Please select or enter a test prompt")

# Simple footer
st.markdown("---")
st.markdown("© OpenManus-RL Team | [GitHub](https://github.com/mannaandpoem/OpenManus)")
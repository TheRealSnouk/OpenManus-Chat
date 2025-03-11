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

# Title and description
st.title("OpenManus-RL")
st.markdown("""
**An open-source initiative for RL-based LLM agent tuning**  
Collaboratively led by **Ulab-UIUC** and **MetaGPT**
""")

# Display the logo
st.image("assets/manus.jpg", width=400)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "SFT Training", "GRPO Training", "Dataset Viewer"]
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

# Footer
st.markdown("---")
st.markdown("""
### OpenManus-RL Team

A collaboration between Ulab-UIUC and MetaGPT

[GitHub Repository](https://github.com/mannaandpoem/OpenManus) | [Documentation](https://github.com/mannaandpoem/OpenManus)
""")
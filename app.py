import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from chatbot_engine import ChatBot

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for a more polished look
st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stChatInputContainer {
            padding-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    # Initialize the backend engine
    st.session_state.chatbot = ChatBot()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings ⚙️")
    st.markdown("Configure your AI assistant parameters below.")

    # Temperature Slider
    temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                     help="Higher values make the output more random, lower values make it more deterministic.")

    # Update chatbot parameters if temperature changes
    st.session_state.chatbot.update_params(temperature=temp)

    st.divider()

    # Clear Chat Button
    if st.button("Clear Chat History 🗑️", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI ---
st.title("Modern AI Assistant 🤖")
st.caption("Powered by Gemma 3 and LangChain")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    # 1. Display user message immediately
    st.chat_message("user").markdown(prompt)

    # 2. Add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate AI response
    with st.chat_message("assistant"):
        # Convert session state history to LangChain message objects
        lc_messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        try:
            with st.spinner("Thinking..."):
                # Use st.write_stream for the "typing" effect
                # The chatbot_engine.get_response returns a generator
                full_response = st.write_stream(st.session_state.chatbot.get_response(lc_messages))

            # 4. Save AI response to session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "connection" in str(e).lower() or "ollama" in str(e).lower():
                st.info("💡 Tip: Make sure Ollama is running locally on your machine.")

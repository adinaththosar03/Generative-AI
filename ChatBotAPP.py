import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Hugging Face model setup
my_hf_api_token = "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"
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=my_hf_api_token,
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Initialize Streamlit app
st.set_page_config(page_title="Simple Chatbot")
st.title("🤖 LangChain + HuggingFace Chatbot")
st.markdown("Chat with **LLaMA 3.1 8B Instruct**")

# Session state to store history
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Input box
user_input = st.text_input("You:", key="user_input", placeholder="Type your message and press Enter...")

# On submit
if user_input:
    # Prepare LangChain format history
    chat_history_local = [SystemMessage(content="You are a helpful assistant.")]
    for msg in st.session_state.history:
        if msg["role"] == "user":
            chat_history_local.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history_local.append(AIMessage(content=msg["content"]))

    # Add latest user message
    chat_history_local.append(HumanMessage(content=user_input))

    # Get model response
    result = model.invoke(chat_history_local)

    # Update history
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": result.content})

# Display chat history
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"**Bot:** {msg['content']}")

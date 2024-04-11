import streamlit as st
from graphRAG import GraphRAG
import os

model = GraphRAG()
get_response = model.get_response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

conversation_file_path = "conversation.txt"


# Create a sidebar for navigation
# st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Methods", ["Graph RAG with Documents",
                                              "Graph RAG with ER", ])





# Create a sidebar for navigation
# st.sidebar.title("Navigation")
# selected_page = st.sidebar.radio("Go to", ["Chat", "Knowledge Graph", "About"])

# Chat Page
if selected_page == "Graph RAG with Documents":
    st.title("Ask me about Inflammation!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get model response
        response, _, _ = get_response(prompt)
        # response = ""

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        with open(conversation_file_path, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")


# Knowledge Graph Page
elif selected_page == "Graph RAG with ER":
    st.title("Ask me about Inflammation!")
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get model response
        # response = get_response(prompt)
        response = ""

        # Display model response in chat message container
        with st.chat_message("bot"):
            st.markdown(response)

        # Add model response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        with open(conversation_file_path, "a") as file:
            file.write(f"user: {prompt}\n")
            file.write(f"bot: {response}\n\n")



# Add the document upload at the end of the sidebar
st.sidebar.header("Upload your document here")
uploaded_file = st.sidebar.file_uploader("", type=["pdf", "txt"])
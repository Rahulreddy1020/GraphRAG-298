import streamlit as st
import networkx as nx
from pyvis.network import Network
# from graphRAG import GraphRAG
import os
from neo4j import GraphDatabase

uri = "neo4j://localhost:7687" # replace with your Neo4j instance URI
user = "neo4j"  # replace with your username
password = "qwerty_102030"  # replace with your password

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_data(query):
    with driver.session() as session:
        result = session.run(query)
        data = []
        for record in result:
            nodes = [node.values() for node in record['nodes']]
            relationships = [relationship.values() for relationship in record['relationships']]
            data.append({'nodes': nodes, 'relationships': relationships})
    return data


def create_graph(data):
    G = nx.Graph()
    for item in data:
        for node in item['nodes']:
            if len(node) > 0:  # Check that node has at least one element
                G.add_node(list(node)[0])  # Convert dict_values to list before indexing
        for relationship in item['relationships']:
            if len(relationship) > 1:  # Check that relationship has at least two elements
                G.add_edge(list(relationship)[0], list(relationship)[1])  # Convert dict_values to list before indexing
    return G

def create_network(G):
    net = Network(notebook=True)
    for node in G.nodes:
        net.add_node(node)
    for edge in G.edges:
        net.add_edge(edge[0], edge[1])
    return net

# Use a Cypher query to get data from your Neo4j database


# model = GraphRAG()
# get_response = model.get_response

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

conversation_file_path = "conversation.txt"


# Create a sidebar for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Knowledge graph with Documents", "Knowledge Graph with ER", "ER"])





# Create a sidebar for navigation
# st.sidebar.title("Navigation")
# selected_page = st.sidebar.radio("Go to", ["Chat", "Knowledge Graph", "About"])

# Chat Page
if selected_page == "Knowledge graph with Documents":
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

if selected_page == "ER":

    # Fetch data from Neo4j
    query = """
    MATCH path = (n)-[r]->(m)
    RETURN 
        collect(distinct n) as nodes, 
        collect(distinct r) as relationships
    LIMIT 5
    """
    data = get_data(query)

    # Create a NetworkX graph from the data
    G = create_graph(data)

    # Create a Pyvis network from the NetworkX graph
    net = create_network(G)

    # Save the network to a HTML file
    net.save_graph("graph.html")

    # Display the network in Streamlit
    st.components.v1.html(open("graph.html", "r").read(), width=800, height=600)

# Add the document upload at the end of the sidebar
st.sidebar.header("Upload your document here")
uploaded_files = st.sidebar.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True)


# # Fetch data from Neo4j
# query = """
# MATCH path = (n)-[r]->(m)
# RETURN 
#     collect(distinct n) as nodes, 
#     collect(distinct r) as relationships
# LIMIT 5
# """
# data = get_data(query)

# # Create a NetworkX graph from the data
# G = create_graph(data)

# # Create a Pyvis network from the NetworkX graph
# net = create_network(G)

# # Save the network to a HTML file
# net.save_graph("graph.html")

# # Display the network in Streamlit
# st.components.v1.html(open("graph.html", "r").read(), width=800, height=600)
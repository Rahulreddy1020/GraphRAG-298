import os
import dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
from helper import Helper



hlp = Helper()
hlp.clear_log_file()
log = hlp.get_logger()


dotenv.load_dotenv()
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(username, password))

def run_query(query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return result.data()
    except Exception as e:
        raise(e)

# Define the Cypher query to retrieve the graph data
query = "MATCH (c)-[r]->(f) RETURN c, r, f LIMIT 10"

# Execute the query and retrieve the results
results = run_query(query)

# Create a PyVis network object
net = Network(height='750px', width='100%', directed=True)

# Add nodes and edges to the PyVis network
for record in results:
    source_node = record["c"]
    target_node = record["f"]
    relationship = record["r"]

    # log.info(f"Source Node: {source_node.keys()}")
    # log.info(f"Target Node: {target_node.keys()}")
    # log.info(f"{relationship}")

    # source_id = sou
    target_id = str(target_node.article)
    relationship_type = type(relationship).__name__

    # net.add_node(source_id, label=source_node["name"] if "name" in source_node else "")
    net.add_node(target_id, label=target_node["name"] if "name" in target_node else "")
    # net.add_edge(source_id, target_id, title=relationship_type)

# Save the interactive graph visualization as an HTML file
net.show("graph.html")

# Close the Neo4j driver
driver.close()
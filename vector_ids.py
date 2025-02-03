from pymilvus import connections, Collection, utility

# Connect to your Milvus instance
connections.connect("default", host='localhost', port='19530')

# Define your collection name
collection_name = "betaCollection"

# Check if the collection exists
if collection_name in utility.list_collections():
    collection = Collection(collection_name)

    # Define a simple query expression to retrieve all records excluding empty text
query = "pk >= 0 AND text != ''"

try:
    # Query the collection
    results = collection.query(expr=query, output_fields=["pk", "text"])
    
    # Process the results
    for result in results:
        try:
            print(f"ID: {result['pk']}, Chunk: {result['text'].encode('utf-8', 'replace').decode('utf-8')}")
        except Exception as e:
            print(f"An error occurred while printing: {str(e)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")

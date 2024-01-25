from flask import Flask, request, jsonify
import json
from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Function to load data from a JSON file
def load_data(file_path):
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as file:
        return json.load(file)

# Load groups and threats data from JSON files
print("Loading groups and threats data")
groups_data = load_data('groups.json')
threat_data_dict = {entry['id']: entry for entry in load_data('data.json')}

# Function to generate embeddings using Roberta model
def get_roberta_embeddings(texts, model, tokenizer):
    model.eval()
    with torch.no_grad():
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        return embeddings

# Function to identify threats based on a Applicatioen Description
def identify_threats(application_description, groups_data, tokenizer, model):
    # Keywords indicating a connection to the American government
    usa_keywords = {
        'United States', 'USA', 'US', 'U.S.', 'America', 'The States', 'Uncle Sam',
        'United States of America', 'The Union', 'US government', 'American government',
        'US military', 'Pentagon', 'White House', 'Department of Defense', 'Federal government', 'FBI', 'Army', 'CIA'
    }

    # Filter groups related to the American government
    usa_related_groups = [group for group in groups_data if any(keyword in " ".join(str(value) for key, value in group.items()) for keyword in usa_keywords)]

    # Enhance the query with additional context
    enhanced_query = "The following system is part of the United States Coast Guard: " + application_description
    print(f"Identifying threats for query: {enhanced_query}")
    query_embedding = get_roberta_embeddings([enhanced_query], model, tokenizer)[0]
    query_embedding_2d = np.expand_dims(query_embedding, axis=0)

    # Compute similarities with selected groups
    group_texts = [" ".join(str(value) for key, value in group.items() if key != 'associated_techniques') for group in usa_related_groups]
    group_embeddings = get_roberta_embeddings(group_texts, model, tokenizer)
    cosine_similarities = cosine_similarity(query_embedding_2d, np.vstack(group_embeddings)).flatten()

    # Select the top 5 most similar APT groups based on Application
    top_groups = []
    similar_groups_indices = cosine_similarities.argsort()[-5:][::-1]
    for index in similar_groups_indices:
        group = usa_related_groups[index]

        # Append all group details except 'associated_techniques'
        group_info = {}
        for key, value in group.items():
            if key != 'associated_techniques':
                group_info[key] = value

        # Add the cosine similarity score
        group_info['Score'] = float(cosine_similarities[index])

        top_groups.append(group_info)

    print("Groups Matched: \n")
    for threats in top_groups:
        print(threats)

    return top_groups

# Function to match threats with entity platforms
def match_threats_with_entities(entities, top_groups, groups_data, threat_data_dict):
    print("Matching threats with entity platforms")
    for entity in entities:
        entity_name = entity.get('name', 'Unknown Entity')
        entity_platforms = entity.get('platforms', '').split(', ')
        print(f"Processing entity: {entity_name}, Platforms: {entity_platforms} \n\n")

        if not any(platform.strip() for platform in entity_platforms):
            print(f"Entity '{entity_name}' has no platforms, skipping\n")
            continue

        entity['threats'] = []

        for group in top_groups:
            group_id = group.get('id')
            if group_id is None:
                print(f"Group ID missing in top_groups, skipping group.")
                continue

            group_name = group.get('name', 'Unknown Group')
            print(f"Current group being processed: {group_name}, ID: {group_id}")

            # Retrieve group data
            group_data = next((item for item in groups_data if item.get('id') == group_id), None)
            if not group_data:
                print(f"Group data not found for ID: {group_id}, skipping group.")
                continue

            for technique in group_data.get('associated_techniques', []):
                tech_id = technique.get('target_ID')
                if tech_id is None:
                    print(f"Technique ID missing in group_data, skipping technique.")
                    continue

                threat = threat_data_dict.get(tech_id, {})
                if not threat:
                    print(f"Threat data not found for ID: {tech_id}, skipping threat.")
                    continue

                if set(threat.get('platforms', [])).intersection(entity_platforms):
                    # Include all data from the threat
                    threat_summary = threat.copy()
                    # Append the associated APT group name
                    threat_summary['associated_apt'] = group_name

                    entity['threats'].append(threat_summary)


        print(f"End of processing for entity: {entity_name}.")

    return entities

# Define the route for processing the JSON data
@app.route('/process', methods=['POST'])
def process_json():
    try:
        print("Received POST request")
        data = request.get_json()

        # Extract the description from the first diagram in the 'detail' section
        diagrams = data.get('detail', {}).get('diagrams', [])
        if diagrams:
            user_query = diagrams[0].get('description', '')
        else:
            user_query = ''
        print("User query extracted:", user_query)

        # Initialize Roberta model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')

        # Identify top threats based on the user query
        top_groups = identify_threats(user_query, groups_data, tokenizer, model)

        # Extract entities from the payload and match threats with their platforms
        entities = []
        for diagram in diagrams:
            for cell in diagram.get('cells', []):
                if 'data' in cell:
                    entities.append(cell['data'])

            # Add the top 5 groups to each diagram
            summary_threat_groups = []
            for group in top_groups:
                # Append all group details except 'associated_techniques'
                group_summary = {key: value for key, value in group.items() if key != 'associated_techniques'}
                summary_threat_groups.append(group_summary)

            diagram['apt_groups'] = summary_threat_groups  # Appending the apt_groups to each diagram

        # Call match_threats_with_entities to modify the entities list by reference, adding matched threats
        match_threats_with_entities(entities, top_groups, groups_data, threat_data_dict)

        return jsonify(data)

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    print('about')
    return 'About'


@app.route('/observation-tuning')
def observation_tuning():
    # Read in observations and transcript json from body
    # Run tuning algorithm
    # Return tuned observations
    return 'Observation tuning'


@app.route('/process-data', methods=['POST', 'GET'])
def process_data():
    # Extract JSON data from request
    print('process data')
    data = request.get_json()
    transcript = data['transcript']
    observations = data['observations']

    print(transcript)
    print(observations)

    # Your processing code goes here
    # Adjust the observations using the transcript
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for transcript and observations
    # Assuming each entry in 'transcript' is a dictionary with a key 'words' that contains the text
    # Assuming each entry in 'transcript' is a dictionary with a key 'words' that contains the text
    transcript_embeddings = model.encode([" ".join([word['text'] for word in paragraph['words']]) for paragraph in transcript], convert_to_tensor=True)

    observation_embeddings = model.encode([observation['text'] for observation in observations], convert_to_tensor=True)

    for obs_embedding, observation in zip(observation_embeddings, observations):
        # Compute similarities
        similarities = util.pytorch_cos_sim(obs_embedding, transcript_embeddings)[0]
        max_similarity_index = similarities.argmax()

    if similarities[max_similarity_index] > 0.75:  # Choose a suitable threshold
        best_paragraph = transcript[max_similarity_index]
        print(best_paragraph)
        observation['playStartTime'] = best_paragraph['words'][0]['start']
        observation['playEndTime'] = min(best_paragraph['words'][-1]['end'], observation['playStartTime'] + 90000)  # 90 seconds


    # Return the adjusted observations as JSON
    return jsonify(observations)
    return 'hello world'




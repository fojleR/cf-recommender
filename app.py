# app.py
import os
import pickle
import time
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from urllib.parse import quote
import logging
from flask_cors import CORS # Import CORS

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable TensorFlow specific warnings or info messages if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = info, 2 = warning, 3 = error
tf.get_logger().setLevel('ERROR') # Use TF's logger setting

MODEL_PATH = 'problem_recommender.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
MAX_SEQUENCE_LEN = 664 # Corresponds to max_len in notebook cell 68

# --- Load Model and Tokenizer ---
logging.info(f"Loading model from {MODEL_PATH}...")
try:
    # GPU memory growth settings (optional but good practice)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logging.error(e)

    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
    # Determine model input length
    try:
        logging.info(f"Model expected input shape: {model.input_shape} (usually None, {MAX_SEQUENCE_LEN - 1})")
        MODEL_INPUT_LENGTH = MAX_SEQUENCE_LEN - 1
    except Exception as e:
         logging.warning(f"Could not determine model input shape automatically: {e}. Assuming input length {MAX_SEQUENCE_LEN - 1}")
         MODEL_INPUT_LENGTH = MAX_SEQUENCE_LEN - 1

except Exception as e:
    logging.error(f"Fatal error loading model: {e}", exc_info=True)
    exit()

logging.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    logging.info("Tokenizer loaded successfully.")
    if not hasattr(tokenizer, 'word_index') or not tokenizer.word_index:
         raise ValueError("Tokenizer does not have a valid word_index.")
    tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
    logging.info(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
except Exception as e:
    logging.error(f"Fatal error loading tokenizer: {e}", exc_info=True)
    exit()

logging.info("Model and Tokenizer loaded. Initializing Flask app...")
app = Flask(__name__)

# --- Initialize CORS ---
# Allows requests from all origins by default during development.
# For production, restrict origins like this:
# origins = ["http://localhost:3000", "https://your-deployed-frontend.com"]
# CORS(app, origins=origins, supports_credentials=True)
CORS(app) # Allows all origins for now

# --- Helper Functions ---

def convert_verdict(verdict):
    """Converts verdict string to 1 (OK) or 0."""
    return 1 if verdict == "OK" else 0

def clean_tag_string(tag_list):
    """Cleans a list of tags from Codeforces API."""
    if not isinstance(tag_list, list):
        return ""
    cleaned_tags = []
    for tag in tag_list:
        processed_tag = ""
        tag_lower = tag.lower()
        for char in tag_lower:
            if 'a' <= char <= 'z':
                processed_tag += char
            else:
                if processed_tag and processed_tag[-1] != ' ':
                     processed_tag += ' '
        cleaned_tag_parts = [part for part in processed_tag.strip().split() if part]
        cleaned_tags.extend(cleaned_tag_parts)
    return " ".join(cleaned_tags)

def apply_user_threshold(group, threshold=2):
    """Keeps max 'threshold' occurrences of each unique tag_feature per user."""
    tag_counts = {}
    result_indices = []
    for index in group.index:
        try:
            tag_feat_value = group.loc[index, 'tag_feature']
            if not isinstance(tag_feat_value, str):
                # This check should hopefully not be needed after the reset_index fix,
                # but kept as a safeguard.
                logging.warning(f"Row index {index}: Unexpected type for tag_feature: {type(tag_feat_value)}. Value: {tag_feat_value}. Skipping.")
                continue
            tag_feat = tag_feat_value
            if tag_feat not in tag_counts:
                tag_counts[tag_feat] = 0
            if tag_counts[tag_feat] < threshold:
                result_indices.append(index)
                tag_counts[tag_feat] += 1
        except KeyError:
             logging.warning(f"Row index {index}: 'tag_feature' column access failed.")
             continue
        except Exception as e:
             logging.error(f"Row index {index}: Unexpected error in apply_user_threshold: {e}", exc_info=True)
             continue
    if not result_indices:
        logging.warning("apply_user_threshold resulted in empty DataFrame.")
        return pd.DataFrame(columns=group.columns)
    else:
        return group.loc[result_indices]


def preprocess_user_data(user_handle):
    """Fetches and preprocesses Codeforces data for a given handle."""
    logging.info(f"Starting preprocessing for handle: {user_handle}")
    encoded_handle = quote(user_handle)

    # --- Fetch Submissions ---
    submission_url = f'https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=1000'
    try:
        logging.info(f"Fetching submissions from {submission_url}")
        submission_request = requests.get(submission_url, timeout=15)
        submission_request.raise_for_status()
        submission_data = submission_request.json()
        if submission_data.get('status') != 'OK':
            logging.error(f"Codeforces API error (submissions) for {user_handle}: {submission_data.get('comment', 'Unknown error')}")
            return None, None
        user_data = pd.DataFrame(submission_data.get('result', []))
        if user_data.empty:
             logging.warning(f"No submission data found for handle: {user_handle}")
             return None, None
        user_data['handle'] = user_handle
        logging.info(f"Fetched {len(user_data)} submissions for {user_handle}.")
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching submissions for {user_handle}.")
        return None, None
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request Error fetching submissions for {user_handle}: {e}")
        return None, None
    except ValueError as e:
        logging.error(f"JSON Decode Error (Submission) for {user_handle}: {e}")
        return None, None

    time.sleep(0.6)

    # --- Expand Problem Data ---
    try:
        if 'problem' not in user_data.columns:
             logging.error(f"'problem' column missing in submission data for {user_handle}.")
             return None, None
        problem_data = pd.json_normalize(user_data['problem'], errors='ignore')
        problem_cols_needed = ['contestId', 'index', 'name', 'rating', 'tags']
        for col in problem_cols_needed:
            if col not in problem_data.columns:
                problem_data[col] = None
        problem_data = problem_data[problem_cols_needed]
        problem_data = problem_data.rename(columns={'rating': 'problemRating'})
        original_cols_needed = ['creationTimeSeconds', 'verdict', 'handle']
        valid_original_cols = [col for col in original_cols_needed if col in user_data.columns]
        original_data_subset = user_data[valid_original_cols].copy()
        original_data_subset.reset_index(drop=True, inplace=True)
        problem_data.reset_index(drop=True, inplace=True)
        combined_data = pd.concat([original_data_subset, problem_data], axis=1)
        user_data = combined_data
    except Exception as e:
        logging.error(f"Error expanding/combining problem data for {user_handle}: {e}", exc_info=True)
        return None, None

    # --- Basic Cleaning & Feature Engineering ---
    logging.info(f"Columns before numeric conversion attempt: {user_data.columns.tolist()}")
    if 'problemRating' in user_data.columns:
        user_data['problemRating'] = pd.to_numeric(user_data['problemRating'], errors='coerce')
    if 'contestId' in user_data.columns:
        user_data['contestId'] = pd.to_numeric(user_data['contestId'], errors='coerce')
    if 'creationTimeSeconds' in user_data.columns:
        user_data['creationTimeSeconds'] = pd.to_numeric(user_data['creationTimeSeconds'], errors='coerce')

    essential_cols = ['contestId', 'index', 'name', 'problemRating', 'tags', 'creationTimeSeconds', 'verdict', 'handle']
    user_data.dropna(subset=[col for col in essential_cols if col in user_data.columns], inplace=True)
    if user_data.empty:
        logging.warning(f"No valid rows left after dropping NaNs in essential columns for {user_handle}.")
        return None, None

    user_data['problemRating'] = user_data['problemRating'].astype(int)
    user_data['contestId'] = user_data['contestId'].astype(int)
    user_data['creationTimeSeconds'] = user_data['creationTimeSeconds'].astype(int)
    user_data['index'] = user_data['index'].astype(str)
    user_data['name'] = user_data['name'].astype(str)

    user_data['ProblemID'] = user_data['contestId'].astype(str) + user_data['index']
    solved_problem_ids = set(user_data['ProblemID'].unique())

    user_data = user_data.sort_values(by='creationTimeSeconds', ascending=False)
    user_data = user_data.drop_duplicates(subset=['ProblemID'], keep='first')
    user_data['verdict'] = user_data['verdict'].apply(convert_verdict)
    user_data['clean_tags_str'] = user_data['tags'].apply(clean_tag_string)

    # Explode tags
    user_data['tags_list'] = user_data['clean_tags_str'].str.split()
    exploded_data = user_data.explode('tags_list')
    exploded_data = exploded_data.rename(columns={'tags_list': 'tag'})
    exploded_data = exploded_data.dropna(subset=['tag'])
    exploded_data = exploded_data[exploded_data['tag'] != '']

    # =============================================================
    # START FIX: Reset index after exploding and cleaning tags
    # =============================================================
    if not exploded_data.empty:
        exploded_data = exploded_data.reset_index(drop=True)
    # =============================================================
    # END FIX
    # =============================================================

    if exploded_data.empty: # Check again after potential reset
        logging.warning(f"No valid tags found after cleaning and exploding for {user_handle}.")
        return None, None

    # Create tag_feature
    exploded_data['tag_feature'] = exploded_data['tag'].astype(str) + exploded_data['problemRating'].astype(str)

    # Apply threshold
    limited_data = apply_user_threshold(exploded_data, threshold=2)
    if limited_data.empty:
        logging.warning(f"No data left after applying threshold for {user_handle}.")
        return None, None

    # Create final sequence
    final_sequence = " ".join(limited_data['tag_feature'].tolist())
    if not final_sequence:
        logging.warning(f"Final tag sequence is empty for {user_handle}.")
        return None, None

    logging.info(f"Generated sequence for {user_handle} with length: {len(final_sequence.split())}")
    return final_sequence, solved_problem_ids


def get_recommendations_from_codeforces(predicted_tags, solved_ids, num_recommendations=10):
    """Fetches problems from Codeforces matching predicted tags, excluding solved ones."""
    recommendations = []
    logging.info(f"Attempting to find problems for {len(predicted_tags)} predicted tags, excluding {len(solved_ids)} solved problems.")
    problemset_url = "https://codeforces.com/api/problemset.problems"
    try:
        logging.info("Fetching full problemset...")
        response = requests.get(problemset_url, timeout=20)
        response.raise_for_status()
        problemset_data = response.json()
        if problemset_data.get('status') != 'OK':
            logging.error(f"Failed to fetch problemset: {problemset_data.get('comment')}")
            return []
        all_problems = problemset_data['result']['problems']
        all_problem_stats = problemset_data['result']['problemStatistics']
        logging.info(f"Fetched {len(all_problems)} problems and {len(all_problem_stats)} stats.")
        stats_map = {f"{stat['contestId']}{stat['index']}": stat['solvedCount'] for stat in all_problem_stats}
    except requests.exceptions.Timeout:
         logging.error("Timeout fetching problemset.")
         return []
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error fetching problemset: {e}")
        return []
    except (ValueError, KeyError) as e:
        logging.error(f"Error parsing problemset data: {e}")
        return []

    found_problems_ids = set()
    for tag_feature in predicted_tags:
        if len(recommendations) >= num_recommendations: break
        try:
            rating_str = ""
            tag_part = ""
            for char in reversed(tag_feature):
                 if char.isdigit(): rating_str = char + rating_str
                 else: tag_part = tag_feature[:-len(rating_str)]; break
            if not tag_part and rating_str == tag_feature: tag_part = ""
            if not rating_str: continue
            rating = int(rating_str)
            tag = tag_part
        except ValueError:
            logging.warning(f"Could not parse tag/rating from predicted feature: {tag_feature}")
            continue

        candidate_problems = []
        for problem in all_problems:
            problem_rating = problem.get('rating')
            problem_tags = problem.get('tags', [])
            contest_id = problem.get('contestId')
            problem_index = problem.get('index')
            if (problem_rating == rating and
                (tag in problem_tags or tag == "") and
                contest_id is not None and
                problem_index is not None):
                problem_id = f"{contest_id}{problem_index}"
                if problem_id not in solved_ids and problem_id not in found_problems_ids:
                    problem_info = {
                        'contestId': contest_id, 'index': problem_index,
                        'name': problem.get('name', 'N/A'), 'rating': problem_rating,
                        'tags': problem_tags, 'url': f"https://codeforces.com/problemset/problem/{contest_id}/{problem_index}",
                        'solvedCount': stats_map.get(problem_id, 0)
                    }
                    candidate_problems.append(problem_info)

        candidate_problems.sort(key=lambda x: x.get('solvedCount', 0), reverse=True)
        for problem in candidate_problems:
             if len(recommendations) < num_recommendations:
                  recommendations.append(problem)
                  found_problems_ids.add(f"{problem['contestId']}{problem['index']}")
             else: break

    logging.info(f"Found {len(recommendations)} problem recommendations.")
    return recommendations

# --- Flask Routes ---

@app.route('/')
def home():
    return jsonify({"message": "Codeforces Problem Recommender API is running!"})

@app.route('/recommend/<handle>', methods=['GET'])
def recommend_problems_api(handle):
    """API endpoint to get recommendations for a Codeforces handle."""
    start_time = time.time()
    if not handle:
        return jsonify({"error": "Codeforces handle is required in the URL path (e.g., /recommend/YourHandle)"}), 400

    logging.info(f"\n--- New Request for handle: {handle} ---")

    # 1. Preprocess user data
    try:
        user_sequence, solved_problem_ids = preprocess_user_data(handle)
        if user_sequence is None or solved_problem_ids is None:
            logging.warning(f"Preprocessing returned None for handle '{handle}'.")
            status_code = 404 # Assume user not found or insufficient data
            return jsonify({"error": f"Could not process data for handle '{handle}'. User might not exist, have insufficient history/tags, or an API error occurred during fetch."}), status_code
    except Exception as e:
         logging.error(f"Unhandled exception during preprocessing for {handle}: {e}", exc_info=True)
         return jsonify({"error": "Internal server error during data preprocessing."}), 500


    # 2. Predict next tags iteratively
    predicted_tags = []
    current_sequence = user_sequence
    num_preds_to_generate = 20

    try:
        logging.info(f"Starting prediction loop for {handle}...")
        for i in range(num_preds_to_generate):
            token_text = tokenizer.texts_to_sequences([current_sequence])[0]
            padded_token_text = pad_sequences([token_text], maxlen=MODEL_INPUT_LENGTH, padding='pre')
            pred_probs = model.predict(padded_token_text, verbose=0)[0]
            pred_probs[0] = 0
            pos = np.argmax(pred_probs)
            predicted_tag = tokenizer.index_word.get(pos)
            if predicted_tag:
                if predicted_tag not in predicted_tags:
                     predicted_tags.append(predicted_tag)
                current_sequence += " " + predicted_tag
            else:
                 logging.warning(f"Predicted index {pos} not found in tokenizer.index_word. Stopping prediction.")
                 break
    except Exception as e:
        logging.error(f"Error during prediction loop for {handle}: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during model prediction."}), 500

    if not predicted_tags:
        logging.warning(f"Could not generate any tag predictions for {handle}.")
        # Return success but with empty recommendations and a message
        return jsonify({
             "user_handle": handle,
             "processing_time_seconds": round(time.time() - start_time, 2),
             "predicted_tags_analyzed": [],
             "recommendations": [],
             "message": "Could not generate any tag predictions based on user history."
             }), 200

    logging.info(f"Predicted top {len(predicted_tags)} tag features for {handle}: {predicted_tags}")

    # 3. Fetch actual problems based on predicted tags
    try:
        recommendations = get_recommendations_from_codeforces(predicted_tags, solved_problem_ids, num_recommendations=10)
    except Exception as e:
        logging.error(f"Error during problem fetching for {handle}: {e}", exc_info=True)
        # Still return success, but indicate fetching failed
        return jsonify({
             "user_handle": handle,
             "processing_time_seconds": round(time.time() - start_time, 2),
             "predicted_tags_analyzed": predicted_tags,
             "recommendations": [],
             "error": "An error occurred while fetching problems from Codeforces based on predictions." # Add specific error message
             }), 500 # Return 500 as problem fetching is part of the core function

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    logging.info(f"Total request processing time for {handle}: {processing_time} seconds.")

    # 4. Return recommendations
    response_data = {
        "user_handle": handle,
        "processing_time_seconds": processing_time,
        "predicted_tags_analyzed": predicted_tags,
        "recommendations": recommendations
    }
    if not recommendations:
         response_data["message"] = "Could not find suitable unsolved problems matching the predicted tags."

    return jsonify(response_data), 200

# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logging.info(f"Starting Flask server on host 0.0.0.0 port {port}...")
    # Use a production server like Waitress or Gunicorn instead of app.run for real deployment
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=port, debug=False) # Turn debug=False for production
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
from flask_cors import CORS # <--- Import CORS

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable TensorFlow specific warnings or info messages if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = info, 2 = warning, 3 = error
tf.get_logger().setLevel('ERROR') # Use TF's logger setting

MODEL_PATH = 'problem_recommender.h5'
TOKENIZER_PATH = 'tokenizer.pkl'
# MAX_SEQUENCE_LEN should be the length used for padding the sequences fed into the model during training.
# If X_train had shape (num_samples, sequence_length), then MAX_SEQUENCE_LEN is sequence_length + 1
# From notebook cell 68, max_len was 664. X was padded_input_sequences[:,:-1], so X had length 663.
# The input to the model needs length 663.
# MAX_SEQUENCE_LEN is used to pad the *full* sequence before potentially trimming for prediction input.
# Let's keep it consistent with the notebook's max_len for the full sequence.
MAX_SEQUENCE_LEN = 664 # Corresponds to max_len in notebook cell 68

# --- Load Model and Tokenizer ---
# Load model and tokenizer globally on startup
logging.info(f"Loading model from {MODEL_PATH}...")
try:
    # Ensure TF doesn't allocate all GPU memory if running on GPU server
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.error(e)

    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
    # Log model input shape if possible
    try:
        # The input shape derived from training X (padded[:,:-1]) is likely (None, MAX_SEQUENCE_LEN - 1)
        logging.info(f"Model expected input shape: {model.input_shape} (usually None, {MAX_SEQUENCE_LEN - 1})")
        MODEL_INPUT_LENGTH = MAX_SEQUENCE_LEN - 1 # Define the actual length the model expects
    except Exception as e:
         logging.warning(f"Could not determine model input shape automatically: {e}. Assuming input length {MAX_SEQUENCE_LEN - 1}")
         MODEL_INPUT_LENGTH = MAX_SEQUENCE_LEN - 1

except Exception as e:
    logging.error(f"Fatal error loading model: {e}", exc_info=True)
    exit() # Exit if model fails to load

logging.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    logging.info("Tokenizer loaded successfully.")
    # Ensure word_index exists and create index_word map
    if not hasattr(tokenizer, 'word_index') or not tokenizer.word_index:
         raise ValueError("Tokenizer does not have a valid word_index.")
    tokenizer.index_word = {index: word for word, index in tokenizer.word_index.items()}
    logging.info(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
except Exception as e:
    logging.error(f"Fatal error loading tokenizer: {e}", exc_info=True)
    exit() # Exit if tokenizer fails to load

logging.info("Model and Tokenizer loaded. Initializing Flask app...")
app = Flask(__name__)

# --- Initialize CORS ---
# Allows requests from all origins by default during development.
# For production, restrict origins like this:
# Replace '*' with your frontend origin(s)
# origins = ["http://localhost:3000", "https://your-deployed-frontend.com"]
# CORS(app, origins=origins, supports_credentials=True) # If you need cookies/auth headers
CORS(app) # <--- Allows all origins for now

# --- Helper Functions (Adapted from Notebook & Refined) ---

def convert_verdict(verdict):
    """Converts verdict string to 1 (OK) or 0."""
    return 1 if verdict == "OK" else 0

def clean_tag_string(tag_list):
    """
    Cleans a list of tags from Codeforces API to match notebook's `convert` function logic.
    Replaces non-alpha characters (like commas, hyphens) with space, keeps a-z, lowercases.
    """
    if not isinstance(tag_list, list):
        return ""
    cleaned_tags = []
    for tag in tag_list:
        processed_tag = ""
        tag_lower = tag.lower() # Process in lowercase
        for char in tag_lower:
            if 'a' <= char <= 'z':
                processed_tag += char
            else:
                 # Replace non-alpha with space, but avoid multiple spaces
                if processed_tag and processed_tag[-1] != ' ':
                     processed_tag += ' '
        # Handle potential multiple words from conversion (e.g., "two pointers" -> "two pointers")
        cleaned_tag_parts = [part for part in processed_tag.strip().split() if part]
        cleaned_tags.extend(cleaned_tag_parts)

    return " ".join(cleaned_tags) # Join cleaned parts


def apply_user_threshold(group, threshold=2):
    """Keeps max 'threshold' occurrences of each unique tag_feature per user."""
    tag_counts = {}
    result_indices = []
    # Iterate in reverse chronological order (which is the input order now)
    for index in group.index:
        try:
            # Explicitly get the value and ensure it's a string
            tag_feat_value = group.loc[index, 'tag_feature']

            # --- Safeguard ---
            if not isinstance(tag_feat_value, str):
                logging.warning(f"Row index {index}: Unexpected type for tag_feature: {type(tag_feat_value)}. Value: {tag_feat_value}. Skipping.")
                continue # Skip this row if it's not a string

            tag_feat = tag_feat_value # Use the confirmed string value

            # --- Original Logic ---
            if tag_feat not in tag_counts:
                tag_counts[tag_feat] = 0
            if tag_counts[tag_feat] < threshold:
                result_indices.append(index)
                tag_counts[tag_feat] += 1
        except KeyError:
             # Should not happen if column exists, but good practice
             logging.warning(f"Row index {index}: 'tag_feature' column access failed.")
             continue
        except Exception as e:
             logging.error(f"Row index {index}: Unexpected error in apply_user_threshold: {e}", exc_info=True)
             continue # Skip row on other errors

    # Return in original (reverse chrono) order based on collected indices
    if not result_indices:
        # Return an empty DataFrame with the same columns if no rows are kept
        logging.warning("apply_user_threshold resulted in empty DataFrame.")
        return pd.DataFrame(columns=group.columns)
    else:
        return group.loc[result_indices]


def preprocess_user_data(user_handle):
    """Fetches and preprocesses Codeforces data for a given handle."""
    logging.info(f"Starting preprocessing for handle: {user_handle}")
    encoded_handle = quote(user_handle) # URL-encode the handle

    # --- Fetch Submissions ---
    submission_url = f'https://codeforces.com/api/user.status?handle={encoded_handle}&from=1&count=1000' # Limit count
    try:
        logging.info(f"Fetching submissions from {submission_url}")
        submission_request = requests.get(submission_url, timeout=15) # Increased timeout
        submission_request.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        submission_data = submission_request.json()
        if submission_data.get('status') != 'OK':
            logging.error(f"Codeforces API error (submissions) for {user_handle}: {submission_data.get('comment', 'Unknown error')}")
            return None, None # Indicate error
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
    except ValueError as e: # Includes JSONDecodeError
        logging.error(f"JSON Decode Error (Submission) for {user_handle}: {e}")
        return None, None

    time.sleep(0.6) # Be nice to Codeforces API (Increased slightly)

    # --- Expand Problem Data ---
    try:
        # Handle cases where 'problem' might be missing or not structured as expected
        if 'problem' not in user_data.columns:
             logging.error(f"'problem' column missing in submission data for {user_handle}.")
             return None, None

        # Normalize the 'problem' column
        # Using errors='ignore' helps if some rows aren't dicts, but check data quality if it happens often.
        problem_data = pd.json_normalize(user_data['problem'], errors='ignore')

        # Define columns needed FROM the problem dict
        problem_cols_needed = ['contestId', 'index', 'name', 'rating', 'tags']
        # Add any missing columns with None before selecting
        for col in problem_cols_needed:
            if col not in problem_data.columns:
                problem_data[col] = None
        # Select only the needed columns FROM the normalized problem data
        problem_data = problem_data[problem_cols_needed]
        problem_data = problem_data.rename(columns={'rating': 'problemRating'}) # Rename here


        # Define columns needed FROM the original submission data
        # IMPORTANT: Exclude the original 'contestId' from user_data if it exists and is not the one we want.
        #            Only keep columns relevant to the processing sequence.
        original_cols_needed = ['creationTimeSeconds', 'verdict', 'handle'] # Add other needed submission-level fields if necessary
        # Filter the original dataframe to keep only needed columns, dropping 'problem'
        # Make sure these columns actually exist in the original user_data first
        valid_original_cols = [col for col in original_cols_needed if col in user_data.columns]
        original_data_subset = user_data[valid_original_cols].copy() # Use .copy() to avoid SettingWithCopyWarning


        # Combine selected columns from original data and problem data
        # Ensure index alignment before concat is crucial
        original_data_subset.reset_index(drop=True, inplace=True)
        problem_data.reset_index(drop=True, inplace=True)
        combined_data = pd.concat([original_data_subset, problem_data], axis=1)
        # Now combined_data is the new 'user_data' for subsequent steps
        user_data = combined_data # Overwrite user_data variable with the clean, combined data

    except Exception as e:
        logging.error(f"Error expanding/combining problem data for {user_handle}: {e}", exc_info=True)
        return None, None # Cannot proceed without problem info

    # --- Basic Cleaning & Feature Engineering ---
    # Convert types robustly, handling potential errors and NaNs
    # Apply pd.to_numeric to the DataFrame columns directly
    logging.info(f"Columns before numeric conversion attempt: {user_data.columns.tolist()}")
    # Check if columns exist before conversion
    if 'problemRating' in user_data.columns:
        user_data['problemRating'] = pd.to_numeric(user_data['problemRating'], errors='coerce')
    if 'contestId' in user_data.columns:
        user_data['contestId'] = pd.to_numeric(user_data['contestId'], errors='coerce') # <--- This line should now work
    if 'creationTimeSeconds' in user_data.columns:
        user_data['creationTimeSeconds'] = pd.to_numeric(user_data['creationTimeSeconds'], errors='coerce')

    # Define essential columns required AFTER combining
    essential_cols = ['contestId', 'index', 'name', 'problemRating', 'tags', 'creationTimeSeconds', 'verdict', 'handle']
    # Drop rows missing essential problem info AFTER type conversion attempts
    user_data.dropna(subset=[col for col in essential_cols if col in user_data.columns], inplace=True)
    if user_data.empty:
        logging.warning(f"No valid rows left after dropping NaNs in essential columns for {user_handle}.")
        return None, None

    # Convert to integers *after* dropping NaNs introduced by 'coerce'
    user_data['problemRating'] = user_data['problemRating'].astype(int)
    user_data['contestId'] = user_data['contestId'].astype(int)
    user_data['creationTimeSeconds'] = user_data['creationTimeSeconds'].astype(int)
    # Continue with other type conversions
    user_data['index'] = user_data['index'].astype(str)
    user_data['name'] = user_data['name'].astype(str)


    # Combine contestId and index for a unique problem identifier
    user_data['ProblemID'] = user_data['contestId'].astype(str) + user_data['index']
    solved_problem_ids = set(user_data['ProblemID'].unique()) # Get solved IDs before dropping duplicates

    # Sort by time DESCENDING (needed for correct sequence generation later)
    user_data = user_data.sort_values(by='creationTimeSeconds', ascending=False)

    # Drop duplicates based on ProblemID, keeping the LATEST submission info (since we sorted descending)
    user_data = user_data.drop_duplicates(subset=['ProblemID'], keep='first')

    # Process Verdict (on the unique problem attempts)
    user_data['verdict'] = user_data['verdict'].apply(convert_verdict)

    # Process Tags: Clean the list and join into space-separated string
    # Apply the cleaning function that matches the notebook's intent
    user_data['clean_tags_str'] = user_data['tags'].apply(clean_tag_string)

    # Explode into one tag per row
    user_data['tags_list'] = user_data['clean_tags_str'].str.split()
    exploded_data = user_data.explode('tags_list')
    exploded_data = exploded_data.rename(columns={'tags_list': 'tag'})
    exploded_data = exploded_data.dropna(subset=['tag']) # Drop rows where tag became NaN/empty
    exploded_data = exploded_data[exploded_data['tag'] != ''] # Ensure tag is not empty string

    if exploded_data.empty:
        logging.warning(f"No valid tags found after cleaning and exploding for {user_handle}.")
        return None, None

    # Create the tag+rating feature used for training/tokenization
    # Ensure tag and problemRating are strings before concatenating
    exploded_data['tag_feature'] = exploded_data['tag'].astype(str) + exploded_data['problemRating'].astype(str)

    # --- Apply Threshold (like notebook's user_threshold - cell 98) ---
    # Keep max 2 occurrences of each unique tag_feature per user
    # Apply threshold function (data is already sorted reverse chronologically)
    limited_data = apply_user_threshold(exploded_data, threshold=2) # Use threshold=2 like notebook cell 98

    if limited_data.empty:
        logging.warning(f"No data left after applying threshold for {user_handle}.")
        return None, None

    # --- Create Final Sequence ---
    # The data is already sorted reverse chronologically
    final_sequence = " ".join(limited_data['tag_feature'].tolist())

    if not final_sequence:
        logging.warning(f"Final tag sequence is empty for {user_handle}.")
        return None, None

    logging.info(f"Generated sequence for {user_handle} with length: {len(final_sequence.split())}")
    # logging.info(f"Sequence snippet: {final_sequence[:100]}...") # Optional debug print

    return final_sequence, solved_problem_ids


def get_recommendations_from_codeforces(predicted_tags, solved_ids, num_recommendations=10):
    """Fetches problems from Codeforces matching predicted tags, excluding solved ones."""
    recommendations = []
    logging.info(f"Attempting to find problems for {len(predicted_tags)} predicted tags, excluding {len(solved_ids)} solved problems.")

    # Fetch the entire problemset once
    problemset_url = "https://codeforces.com/api/problemset.problems"
    try:
        logging.info("Fetching full problemset...")
        response = requests.get(problemset_url, timeout=20) # Increased timeout
        response.raise_for_status()
        problemset_data = response.json()
        if problemset_data.get('status') != 'OK':
            logging.error(f"Failed to fetch problemset: {problemset_data.get('comment')}")
            return [] # Return empty if problemset fetch fails

        all_problems = problemset_data['result']['problems']
        all_problem_stats = problemset_data['result']['problemStatistics']
        logging.info(f"Fetched {len(all_problems)} problems and {len(all_problem_stats)} stats.")
        # Create a map for quick stat lookup {problemId: solvedCount}
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

    found_problems_ids = set() # Keep track of added problem IDs (contestId + index)

    # Iterate through predicted tags (which are tag+rating features)
    for tag_feature in predicted_tags:
        if len(recommendations) >= num_recommendations:
            break

        # Robustly extract tag and rating from the predicted feature (e.g., "greedy1400")
        try:
            # Find the start of the rating digits
            rating_str = ""
            tag_part = ""
            # Iterate from end to find digits for rating
            for char in reversed(tag_feature):
                 if char.isdigit():
                     rating_str = char + rating_str
                 else:
                     # The rest is the tag, up to the first digit found from the end
                     tag_part = tag_feature[:-len(rating_str)]
                     break # Stop once first non-digit is found from end
            # If loop finishes without break (all digits), tag_part needs assignment
            if not tag_part and rating_str == tag_feature:
                tag_part = "" # Handle cases like "1500" (rating only, no tag)

            if not rating_str: continue # Skip if no rating digits found
            rating = int(rating_str)
            tag = tag_part

            # Allow empty tag if only rating was predicted (though unlikely from training data)
            # if not tag: continue # Skip if tag part is empty - CHANGED: allow empty tag

        except ValueError:
            logging.warning(f"Could not parse tag/rating from predicted feature: {tag_feature}")
            continue # Skip this predicted tag

        #logging.info(f"Searching for problems with tag='{tag}' and rating={rating}")

        # Filter the fetched problemset
        candidate_problems = []
        for problem in all_problems:
            problem_rating = problem.get('rating')
            problem_tags = problem.get('tags', [])
            contest_id = problem.get('contestId')
            problem_index = problem.get('index')

            # Check conditions carefully
            # Match rating, tag must be in problem tags (or predicted tag is empty), and IDs must exist
            if (problem_rating == rating and
                (tag in problem_tags or tag == "") and # Match tag OR allow empty predicted tag
                contest_id is not None and
                problem_index is not None):

                problem_id = f"{contest_id}{problem_index}"

                # Check if solved or already recommended
                if problem_id not in solved_ids and problem_id not in found_problems_ids:
                    problem_info = {
                        'contestId': contest_id,
                        'index': problem_index,
                        'name': problem.get('name', 'N/A'),
                        'rating': problem_rating,
                        'tags': problem_tags,
                        'url': f"https://codeforces.com/problemset/problem/{contest_id}/{problem_index}",
                        'solvedCount': stats_map.get(problem_id, 0) # Get solved count
                    }
                    candidate_problems.append(problem_info)

        # Sort candidates by solved count (descending) to recommend popular ones first
        candidate_problems.sort(key=lambda x: x.get('solvedCount', 0), reverse=True)

        # Add candidates to recommendations until limit is reached
        for problem in candidate_problems:
             if len(recommendations) < num_recommendations:
                  recommendations.append(problem)
                  # Add the problem ID (contestId + index) to prevent duplicates
                  found_problems_ids.add(f"{problem['contestId']}{problem['index']}")
             else:
                  break # Stop adding if we reached the desired number


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
    user_sequence, solved_problem_ids = preprocess_user_data(handle)
    if user_sequence is None or solved_problem_ids is None:
        logging.warning(f"Preprocessing failed for handle '{handle}'.")
        # Check if solved_problem_ids is None, could mean user not found or has no history
        status_code = 404 if solved_problem_ids is None else 500
        return jsonify({"error": f"Could not process data for handle '{handle}'. User might not exist, have insufficient history/tags, or an API error occurred."}), status_code

    # 2. Predict next tags iteratively
    predicted_tags = []
    current_sequence = user_sequence # Start with the user's history
    num_preds_to_generate = 20 # Generate more tags than needed to feed into problem search

    try:
        logging.info(f"Starting prediction loop for {handle}...")
        for i in range(num_preds_to_generate):
            # Tokenize current sequence
            token_text = tokenizer.texts_to_sequences([current_sequence])[0]
            # Pad sequence - Use the determined MODEL_INPUT_LENGTH
            padded_token_text = pad_sequences([token_text], maxlen=MODEL_INPUT_LENGTH, padding='pre')

            # Predict probabilities
            pred_probs = model.predict(padded_token_text, verbose=0)[0] # verbose=0 suppresses progress bar

            # Get index of the highest probability (excluding padding token 0)
            pred_probs[0] = 0 # Ensure padding token isn't chosen
            pos = np.argmax(pred_probs)
            pred_confidence = pred_probs[pos] # Get confidence score

            # Decode index to word (tag_feature)
            predicted_tag = tokenizer.index_word.get(pos)

            if predicted_tag:
                #logging.info(f"Prediction {i+1}: {predicted_tag} (Conf: {pred_confidence:.4f})")
                if predicted_tag not in predicted_tags: # Avoid immediate duplicate tag predictions
                     predicted_tags.append(predicted_tag)
                # Append the prediction to the sequence for the next iteration
                current_sequence += " " + predicted_tag
            else:
                 logging.warning(f"Predicted index {pos} not found in tokenizer.index_word. Stopping prediction.")
                 break # Stop if we can't decode

    except Exception as e:
        logging.error(f"Error during prediction loop for {handle}: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during model prediction."}), 500

    if not predicted_tags:
        logging.warning(f"Could not generate any tag predictions for {handle}.")
        return jsonify({"error": "Could not generate any tag predictions based on user history."}), 500

    logging.info(f"Predicted top {len(predicted_tags)} tag features for {handle}: {predicted_tags}")

    # 3. Fetch actual problems based on predicted tags
    recommendations = get_recommendations_from_codeforces(predicted_tags, solved_problem_ids, num_recommendations=10) # Fetch top 10 problems

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    logging.info(f"Total request processing time for {handle}: {processing_time} seconds.")


    # 4. Return recommendations
    response_data = {
        "user_handle": handle,
        "processing_time_seconds": processing_time,
        "predicted_tags_analyzed": predicted_tags, # Return all generated tags
        "recommendations": recommendations
    }
    if not recommendations:
         response_data["message"] = "Could not find suitable unsolved problems matching the predicted tags."


    return jsonify(response_data), 200

# --- Run the App ---
if __name__ == '__main__':
    # Use port from environment variable or default to 5001
    port = int(os.environ.get('PORT', 5001))
    # Run on 0.0.0.0 to make it accessible externally (e.g., in Docker/cloud)
    logging.info(f"Starting Flask server on host 0.0.0.0 port {port}...")
    # Use a production server like Waitress or Gunicorn instead of app.run for real deployment
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=port, debug=False) # Turn debug=False for production
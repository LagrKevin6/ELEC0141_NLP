import pandas as pd
import ast
import torch

import faiss
import matplotlib.pyplot as plt

# Loading policy from Excel
excel_path = 'Dataset/Policy.xlsx'

policy_df = pd.read_excel(excel_path,sheet_name= 0, engine='openpyxl')
prompt_df = pd.read_excel(excel_path,sheet_name= 1, engine='openpyxl')
label_df = pd.read_excel(excel_path,sheet_name= 2, engine='openpyxl')

policy_scripts = policy_df['Policy Scripts'].tolist()

# ======================================================================================================================
def parse_labels(label_str):
    try:
        # Safely evaluate the string as a Python literal (list in this case)
        return ast.literal_eval(label_str)
    except ValueError:
        # In case of an error (e.g., malformed string), return an empty list or handle accordingly
        return []

# Apply the conversion to each row in the 'CorrectLabels' column
label_df['True'] = label_df['label list'].apply(parse_labels)

# ======================================================================================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

# Gemma 2B it
Token = "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"
global tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token=Token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",token=Token,
    trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



# Embedding vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(policy_scripts).toarray()

# option for refitting with new words
refit_flag = True

if(refit_flag == True):
    # Extract features from new docs
    new_features = TfidfVectorizer().fit(prompt_df['Prompt']).get_feature_names_out()

    # Combine the features
    combined_features = vectorizer.get_feature_names_out().tolist() + list(set(new_features) - set(vectorizer.get_feature_names_out()))

    # Create a new vectorizer with the updated vocabulary
    vectorizer = TfidfVectorizer(vocabulary=combined_features)
    vectorizer.fit((policy_scripts) + (prompt_df['Prompt']).tolist())  # Fit the updated vectorizer to new docs

    X = vectorizer.fit_transform(policy_scripts).toarray()

# Convert to float32 for FAISS compatibility
X = np.array(X, dtype='float32')



# initialise with the dimension of the vectors
d = X.shape[1]

# Add index
index = faiss.IndexFlatL2(d)
index.add(X)

def evaluate_retrieval(search_result, true_labels):
    """
    Check if all true labels are contained within the predicted labels.
    Args:
    predicted_labels (list): The labels retrieved by the search model.
    true_labels (list): The correct labels listed in the Excel file.

    Returns:
    bool: True if all true labels are in the predicted labels, False otherwise.
    """
    #force the input to be sets;
    y_pred = set(search_result)
    y_true = set(true_labels)

    # check coverage
    correct = y_true.intersection(y_pred)
    
    # Calculate coverage
    coverage = len(correct) / len(y_true)

    # note missed labels
    misses = list(y_true - y_pred)

    return y_true.issubset(y_pred), coverage, misses

global default_files
default_files = False

def k_top_search(upper_thres, vectorizer, index, prompt_df, label_df, lower_thres = 5):
    """
    loop through all k in a range, from lower thres (5 by default) to upper thres,
    giving a figure showing the accuracy, coverage and average token number over different k
    Args:
    upper_thres (int): Number of top searches upper limit
    lower_thres (int): Number of top searches lower limit
    vectorizer: Tfidvectorizer, fitted
    index: faiss object after index addition
    prompt_df: pd df, must contain 'Prompt' column for queries
    label_df: pd df, must contain 'True' column for true labels
    Returns:
    null
    """
    # lists for plots
    accuracies = []
    coverages = []
    num_tokens = []

    for k in range(lower_thres, upper_thres+1):
        indices_list = []
        missed_list = []
        token_count = 0

        for query in prompt_df['Prompt']:

            query_vector = vectorizer.transform([query]).toarray()
            query_vector = np.array(query_vector, dtype='float32')
            _, indices = index.search(query_vector, k)

            # implement default file augment
            temp_q = indices.flatten().tolist()
            if default_files == True:
                for element in [0,44]:
                    if element not in temp_q:
                        temp_q.append(element)

            indices_list.append(temp_q)

            # combining the full query with full searched docs
            combined_query = query
            for temp in indices.flatten().tolist():
                combined_query = combined_query + ' ' + policy_scripts[temp]

            # tokenize
            tokens = tokenizer.tokenize(combined_query)
            
            # Return the number of tokens
            token_count += len(tokens)

        # Adding search results for further check
        prompt_df['TopIndices'] = indices_list


        #evaluate
        accu_count = 0
        accu_cover_count = 0
        for i in range(len(prompt_df)):
            temp, cover, missed = evaluate_retrieval(prompt_df['TopIndices'][i] , label_df['True'][i])

            accu_count += int(temp)
            accu_cover_count += cover

            missed_list.append(missed)

        accuracy_1 = accu_count/len(prompt_df)
        coverage_1 = accu_cover_count/len(prompt_df)
        print(f"For top {k} searches:\nAccuracy of search results containing all correct labels: {accuracy_1 * 100},\n Average coverage of correct labels: {coverage_1 * 100}")

        token_1 = token_count/len(prompt_df)
        print(f"Average tokens combining the query and retrieved docs: {token_1}")

        # Add to the lists
        accuracies.append(accuracy_1)
        coverages.append(coverage_1)
        num_tokens.append(token_1)
        
    return accuracies, coverages, num_tokens, missed_list

upper_search = 20
a,b,c, last_missed_list = k_top_search(upper_thres= upper_search,vectorizer = vectorizer, index = index, prompt_df = prompt_df, label_df = label_df)

missed_df = pd.DataFrame({'Missed_Labels':last_missed_list})

# Flatten the lists of missed labels into a single list
missed_labels = [label for sublist in missed_df['Missed_Labels'] for label in sublist]

# Calculate frequency counts of missed labels
missed_labels_counts = pd.Series(missed_labels).value_counts().to_dict()

# Print the frequency counts
print("Missed labels and their frequencies:")
for label, count in missed_labels_counts.items():
    print(f"{label}: {count}")



# restart the loop, with default file selection built in
default_files = True
a_new,b_new,c_new, last_missed_list = k_top_search(upper_thres= upper_search,vectorizer = vectorizer, index = index, prompt_df = prompt_df, label_df = label_df)

# Plot acc and cover.
plt.figure(figsize=(8, 6))
plt.plot(range(5, upper_search+1), a, label='Accuracy (Fully Match)')
plt.plot(range(5, upper_search+1), a_new, label='Accuracy with Default Selection(Fully Match)')
plt.plot(range(5, upper_search+1), b, label='Average Coverage')
plt.plot(range(5, upper_search+1), b_new, label='Average Coverage with Default Selection')
plt.xlabel('Num of Searched Results')
plt.ylabel('Percentage')
plt.title('Fully Match Accuracy and Average Coverage vs. Num of Searched Results Included')
plt.legend()
plt.savefig('images/acc_1.png')  # Save the plot
plt.close()  # Close the figure to release memory 

# Plot number of tokens
plt.figure(figsize=(8, 6))
plt.plot(range(5, upper_search+1), c,marker = 'o', label='Num of Tokens')
plt.plot(range(5, upper_search+1),  c_new, label='Num of Token with Default Selection')
plt.xlabel('Num of Searched Results')
plt.ylabel('Total Number of Tokens')
plt.title('Number of Tokens vs. k')
plt.legend()
plt.savefig('images/num_tokens_1.png')  # Save the plot
plt.close()  # Close the figure to release memory

# ======================================================================================================================
# Generation
query_text = 'Does GNEI provide travel insurance? Provided with receipts and pre-approval for the travel insurance'
k = 10

query_vector = vectorizer.transform([query_text]).toarray()
query_vector = np.array(query_vector, dtype='float32')
_, indices = index.search(query_vector, k)

# implement default file augment
temp_q = indices.flatten().tolist()
if default_files == True:
    for element in [0,44]:
        if element not in temp_q:
            temp_q.append(element)

indices = temp_q

# combining the full query with full searched docs
combined_query = query_text
for temp in indices:
    combined_query = combined_query + ' ' + policy_scripts[temp]

fully_load_query = query_text
for i in range(len(policy_df)):
    fully_load_query = fully_load_query + ' ' + policy_scripts[i]

# tokenize
Poor_tokens = tokenizer.tokenize(query_text)
RAG_tokens = tokenizer.tokenize(combined_query)
Fully_load_tokens = tokenizer.tokenize(fully_load_query)

print(len(Poor_tokens),len(RAG_tokens),len(Fully_load_tokens))

input_ids = tokenizer(query_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens = 400, min_new_tokens = 5, repetition_penalty = 1.2)

# trim the output by removing prompt
model_response = tokenizer.decode(outputs[0])

trimmed_output = model_response[len(query_text)+5:]
print(trimmed_output)

input_ids = tokenizer(combined_query, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens = 400, min_new_tokens = 5, repetition_penalty = 1.2)

# trim the output by removing prompt
model_response = tokenizer.decode(outputs[0])

trimmed_output = model_response[len(combined_query)+5:]
print(trimmed_output)

input_ids = tokenizer(combined_query, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids,max_new_tokens = 400, min_new_tokens = 5, repetition_penalty = 1.2)

# trim the output by removing prompt
model_response = tokenizer.decode(outputs[0])

trimmed_output = model_response[len(combined_query)+5:]
print(trimmed_output)

# ======================================================================================================================

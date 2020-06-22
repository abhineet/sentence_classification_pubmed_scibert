import pandas as pd
from keras.preprocessing.sequence import pad_sequences


def _extract_df(abs_key, abstract):
    LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}
    # NUM_LABELS = len(LABELS)
    # COLUMNS = {'abstract', 'sentence', 'label'}
    rows = []
    for sentence in abstract:
        try:
            label, sentence = sentence.split('\t')
        except ValueError:
            # line just contains ''
            continue
        label = LABELS[label.lower()]
        row = {'label': label, 'sentence': sentence, 'abstract': abs_key}
        rows.append(row)
    return pd.DataFrame(rows)


def preprocess_data(path):
    #path = '/content/drive/My Drive/SciBert/train.txt'
    with open(path, 'r') as infile:
        txt = infile.read()
        # split per abstract
        txt = txt.split('###')
        # split lines
        txt = [abstract.split('\n') for abstract in txt]
        # collect into dict[abs_key] = list of abs sentences
        txt = {l[0]: l[1:] for l in txt}

    dfs = [_extract_df(abs_key, abstract) for abs_key, abstract in txt.items()]
    df = pd.concat(dfs, axis=0)

    df['number_of_words'] = df.sentence.apply(lambda x: len(x.split()))
    #filter out sentences which are longer than 78 (99 percentile)
    df = df[(df.number_of_words <= 78)]
    return df

def get_encoded_data(tokenizer,sentences):
    # Tokenize  and map the tokens to their word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #https://huggingface.co/transformers/main_classes/tokenizer.html
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)


    #padding the data
    MAX_LEN = 80
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    #attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return input_ids, attention_masks


def get_torch_tensors(data_inputs,data_labels,masks):
    import torch
    tensor_inputs = torch.tensor(data_inputs)
    tensor_labels = torch.tensor(data_labels)
    tensor_masks = torch.tensor(masks)
    return tensor_inputs,tensor_labels,tensor_masks

def break_into_chunks(l,n):
    x = [l[i:i + n] for i in range(0, len(l), n)]
    return x

def flatten_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


import numpy as np
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_classification_report(prediction_dataloader,model,LABELS):
    # Tracking variables
    predictions, true_labels = [], []
    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')

    from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, \
        confusion_matrix, f1_score
    int2label = {}
    for k, v in LABELS.items():
        int2label[v] = k
    true_labels_flat = []
    for batch in true_labels:
        for i in batch:
            temp = int2label[i]
            true_labels_flat.append(temp)

    pred_labels = []
    for batch in predictions:
        for i in batch:
            key = i.argmax(-1)
            temp = int2label[key]
            pred_labels.append(temp)

    print(classification_report(true_labels_flat, pred_labels, digits=4))


import torch
# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
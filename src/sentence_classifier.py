import yaml
import sys
import torch
import getopt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel ,BertForSequenceClassification
from Utils.utils import *
from Utils.finetune import *


# Constants
LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}
NUM_LABELS = len(LABELS)
COLUMNS = {'abstract', 'sentence', 'label'}


def main(method, cfg):
    if method == 'train':
        # get all the training data for preparaing the label set and store the labels for future use
        df = preprocess_data(cfg['data']['train'])

        sentences = df.sentence.values
        labels = df.label.values

        tokenizer = BertTokenizer.from_pretrained(cfg['data']['scibert_model'], do_lower_case=True)

        training_inputs, training_masks = get_encoded_data(tokenizer, sentences)

        # convert to torch tensors
        tensor_inputs, tensor_labels, tensor_masks = get_torch_tensors(training_inputs, labels, training_masks)
        batch_size = cfg['hyperparams']['batch_size']

        # Create the DataLoader for our training set.
        train_data = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model, scheduler, optimizer = initialise_model(cfg, train_dataloader)
        epochs = cfg['hyperparams']['epochs']
        model = finetunemodel(model, scheduler, epochs, train_dataloader, optimizer)
        save_model(model, cfg, tokenizer)
        sys.exit(1)

    elif method == 'eval':

        # load the labels from training data:label is key
        df = preprocess_data(cfg['data']['test'])
        sentences = df.sentence.values
        labels = df.label.values

        tokenizer = BertTokenizer.from_pretrained(cfg['data']['scibert_model'], do_lower_case=True)
        test_inputs, test_masks = get_encoded_data(tokenizer, sentences)

        # convert to torch tensors
        tensor_inputs, tensor_labels, tensor_masks = get_torch_tensors(test_inputs, labels, test_masks)
        batch_size = cfg['hyperparams']['batch_size']

        # Create the DataLoader for our testing set.
        test_data = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)
        test_sampler = SequentialSampler(test_data)
        prediction_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


        #load saved  model for eval
        model = BertForSequenceClassification.from_pretrained(cfg['data']['scibert_finetuned_model'])
        model.eval()

        print_classification_report(prediction_dataloader,model,LABELS)

    else:
        print("Please check the argument . Expected value train or eval")
        sys.exit(1)


if __name__ == "__main__":
    try:
        options, args = getopt.getopt(sys.argv[1:], "mh", ["mode="])
        for name, value in options:
            if name in ('-m', '--mode'):
                mode = value
                assert mode == "train" or mode == "eval"
            if name in ('-h', '--help'):
                print ('python sentence_classifier.py --mode eval\\train ')
                sys.exit(1)
    except getopt.GetoptError as err:
        print("Seems arguments are wrong..")
        print("usage:: python sentence_classifier.py --mode eval\\train")
        print ("Ex:: python sentence_classifier.py --mode eval")
        sys.exit(1)

    with open('../config/config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(mode,config)

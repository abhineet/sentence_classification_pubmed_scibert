import numpy as np
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random

from Utils.utils import  *



def initialise_model(cfg,train_dataloader):
    # Load BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained(
        cfg['data']['scibert_model'],
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False,
    )
    # run on the GPU.
    # model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr= float(cfg['hyperparams']['learning_rate']),
                      eps=float(cfg['hyperparams']['epsilon'])
                      )
    epochs = cfg['hyperparams']['epochs']
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    return model,scheduler,optimizer


def finetunemodel(model,scheduler,epochs,train_dataloader,optimizer):
    if torch.cuda.is_available():
        # use GPU
        device = torch.device("cuda")
        print('GPU INF:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('GPU not available, using the CPU.')
        device = torch.device("cpu")
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training started ...')
        # Measure epoch time interval.
        start_time = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            #  update every 40 batches.
            if step % 40 == 0 and not step == 0:
                #  elapsed time in minutes.
                elapsed = format_time(time.time() - start_time)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Clear any previously calculated gradients before performing a
            # backward pass.
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # average loss in training.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - start_time)))

    return model


def save_model(model,cfg,tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    OUTPUT_DIR = cfg['finetunedmodel']['vocab']
    output_model_file = cfg['finetunedmodel']['model']  # os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
    output_config_file =  cfg['finetunedmodel']['config']  # os.path.join(OUTPUT_DIR, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(OUTPUT_DIR)
    print("DONE!!!")

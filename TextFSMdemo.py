import subprocess
from transformers import T5ForConditionalGeneration, T5Tokenizer

import os

model_path = 't5-small'  # the path where you saved your model
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('t5-small')

flag_status = 0

def format_textFSM(str):
    str = str.replace(')', ')\n')
    str = str.replace('start', 'start\n')
    str = str.replace('Start', 'Start\n')
    str = str.replace('record', 'record\n')
    str = str.replace('Record', 'Record\n')
    str = str.replace('^', '\n^')
    return str

def predict_modelxx(data):
    f = open(data, 'r')
    cli_data = f.read()

    print(cli_data)
    print('-' * 100)
    cli = "Create only a TextFSM template.   \n"
    cli += cli_data
    cli = cli.replace('\n', '        ')

    result_str = do_correction(cli_data, model, tokenizer)

    result_fsm = "Something went wrong. Complete only correct TextFSM template about the next text and nothing else.   "+result_str
    cli = result_fsm.replace('\n', '        ')

    data_ok = "pieces ask \"" + cli + "\""
    result = subprocess.run(data_ok, shell=True, capture_output=True, text=True)
    result_fsm = result.stdout
    return result_fsm
# CLI format
def predict_model(data):
    with open(data, 'r') as cli_file:
        cli_data = cli_file.read()

    status = do_correction(cli_data, model, tokenizer)
    print(cli_data)
    print('-' * 100)
    cli_data = "How many line break is there in the sentence?  Calculate you! 'What is your name?\n My name is John.\n'"
    cli =""# "Create only a TextFSM template to parse the next cli's output.  the cli's output is the following:    \n"
    cli += cli_data
    # cli = cli.replace('\n', '        ')
    result_fsm = "Something went wrong."
    if status:
        data_ok = "pieces ask \"" + cli + "\""
        result = subprocess.run(data_ok, shell=True, capture_output=True, text=True)
        result_fsm = result.stdout
    return result_fsm

#json format
def predict_model_j(data):
    f = open(data, 'r')
    cli_data = f.read()
    status = do_correction(cli_data, model, tokenizer)
    print(cli_data)
    print('-' * 100)
    cli = "Create only a TextFSM template.   \n"
    cli += cli_data
    cli = cli.replace('{', ' ')
    cli = cli.replace('}', ' ')
    cli = cli.replace('[', ' ')
    cli = cli.replace(']', ' ')
    cli = cli.replace('\"', ' ')
    cli = cli.replace(',', ' ')
    cli = cli.replace('\n', '        ')
    result_fsm = "Something went wrong."
    if status:
        data_ok = "pieces ask \"" + cli + "\""
        result = subprocess.run(data_ok, shell=True, capture_output=True, text=True)
        result_fsm = result.stdout
    return result_fsm

def do_correction(text, model, tokenizer):
    flag_status = 0
    input_text = f"Generate TextFSM: {text}"
    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )

    # Get correct sentence ids.
    corrected_ids = model.generate(
        inputs,
        max_length=512,
        num_beams=5,  # `num_beams=1` indicated temperature sampling.
        early_stopping=True
    )

    # Decode.
    corrected_sentence = tokenizer.decode(
        corrected_ids[0],
        skip_special_tokens=True
    )
    # if corrected_sentence!="":
    #     flag_status = 1
    # return flag_status
    return corrected_sentence

result = predict_model("show_version.raw")
# result = predict_model_j("show_arpnd_arp-entries_pipe_as_json.raw")
print("TextFSM templeate")
print(result)
# print(result.stderr)
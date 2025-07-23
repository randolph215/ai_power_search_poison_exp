import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

def main():
    base_model = "yahma/llama-7b-hf"
    lora_weights = "tloen/alpaca-lora-7b"
    prompter = Prompter("")
    tokenizer = LlamaTokenizer.from_pretrained(base_model,lora_weights, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

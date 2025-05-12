from transformers import AutoTokenizer
import os
os.environ["TRANSFORMERS_CACHE"] = "/playpen-nas-ssd4/awang/SeViLA/cache/"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl", force_download=True)





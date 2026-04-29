"""
Fine-tuning script: BSC-LT/salamandra-2b-instruct (Causal LM) using SFT + LoRA with 4-bit QLoRA and CodeCarbon (offline).

Input dataset:
- JSONL file with at least the fields: "source", "target".
  Each line: {"source": "...", "target": "..."}

Training setup:
- Chat-style formatting using role-based messages:
    [{"role": "user", "content": source},
     {"role": "assistant", "content": target}]
- Conversion to model-specific prompt format via tokenizer.apply_chat_template().
- 4-bit quantization via BitsAndBytes (nf4) for memory-efficient training (QLoRA).
- LoRA adapters configured for task_type="CAUSAL_LM".

Outputs:
- LoRA-adapted model saved under `output_dir`.
- Tokenizer saved under `output_dir/tokenizer`.
- CodeCarbon emissions CSV: `emissions_<project_name>.csv` (converted to EU CSV format).
"""

import os
import csv
import torch
import pandas as pd, os
from trl import SFTTrainer
from peft import LoraConfig
from datasets import Dataset
from codecarbon import OfflineEmissionsTracker
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

# ---- METADATA Y CODECARBON ----
nombre_script = os.path.basename(__file__)        
nombre_base = os.path.splitext(nombre_script)[0]    
project_name = nombre_base.replace("_train", "") 

tracker = OfflineEmissionsTracker(
    project_name=project_name, 
    output_file=f"emissions_{project_name}.csv", 
    country_iso_code="ESP"
)
tracker.start()

# ---- CARGA DEL DATASET ----
df = pd.read_json("../../dataset_source_target.jsonl", lines=True)

# ---- FORMATEO TIPO INSTRUCT PARA (Salamandra) ---- 
df["message"] = df.apply(
    lambda row: [
        {"role": "user", "content": row["source"]},
        {"role": "assistant", "content": row["target"]}
    ],
    axis=1
)

# ---- CONVERSION A HUGGINGFACE DATASET ----
dataset = Dataset.from_dict({"message": df["message"].tolist()})

# ---- CARGA DEL MODELO + CUANTIZACIÓN 4-bit (QLoRA) ----
model_name = "BSC-LT/salamandra-2b-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# ---- TOKENIZER + PADDING ----
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for msgs in examples["message"]:
        txt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False 
        )
        texts.append(txt)
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names,  
)

# ---- HIPERPARAMETROS Y ADAPTADOR LORA (CAUSAL_LM) ----
lora_alpha = 32 
lora_dropout = 0.10 
lora_r = 8 

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# ---- CONFIGURACION DE ARGUMENTOS PARA EL ENTRENAMIENTO ----
batch_size = 1 
gradient_accumulation_steps = 8 
optim = "paged_adamw_32bit"
learning_rate = 2e-5
max_grad_norm = 1
warmup_ratio = 0.1
lr_scheduler_type = "constant" 

# ---- TRAINING ARGUMENTS ----
training_args = TrainingArguments(
    output_dir="../../finetuning/FineTuningSalamandra-2B-Instruct",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,    
    learning_rate=learning_rate,    
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    load_best_model_at_end=True,
    save_strategy='no',
    save_total_limit=2,
    logging_steps=100,
    report_to="none"
)

# ---- SFT TRAINER CONFIG ----
max_seq_length=1024
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=False
)

# ---- ENTRENAMIENTO Y GUARDADO DEL MODELO ----
trainer.train()
trainer.save_model()
tokenizer.save_pretrained("../../finetuning/FineTuningSalamandra-2B-Instruct/tokenizer")

# ---- POSTPROCESO CODECARBON CSV A (FORMATO EUROPEO)
emissions_data = tracker.stop()

csv_file = f"emissions_{project_name}.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    df.to_csv(csv_file, sep=';', decimal=',', index=False, encoding='utf-8')
    print(f"✅ Archivo convertido al formato europeo: {csv_file}")
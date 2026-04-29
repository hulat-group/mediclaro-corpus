"""
Fine-tuning script: HiTZ/Medical-mT5-xl (Seq2Seq) with LoRA + 4-bit QLoRA and CodeCarbon (offline).

Input dataset:
- JSONL file with at least the fields: "source", "target".
  Each line: {"source": "...", "target": "..."}

Training setup:
- Encoder–decoder architecture (Seq2SeqTrainer).
- 4-bit quantization via BitsAndBytes (nf4).
- LoRA adapters applied to SEQ_2_SEQ_LM task type.

Outputs:
- LoRA-adapted model saved under `output_dir`.
- Tokenizer saved under `output_dir/tokenizer`.
- CodeCarbon emissions CSV: `emissions_<project_name>.csv` (converted to EU CSV format).
"""

import os
import csv
import torch
import numpy as np
import pandas as pd, os
from peft import LoraConfig
from datasets import Dataset
from codecarbon import OfflineEmissionsTracker
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

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

# ---- FORMATEO Y CONCATENAMIENTO SOURCE Y TARGET EN UNA SOLA COLUMNA ----
df['text'] = df.apply(lambda row: 'TEXTO A SIMPLIFICAR: '+ row['source'] + "\nRESPUESTA: " + row['target'], axis = 1)
df['source'] = "TEXTO A SIMPLIFICAR: " + df['source']
df['target'] = "RESPUESTA: " + df['target']
df

# ---- CREAR Y CONVERSION A HUGGINGFACE DATASET ----
snips_dataset = Dataset.from_dict(
    dict(
        source=df['source'].tolist(),
        target=df['target'].tolist()
    )
)
snips_dataset

# ---- CARGA DEL MODELO + CUANTIZACIÓN 4-bit (QLoRA) ----
model_name="HiTZ/Medical-mT5-xl"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    device_map="auto"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False 
model.get_memory_footprint()

# ---- TOKENIZER + PADDING ----
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ---- HIPERPARAMETROS Y ADAPTADOR LORA (Seq2Seq) ----
lora_alpha = 32 
lora_dropout = 0.10 
lora_r = 8 

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model.add_adapter(peft_config, adapter_name="adapter_4")
model.set_adapter("adapter_4")

# ---- PREPROCESAMIENTO (ENTRADA/SALIDA) ----
def get_feature(batch):
  encodings = tokenizer(batch['source'], text_target=batch['target'],
                        max_length=480, truncation=True)

  encodings = {'input_ids': encodings['input_ids'],
               'attention_mask': encodings['attention_mask'],
               'labels': encodings['labels']}

  return encodings

# ---- TOKENIZACION AL DATASET ----
snips_dataset_pt = snips_dataset.map(get_feature, batched=True)
columns = ['input_ids', 'labels', 'attention_mask']
snips_dataset_pt.set_format(type='torch', columns=columns)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ---- TRAINING ARGUMENTS ----
max_grad_norm = 0.3
warmup_ratio = 0.03
lr_scheduler_type = "constant" 
batch_size = 1 
args = Seq2SeqTrainingArguments(
    output_dir = "../../finetuning/FineTuningMT5_XL",
    learning_rate=5e-5,
    optim="paged_adamw_32bit",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,    
    weight_decay=0.02,
    fp16=True,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    load_best_model_at_end=True,
    save_strategy='no',
    save_total_limit=2
)

# ---- INICIAR ENTRENAMIENTO ----
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=snips_dataset_pt,
    tokenizer=tokenizer,
)

# ---- ENTRENAMIENTO Y GUARDADO DEL MODELO ----
trainer.train()
trainer.save_model()
tokenizer.save_pretrained("../../finetuning/FineTuningMT5_XL/tokenizer")

# ---- POSTPROCESO CODECARBON CSV A (FORMATO EUROPEO)
emissions_data = tracker.stop()

csv_file = f"emissions_{project_name}.csv"
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    df.to_csv(csv_file, sep=';', decimal=',', index=False, encoding='utf-8')
    print(f"✅ Archivo convertido al formato europeo: {csv_file}")
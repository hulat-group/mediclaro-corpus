"""
Inference script: Simplification with fine-tuned Llama-3.2-1B-Instruct (LoRA adapter) + CodeCarbon (offline).

This script performs text simplification using a previously fine-tuned model:
- Base model: meta-llama/Llama-3.2-1B-Instruct
- LoRA adapter loaded from a local fine-tuning directory via PEFT.

Input:
- Folder containing .txt files (clinical cases).
- Files are automatically sorted by case number extracted from the filename
  (pattern: "CasoClinico<NUMBER>").

Outputs:
- Simplified .txt files saved in the specified output directory.
- CodeCarbon emissions CSV: `emissions_<project_name>.csv`.

"""

import os
import re
import torch
import pandas as pd
from peft import PeftModel
from codecarbon import OfflineEmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================
# CODECARBON (OFFLINE) - CONFIG
# =============================
nombre_script = os.path.basename(__file__)
nombre_base = os.path.splitext(nombre_script)[0]
project_name = nombre_base.replace("_1", "")

tracker = OfflineEmissionsTracker(
    project_name=project_name,
    output_file=f"emissions_{project_name}.csv",
    country_iso_code="ESP"
)

# =======================
# CONFIGURACIÓN DE RUTAS
# =======================
input_folder = "../../originales_txt_test"         
output_folder = "../../Conjunto1_finetunning/simplificaciones/llama3_2_1B_instruct"      
os.makedirs(output_folder, exist_ok=True)

# ============================================
# ORDENAR ARCHIVOS POR NÚMERO DE CASO CLÍNICO 
# ============================================
def get_case_number(filename):
    """Extrae el número inmediatamente después de 'CasoClinico' (por ejemplo, 36 de 'CasoClinico36-2022-49.txt')."""
    match = re.search(r'CasoClinico(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

files = sorted(
    [f for f in os.listdir(input_folder) if f.endswith(".txt")],
    key=get_case_number
)

# =======================
# EJECUCIÓN CON TRACKING
# =======================
tracker.start()

try:
    # ==========================
    # CARGA DEL MODELO Y TOKENIZER
    # ==========================
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_dir = "../../finetuning/FineTuningLlama-3.2-1B-Instruct" 

    print("🔹 Cargando modelo base y adaptador LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Modelo y tokenizer cargados correctamente.\n")

    # ==========================
    # PROCESAMIENTO DE ARCHIVOS
    # ==========================
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Lee el texto original
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Formato tipo chat
        messages = [{"role": "user", "content": "Simplifica el siguiente texto:\n" + text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenizar e inferir
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.25,
                top_p=0.85, 
                top_k=10, 
                num_beams=3, 
                no_repeat_ngram_size=4
            )

        # Decodifica y limpia el texto generado
        simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified_text = simplified_text.split("assistant", 1)[-1].strip()

        # Se guarda el texto simplificado
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(simplified_text)

        print(f"✅ Simplificado: {filename}")

    print("\n🎯 Proceso completado. Los archivos simplificados están en:")
    print(f"   {os.path.abspath(output_folder)}")

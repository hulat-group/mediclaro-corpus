"""
Inference script: Text simplification using the base model BSC-LT/salamandra-2b-instruct (Causal LM) with prompt engineering and CodeCarbon tracking (offline).

Input:
- Folder containing original .txt files (clinical cases).
- Files are automatically sorted by case number extracted from filenames
  (pattern: "CasoClinico<NUMBER>").

Outputs:
- Simplified .txt files saved in the specified output directory.
- CodeCarbon emissions CSV: `emissions_<project_name>_text.csv`
  (converted to European CSV format).

Post-processing:
- Removes assistant role markers (e.g., "<|start_header_id|>assistant<|end_header_id|>"
  or "assistant") from generated outputs before saving.

"""

import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from codecarbon import OfflineEmissionsTracker

# =============================
# CODECARBON (OFFLINE) - CONFIG
# =============================
nombre_script = os.path.basename(__file__)
nombre_base = os.path.splitext(nombre_script)[0]
project_name = nombre_base.replace("_c3", "")

tracker = OfflineEmissionsTracker(
    project_name=project_name,
    output_file=f"emissions_{project_name}_text.csv",
    country_iso_code="ESP"
)

# ======================
# CONFIGURACIÓN DE RUTAS
# ======================
input_folder = "../../originales_txt_test"          
output_folder = "../../Conjunto3_prompt/simplificaciones/salamandra_2B_instruct"     
os.makedirs(output_folder, exist_ok=True)

# ===========================================
# ORDENAR ARCHIVOS POR NÚMERO DE CASO CLÍNICO
# ===========================================
def get_case_number(filename):
    """Extrae el número inmediatamente después de 'CasoClinico' (por ejemplo, 36 de 'CasoClinico36-2022-49.txt')."""
    match = re.search(r'CasoClinico(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

files = sorted(
    [f for f in os.listdir(input_folder) if f.endswith(".txt")],
    key=get_case_number
)

# ======================
# EJECUCIÓN CON TRACKING
# ======================
tracker.start()

try:
    # ============================
    # CARGA DEL MODELO Y TOKENIZER
    # ============================
    base_model = "BSC-LT/salamandra-2b-instruct"

    print("🔹 Cargando modelo base y adaptador LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print("✅ Modelo y tokenizer cargados correctamente.\n")

    # =========================
    # PROCESAMIENTO DE ARCHIVOS
    # =========================
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Lee el texto original
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Prompt personalizado para simplificación
        custom_prompt = f"""Simplifica el siguiente texto en español para que sea claro, breve y fácil de entender.
        Reglas:
        1. Usa frases y párrafos cortos.
        2. Mantén solo la información clave y elimina detalles innecesarios.
        3. Mantén los títulos que comienzan con "#".
        4. Si el texto tiene enumeraciones, puedes usar listas.
        5. Usa conectores naturales y orden Sujeto-Verbo-Objeto.
        6. Evita tecnicismos; si aparecen, usa palabras más comunes o explica brevemente su significado.

        Texto original:
        {text}

        Texto simplificado:
        """  

        # Formato tipo chat
        messages = [{"role": "user", "content": custom_prompt}]
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
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
            simplified_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        elif "assistant" in generated_text:
            simplified_text = generated_text.split("assistant", 1)[-1].strip()
        else:
            simplified_text = generated_text.strip()

        # Se guarda el texto simplificado
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(simplified_text)

        print(f"✅ Simplificado: {filename}")

    print("\n🎯 Proceso completado. Los archivos simplificados están en:")
    print(f"   {os.path.abspath(output_folder)}")

finally:
    # ==========================
    # STOP TRACKER + CSV EUROPEO
    # ==========================
    _ = tracker.stop()

    csv_file = f"emissions_{project_name}_text.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df.to_csv(csv_file, sep=';', decimal=',', index=False, encoding='utf-8')
        print(f"✅ Archivo convertido al formato europeo: {csv_file}")
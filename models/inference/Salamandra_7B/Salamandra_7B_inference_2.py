"""
Inference script: Text simplification using fine-tuned BSC-LT/salamandra-7b-instruct (Causal LM) with LoRA adapters (PEFT), with prompt engineering and CodeCarbon tracking (offline).

This script generates simplified versions of Spanish clinical texts using:
- Base model: BSC-LT/salamandra-7b-instruct.
- LoRA adapter previously fine-tuned for text simplification.
- A structured prompt defining explicit clarity and readability constraints.

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
project_name = nombre_base.replace("_c2", "")

tracker = OfflineEmissionsTracker(
    project_name=project_name,
    output_file=f"emissions_{project_name}_text.csv",
    country_iso_code="ESP"
)

# ======================
# CONFIGURACIÓN DE RUTAS
# ======================
input_folder = "../../originales_txt_test"          
output_folder = "../../Conjunto2_fprompt/simplificaciones/salamandra_7B_instruct"    
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
    base_model = "BSC-LT/salamandra-7b-instruct"
    adapter_dir = "../../finetuning/FineTuningSalamandra-7B-Instruct"  

    print("🔹 Cargando modelo base y adaptador LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    
        # Tokenizar e inferir
        inputs = tokenizer(custom_prompt, return_tensors="pt").to("cuda")
    
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

        # Busca y se queda solo con el contenido posterior a ### Response:
        pattern = r"Texto\s*simplificado\s*:\s*(.*)"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            simplified_text = match.group(1).strip()
        else:
            simplified_text = generated_text.strip()

        simplified_text = re.sub(r"\n{3,}", "\n\n", simplified_text).strip()

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
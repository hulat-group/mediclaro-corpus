"""
Inference script: Text simplification using using fine-tuned BSC-LT/salamandra-7b-instruct (Causal LM) with LoRA adapters and reinforced prompt, including CodeCarbon tracking (offline).

Input:
- Folder containing original .txt files (clinical cases).
- Files are automatically sorted by case number extracted from filenames
  (pattern: "CasoClinico<NUMBER>").

Outputs:
- Adapted/simplified .txt files saved in the specified output directory.
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
project_name = nombre_base.replace("_c4", "")

tracker = OfflineEmissionsTracker(
    project_name=project_name,
    output_file=f"emissions_{project_name}_text.csv",
    country_iso_code="ESP"
)

# ======================
# CONFIGURACIÓN DE RUTAS
# ======================
input_folder = "../../originales_txt_test"         
output_folder = "../../Conjunto4_reforzado/simplificaciones/salamandra_7B_instruct"   
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

    print(" Cargando modelo base y adaptador LoRA...")
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
        custom_prompt = f"""Eres un asistente experto en Lenguaje Claro basado en la norma UNE-ISO 24495-1, especializado en adaptar notas clínicas para que sean claras, comprensibles y útiles para pacientes. 

        Aplica estos criterios:
        0. Generalidades
        - Utiliza frases claras, cortas y con una sola idea por oración.
        - Usa voz activa y tono respetuoso y empático.
        - Prefiere palabras sencillas, evitando tecnicismos o explicándolos claramente.
        - Usa listas y párrafos breves para facilitar la lectura.
        - Mantén un tono inclusivo y profesional, adecuado para pacientes sin formación médica.
        1. Estructura
        - Permite que la sección Evolución quede vacía si no hay datos. 
        - Anticipa el Diagnóstico antes del Tratamiento y Evolución, si es pertinente.  
        - Segrega la información en secciones claras y en el siguiente orden: Motivo de la consulta, Historia Clínica, Examen, Diagnóstico, Tratamiento y Evolución.  
        2. Redacción
        - Abreviaturas y siglas con formato alternativo facilitado, si es pertinente: por ejemplo, en vez de TSH se dirá de “indicador de tiroides (TSH)”.
        - Búsqueda de léxico alternativo facilitado para expresiones médicas.
        - Mantén el diagnóstico médico, pero se puede acompañar de una explicación comprensible.

        Adapta el siguiente texto, siguiendo estas pautas
        Texto original:
        {text}
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
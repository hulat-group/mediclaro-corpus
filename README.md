# 📊 BENCHMARK FOR SPANISH TEXT SIMPLIFICATION EVALUATION

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 🌐

---

## 👥 Team

**HULAT-UC3M (Human Language and Accessibility Technologies)**  
Universidad Carlos III de Madrid  

- Paloma Martínez
- Lourdes Moreno  
- Jesús M. Sánchez-Gomez  
- Marco Antonio Sanchez-Escudero

---

## 📝 Approach Summary

A benchmark for Spanish medical text simplification models based on human-written Easy-to-Read simplifications 

---

## 📈 Results


---

## 📚 Corpus

The corpus with the 50 original texts and the adapted texts by three different experts is available on demand in Zenodo: [MEDICLARO CORPUS](http://dx.doi.org/10.5281/zenodo.18385767).

---

## 🤖 Models

- **Llama-3.2-1B-Instruct**
  - [Official model on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    
- **Llama-3.2-3B-Instruct**
  - [Official model on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
    
- **Medical-mT5-xl**
  - [Official model on Hugging Face](https://huggingface.co/HiTZ/Medical-mT5-xl)
   
- **RigoChat-7B-v2**  
  - [Official model on Hugging Face](https://huggingface.co/IIC/RigoChat-7b-v2)  

- **Salamandra-2B-instruct**
  - [Official model on Hugging Face](https://huggingface.co/BSC-LT/salamandra-2b-instruct)
    
- **Salamandra-7B-instruct**
  - [Official model on Hugging Face](https://huggingface.co/BSC-LT/salamandra-7b-instruct)  

⚠️ Model weights are **not included** in this repository due to size restrictions.  

---

## ⚙️ Repository Structure

- notebooks/ → Exploratory analysis and internal evaluation7
- docs/ → Simplified texts 
- json/ → Metrics result
- src/ → Source code (models, metrics, prompts)

---

## 📖 Citation

---

## Funding

This work has been supported by grants:
- PID2023-148577OB-C21 (Human-Centered AI: User-Driven Adapted Language Models-HUMAN\_AI) by MICIU/AEI/10.13039/501100011033 and by FEDER/UE.
- 2024/00752/001 (Adaptación y optimización de modelos de lenguaje de gran tamaño para la generación de textos en lenguaje claro y lectura fácil en español, Adaptation and optimization of large language models for the generation of plain language and easy-to-read texts in Spanish) by Universidad Carlos III de Madrid, Ayudas para la Actividad Investigadora de los Jóvenes Doctores, del Programa Propio de Investigación.

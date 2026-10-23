# -*- coding: utf-8 -*-
import os
import re
import json
import time
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Defina GEMINI_API_KEY no .env")

MODELO = "gemini-3.5-flash-lite"
PASTA_TEXTOS  = "textos_extraidos"
PASTA_SAIDA   = "resultados"
ARQ_MATRIZ    = "matriz_referencia_enem.json"
ARQ_GABARITO  = "gabarito.txt"

IDENTIFICACAO_CADERNO = {
    "ano": 2016, "dia": 1, "numero": 1, "cor": "AZUL", "aplicacao": "APLICAÇÃO_REGULAR"
}
ESTUDANTE = {
    "nome_completo": "Brenda Vitória Soares Alessio",
    "colegio": "CCM Darcy José Costa",
    "turma": "3B",
}

os.makedirs(PASTA_SAIDA, exist_ok=True)

def carregar_gabarito(caminho):
    gabarito = {}
    if not os.path.exists(caminho): return gabarito
    padrao = re.compile(r"^\s*(\d+)\s*[-\.\)]?\s*([A-Ea-e])(?:\s*\((\w+)\))?\s*$")
    with open(caminho, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"): continue
            m = padrao.match(l)
            if m:
                num, letra, idioma = int(m.group(1)), m.group(2).upper(), m.group(3)
                if idioma: gabarito[(num, idioma.lower())] = letra
                else: gabarito[num] = letra
    return gabarito

def carregar_json(c):
    with open(c, encoding="utf-8") as f: return json.load(f)

SUGESTION_PROMPT = """Você é um analista do ENEM. Classifique a questão segundo a Matriz.
Retorne APENAS o JSON no schema fornecido. Não invente itens fora da Matriz.
Atenção: NÃO use caracteres de barra invertida (como \\pi ou \\frac) no JSON. Descreva as fórmulas em texto puro."""

def montar_prompt(txt, matriz, estrutura, gab, num):
    return f"""{SUGESTION_PROMPT}
MATRIZ: {json.dumps(matriz, ensure_ascii=False)}
SCHEMA: {json.dumps(estrutura, ensure_ascii=False)}
METADADOS: Nº {num} | Idioma: português | Ano: {IDENTIFICACAO_CADERNO['ano']}
GABARITO: {gab if gab else "NÃO DISPONÍVEL"}
TEXTO: \"\"\"{txt}\"\"\"
Retorne o JSON."""

def analisar_questao(txt, matriz, estrutura, gab, num):
    prompt = montar_prompt(txt, matriz, estrutura, gab, num)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        }
    }

    for t in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                dados = resp.json()
                texto = dados['candidates'][0]['content']['parts'][0]['text']
                texto = re.sub(r"^```(?:json)?|```$", "", texto, flags=re.MULTILINE).strip()
                
                # Correção de escape
                texto = re.sub(r'\\(?![\\"/bfnrtu])', r'\\\\', texto)
                texto = texto.replace("\n", " ").replace("\r", " ")

                return json.loads(texto)
            elif resp.status_code == 429:
                print(f"    [COTA] Limite de requisições. Esperando 70s... ({t+1}/5)")
                time.sleep(70)
            elif resp.status_code == 503:
                print(f"    [SERVIDOR LOTADO] Google indisponível. Esperando 15s... ({t+1}/5)")
                time.sleep(15)
            else:
                print(f"    [ERRO HTTP {resp.status_code}] {resp.text[:150]}")
                time.sleep(5)
        except Exception as e:
            print(f"    [ERRO DE JSON] {t+1}/5: {e}")
            time.sleep(5)
    return None

def main():
    print("RECUPERAÇÃO — ENEM 2016-1-1 AZUL (Aplicação Regular)")
    
    # Lista fixa das questões que falharam
    faltantes = ["questao-6.txt", "questao-64.txt", "questao-65.txt", "questao-67.txt"]
    
    matriz = carregar_json(ARQ_MATRIZ)
    estrutura = carregar_json("estrutura.json")
    gabarito = carregar_gabarito(ARQ_GABARITO)
    
    for nome_arq in tqdm(faltantes, desc="Recuperando"):
        arq = os.path.join(PASTA_TEXTOS, nome_arq)
        if not os.path.exists(arq):
            print(f"[AVISO] {arq} não encontrado.")
            continue
            
        with open(arq, encoding="utf-8") as f: txt = f.read().strip()
        if not txt: continue
        
        # Extrai o número do arquivo
        num = int(re.search(r"(\d+)", nome_arq).group(1))
        gab = gabarito.get(num)
        
        print(f"\n>>> Processando {nome_arq} (nº {num}) gab={gab}")
        res = analisar_questao(txt, matriz, estrutura, gab, num)
        
        if res:
            q = res.setdefault("questao", {})
            q.setdefault("identificacao_caderno", {}).update(IDENTIFICACAO_CADERNO)
            iq = q.setdefault("identificacao_questao", {})
            iq.update({"numero": num, "ingles": False, "espanhol": False})
            cq = q.setdefault("conteudo_questao", {})
            cq["caminho_texto"] = arq.replace("\\", "/")
            if gab: cq["gabarito"] = gab
            res["estudante"] = ESTUDANTE
            
            nome_base = Path(arq).stem
            with open(os.path.join(PASTA_SAIDA, f"{nome_base}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"    ✅ Salvo com sucesso: {nome_base}.json")
        else:
            print(f"    ❌ Falhou novamente. O Google pode estar instável.")
            
    print("\n🏁 Recuperação finalizada!")

if __name__ == "__main__":
    main()
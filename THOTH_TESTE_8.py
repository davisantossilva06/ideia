import os
import re
import json
import time
import shutil
import sqlite3
import random
import threading
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import PyPDF2

try:
    from sentence_transformers import SentenceTransformer
    HAVE_ST = True
except Exception:
    HAVE_ST = False

try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")

def limpa_txt(t: str) -> str:
    t = (t or "").replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def chunkear(texto: str, alvo=1000, overlap=150):
    texto = limpa_txt(texto)
    if len(texto) <= alvo:
        return [texto] if texto else []
    partes = []
    i = 0
    L = len(texto)
    while i < L:
        j = i + alvo
        if j >= L:
            partes.append(texto[i:].strip()); break
        corte = texto.rfind(". ", i, j)
        if corte == -1:
            corte = texto.rfind(" ", i, j)
        if corte == -1:
            corte = j
        partes.append(texto[i:corte].strip())
        i = max(corte - overlap, 0) + 1
    return [p for p in partes if len(p) > 24]

class BancoDeDadosConhecimento:
    def __init__(self, caminho_base="thoth_conhecimento"):
        self.caminho_base = Path(caminho_base)
        self._criar_estrutura_inicial()
        self.conexao = sqlite3.connect(self.caminho_base / "conhecimento.db")
        self._criar_tabelas()

    def _criar_estrutura_inicial(self):
        self.caminho_base.mkdir(exist_ok=True)
        (self.caminho_base / "conversas").mkdir(exist_ok=True)
        (self.caminho_base / "documentos").mkdir(exist_ok=True)
        (self.caminho_base / "backups").mkdir(exist_ok=True)

    def _criar_tabelas(self):
        c = self.conexao.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS conhecimentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conteudo TEXT NOT NULL,
            fonte_url TEXT,
            caminho_arquivo TEXT,
            tipo_conteudo TEXT,
            quantidade_palavras INTEGER,
            data_processamento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags TEXT,
            interconexoes TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sessao_id TEXT UNIQUE,
            titulo_sessao TEXT,
            data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ultima_interacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS mensagens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sessao_id TEXT,
            mensagem_usuario TEXT,
            resposta_thoth TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sessao_id) REFERENCES conversas (sessao_id)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS pastas_conhecimento (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caminho_pasta TEXT UNIQUE,
            capacidade_total_gb REAL,
            espaco_utilizado_gb REAL DEFAULT 0,
            data_adicao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self.conexao.commit()

    def adicionar_conhecimento(self, conteudo, fonte_url=None, caminho_arquivo=None, tipo_conteudo="texto"):
        c = self.conexao.cursor()
        tags = self._extrair_tags(conteudo)
        inter = self._identificar_interconexoes(conteudo)
        c.execute("""
        INSERT INTO conhecimentos (conteudo, fonte_url, caminho_arquivo, tipo_conteudo, quantidade_palavras, tags, interconexoes)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            conteudo, fonte_url, caminho_arquivo, tipo_conteudo, len(conteudo.split()), tags, inter
        ))
        self.conexao.commit()
        return c.lastrowid

    def _extrair_tags(self, conteudo):
        palavras = conteudo.lower().split()
        stop = {'o','a','os','as','de','da','do','em','um','uma','é','são','que','com','para','por','na','no'}
        freq = {}
        for p in palavras:
            if p not in stop and len(p) > 3:
                freq[p] = freq.get(p, 0) + 1
        tags_ordenadas = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return json.dumps([t for t,_ in tags_ordenadas])

    def _identificar_interconexoes(self, novo_conteudo):
        c = self.conexao.cursor()
        c.execute('SELECT id, conteudo FROM conhecimentos ORDER BY id DESC LIMIT 80')
        base = c.fetchall()
        inter = []
        novo = set(limpa_txt(novo_conteudo).lower().split())
        for kid, cont in base:
            comum = novo.intersection(set(limpa_txt(cont).lower().split()))
            if len(comum) > 8:
                inter.append(f"link_{kid}_{len(comum)}")
        return json.dumps(inter)

    def buscar_conhecimento_relevante(self, pergunta, limite=5):
        c = self.conexao.cursor()
        termos = [t for t in limpa_txt(pergunta).lower().split() if len(t) > 2]
        if not termos:
            return []
        cond = " OR ".join(["conteudo LIKE ?" for _ in termos])
        params = [f'%{p}%' for p in termos]
        c.execute(f"""
        SELECT conteudo, fonte_url, tipo_conteudo, tags, interconexoes
        FROM conhecimentos WHERE {cond}
        ORDER BY quantidade_palavras DESC LIMIT ?
        """, params + [limite])
        return c.fetchall()

    def criar_sessao(self, titulo=None):
        sessao = f"sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        c = self.conexao.cursor()
        c.execute("INSERT INTO conversas (sessao_id, titulo_sessao) VALUES (?, ?)",
                  (sessao, titulo or f"Conversa {datetime.now().strftime('%d/%m/%Y %H:%M')}"))
        self.conexao.commit()
        return sessao

    def salvar_msg(self, sessao_id, msg, resp):
        c = self.conexao.cursor()
        c.execute("INSERT INTO mensagens (sessao_id, mensagem_usuario, resposta_thoth) VALUES (?,?,?)",
                  (sessao_id, msg, resp))
        c.execute("UPDATE conversas SET ultima_interacao=CURRENT_TIMESTAMP WHERE sessao_id=?",(sessao_id,))
        self.conexao.commit()

    def listar_sessoes(self):
        c = self.conexao.cursor()
        c.execute("SELECT sessao_id, titulo_sessao, data_criacao, ultima_interacao FROM conversas ORDER BY ultima_interacao DESC")
        return c.fetchall()

    def historico(self, sessao_id):
        c = self.conexao.cursor()
        c.execute("SELECT mensagem_usuario, resposta_thoth, timestamp FROM mensagens WHERE sessao_id=? ORDER BY timestamp ASC",
                  (sessao_id,))
        return c.fetchall()

    def adicionar_pasta_conhecimento(self, caminho_pasta, capacidade_gb):
        c = self.conexao.cursor()
        caminho_abs = str(Path(caminho_pasta).resolve())
        try:
            c.execute("INSERT INTO pastas_conhecimento (caminho_pasta, capacidade_total_gb) VALUES (?,?)",
                      (caminho_abs, capacidade_gb))
            self.conexao.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def listar_pastas_conhecimento(self):
        c = self.conexao.cursor()
        c.execute("SELECT * FROM pastas_conhecimento")
        pastas = c.fetchall()
        out = []
        for p in pastas:
            caminho = p[1]
            uso_gb = self._tamanho_pasta_em_gb(caminho) if os.path.exists(caminho) else 0.0
            out.append(p + (uso_gb,))
        return out

    def _tamanho_pasta_em_gb(self, caminho):
        total = 0
        for arq in Path(caminho).rglob("*"):
            if arq.is_file():
                total += arq.stat().st_size
        return total/(1024**3)

    def estatisticas(self):
        c = self.conexao.cursor()
        c.execute('SELECT COUNT(*) FROM conhecimentos WHERE tipo_conteudo="livro"')
        livros = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM conhecimentos')
        total = c.fetchone()[0]
        c.execute('SELECT SUM(quantidade_palavras) FROM conhecimentos')
        palavras = c.fetchone()[0] or 0
        c.execute('SELECT COUNT(DISTINCT sessao_id) FROM conversas')
        conversas = c.fetchone()[0]
        return dict(livros_processados=livros, total_conhecimentos=total, total_palavras=palavras, total_conversas=conversas)

    def aprender(self, texto: str, fonte_url=None, caminho_arquivo=None, tipo="memoria"):
        texto = limpa_txt(texto or "")
        if not texto: return None
        rid = self.adicionar_conhecimento(texto, fonte_url, caminho_arquivo, tipo)
        return rid

class IntegradorConhecimento:
    def __init__(self, banco: BancoDeDadosConhecimento):
        self.banco = banco
        self.cor = {'info':'\033[94m','ok':'\033[92m','warn':'\033[93m','err':'\033[91m','run':'\033[96m','rst':'\033[0m'}

    def _status(self, msg, tipo='info'):
        print(f"{self.cor.get(tipo,self.cor['info'])}[{datetime.now().strftime('%H:%M:%S')}] {msg}{self.cor['rst']}")

    def integrar_url(self, url, usar_tor=False):
        self._status(f"Iniciando integração: {url}", 'run')
        try:
            html = self._baixar(url, usar_tor)
            if not html:
                self._status("Falha ao obter conteúdo", 'err'); return False
            tipo = self._detectar_tipo(html)
            self._status(f"Tipo detectado: {tipo}", 'ok')
            if tipo == "livro":
                texto = self._processar_livro(html)
            else:
                texto = self._processar_site(html)
            self.banco.adicionar_conhecimento(texto, fonte_url=url, caminho_arquivo=None, tipo_conteudo=tipo)
            self._status("Integração concluída.", 'ok')
            return True
        except Exception as e:
            self._status(f"Erro: {e}", 'err')
            return False

    def _baixar(self, url, usar_tor):
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        kwargs = dict(headers=headers, timeout=30)
        if usar_tor:
            kwargs['proxies'] = {'http':'socks5h://127.0.0.1:9050','https':'socks5h://127.0.0.1:9050'}
        r = requests.get(url, **kwargs)
        r.raise_for_status()
        return r.text

    def _detectar_tipo(self, html):
        sopa = BeautifulSoup(html, 'html.parser')
        txt = sopa.get_text(separator=" ").lower()
        marcadores = ['capitulo','capítulo','pagina','página','livro','autor','editora']
        return "livro" if any(m in txt for m in marcadores) else "website"

    def _processar_livro(self, html):
        sopa = BeautifulSoup(html,'html.parser')
        for e in sopa(["script","style","nav","header","footer","noscript"]): e.decompose()
        linhas = [l.strip() for l in sopa.get_text("\n").split("\n") if len(l.strip())>30]
        texto = "\n".join(linhas)
        self._status(f"Livro processado: {len(texto)} chars", 'ok')
        return texto

    def _processar_site(self, html):
        sopa = BeautifulSoup(html, 'html.parser')
        for e in sopa(["script","style","nav","header","footer","noscript"]): e.decompose()
        linhas = [l.strip() for l in sopa.get_text("\n").split("\n") if len(l.strip())>50]
        texto = "\n".join(linhas)
        self._status(f"Site processado: {len(texto)} chars", 'ok')
        return texto

    def processar_pdf(self, caminho_pdf: str):
        self._status(f"Processando PDF: {caminho_pdf}", 'run')
        try:
            with open(caminho_pdf,'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full = []
                for i, pg in enumerate(reader.pages, 1):
                    txt = pg.extract_text() or ""
                    full.append(f"\n--- Pagina {i} ---\n{txt}")
                    self._status(f"Página {i}/{len(reader.pages)}", 'info')
                destino = self.banco.caminho_base / "documentos" / Path(caminho_pdf).name
                shutil.copy2(caminho_pdf, destino)
                self.banco.adicionar_conhecimento("\n".join(full), fonte_url=None, caminho_arquivo=str(destino), tipo_conteudo="pdf")
            self._status("PDF processado.", 'ok')
            return True
        except Exception as e:
            self._status(f"Erro PDF: {e}", 'err'); return False

class IndiceSemantico:
    def __init__(self, banco: BancoDeDadosConhecimento, modelo="thenlper/gte-base"):
        self.banco = banco
        self.modelo_nome = modelo
        self.modelo = None
        self.dim = 384
        self.textos = []
        self.metas = []
        self.idx = None
        self._carregar_modelo()
        self._reconstruir()

    def _carregar_modelo(self):
        if HAVE_ST:
            try:
                self.modelo = SentenceTransformer(self.modelo_nome)
                self.dim = self.modelo.get_sentence_embedding_dimension()
            except Exception:
                self.modelo = None

    def _embed(self, textos):
        if self.modelo is None:
      
            import random
            vecs = []
            for t in textos:
                random.seed(hash(t) & 0xffffffff)
                vecs.append([random.random() for _ in range(self.dim)])
            return vecs
        return self.modelo.encode(textos, convert_to_numpy=True, normalize_embeddings=False, batch_size=64)

    def _reconstruir(self):
        c = self.banco.conexao.cursor()
        c.execute("SELECT conteudo, fonte_url, caminho_arquivo FROM conhecimentos ORDER BY id ASC")
        linhas = c.fetchall()
        lotes, metas = [], []
        for cont, f, arq in linhas:
            for p in chunkear(cont, alvo=900, overlap=120):
                lotes.append(p); metas.append((f, arq))
        if not lotes:
            self.textos, self.metas, self.idx = [], [], None
            return
        emb = self._embed(lotes)
        self.textos, self.metas = lotes, metas
        if HAVE_FAISS:
            import numpy as np
            emb = np.array(emb, dtype="float32")
            faiss.normalize_L2(emb)
            self.idx = faiss.IndexFlatIP(self.dim)
            self.idx.add(emb)
        else:
            self.idx = emb  

    def buscar(self, consulta: str, k=6):
        consulta = limpa_txt(consulta)
        if not self.textos:
            return []
        q = self._embed([consulta])[0]
        if HAVE_FAISS and isinstance(self.idx, faiss.IndexFlatIP):
            import numpy as np
            q = np.array([q], dtype="float32"); faiss.normalize_L2(q)
            D, I = self.idx.search(q, min(k, len(self.textos)))
            out = []
            for idx, score in zip(I[0], D[0]):
                if idx < 0: continue
                out.append({"texto": self.textos[idx], "fonte": self.metas[idx][0], "arquivo": self.metas[idx][1], "score": float(score)})
            return out
       
        from math import sqrt
        def cos(a,b):
            num = sum(x*y for x,y in zip(a,b))
            den = sqrt(sum(x*x for x in a)) * sqrt(sum(y*y for y in b))
            return (num/den) if den else 0.0
        scores = [(i, cos(q, v)) for i,v in enumerate(self.idx)]
        scores.sort(key=lambda x: x[1], reverse=True)
        out=[]
        for i, sc in scores[:k]:
            out.append({"texto": self.textos[i], "fonte": self.metas[i][0], "arquivo": self.metas[i][1], "score": float(sc)})
        return out

from typing import Optional, Dict, Any

class GoogleDriveCliente:
    ESCOPO = ["https://www.googleapis.com/auth/drive.file"] 

    def __init__(self, pasta_local_base: Path):
        self.pasta_local = Path(pasta_local_base)
        self._service = None  

    def _autenticar(self):
       
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        token_path = Path("token.json")
        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self.ESCOPO)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                credfile = Path("credentials.json")
                if not credfile.exists():
                    raise FileNotFoundError(
                        "credentials.json não encontrado. Crie um OAuth Client ID (Desktop) no Google Cloud e salve aqui.")
                flow = InstalledAppFlow.from_client_secrets_file(str(credfile), self.ESCOPO)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json(), encoding="utf-8")

        from googleapiclient.discovery import build
        self._service = build("drive", "v3", credentials=creds)

    @property
    def service(self):
        if self._service is None:
            self._autenticar()
        return self._service

    def _buscar_pasta_por_nome(self, nome: str) -> Optional[str]:
        q = f"name = '{nome}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        resp = self.service.files().list(q=q, pageSize=10, fields="files(id, name)").execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def _criar_pasta(self, nome: str, parent_id: Optional[str]=None) -> str:
        meta = {
            "name": nome,
            "mimeType": "application/vnd.google-apps.folder"
        }
        if parent_id:
            meta["parents"] = [parent_id]
        pasta = self.service.files().create(body=meta, fields="id").execute()
        return pasta["id"]

    def _assegurar_pasta(self, nome: str, parent_id: Optional[str]=None) -> str:
        pid = self._buscar_pasta_por_nome(nome)
        if pid: return pid
        return self._criar_pasta(nome, parent_id)

    def _tamanho_total_local(self) -> int:
        total = 0

        alvos = [
            self.pasta_local / "conhecimento.db",
            self.pasta_local / "documentos",
            self.pasta_local / "conversas",
            self.pasta_local / "backups",
        ]
        for alvo in alvos:
            if alvo.is_file():
                total += alvo.stat().st_size
            elif alvo.is_dir():
                for arq in alvo.rglob("*"):
                    if arq.is_file():
                        total += arq.stat().st_size
        return total

    def _upload_arquivo(self, caminho: Path, parent_id: str):
        from googleapiclient.http import MediaFileUpload
        body = {"name": caminho.name, "parents": [parent_id]}
        media = MediaFileUpload(str(caminho), resumable=True)
        req = self.service.files().create(body=body, media_body=media, fields="id")
        resp = None
        while resp is None:
            status, resp = req.next_chunk()
        
        return resp["id"]

    def _assegurar_subpasta(self, nome: str, parent_id: str) -> str:
  
        q = f"'{parent_id}' in parents and name = '{nome}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
        resp = self.service.files().list(q=q, pageSize=10, fields="files(id,name)").execute()
        files = resp.get("files", [])
        if files:
            return files[0]["id"]
        return self._criar_pasta(nome, parent_id)

    def enviar_cerebro(self) -> Dict[str, Any]:
        """
        Cria (ou reaproveita) a pasta 'THOTH' no Drive e envia:
        - conhecimento.db
        - /documentos, /conversas, /backups
        Retorna um dicionário com bytes_enviados e ids criados.
        """
        total_bytes = self._tamanho_total_local()
        print(f"\n[Drive] Tamanho total a enviar: {total_bytes/(1024**2):.2f} MB")

        raiz_id = self._assegurar_pasta("THOTH") 

        enviados = 0
        ids = {"raiz": raiz_id, "arquivos": []}

        db_path = self.pasta_local / "conhecimento.db"
        if db_path.exists():
            print(f"[Drive] Enviando DB: {db_path.name}")
            arq_id = self._upload_arquivo(db_path, raiz_id)
            ids["arquivos"].append({"nome": db_path.name, "id": arq_id})
            enviados += db_path.stat().st_size
          
        for nome in ["documentos","conversas","backups"]:
            sub = self.pasta_local / nome
            if not sub.exists(): 
                continue
            sub_id = self._assegurar_subpasta(nome, raiz_id)
            for arq in sub.rglob("*"):
                if arq.is_file():
                    rel = arq.relative_to(sub)
                 
                    parent = sub_id
                    partes = list(rel.parts)
                    if len(partes) > 1:
                   
                        for pasta in partes[:-1]:
                            parent = self._assegurar_subpasta(pasta, parent)
                    print(f"[Drive] Enviando: {arq.name}")
                    hid = self._upload_arquivo(arq, parent)
                    ids["arquivos"].append({"nome": str(rel), "id": hid})
                    enviados += arq.stat().st_size

        manifest = {
            "gerado_em": datetime.utcnow().isoformat()+"Z",
            "tamanho_total_bytes": total_bytes,
            "enviado_total_bytes": enviados,
            "arquivos": ids["arquivos"]
        }
        man_local = self.pasta_local / "manifest.json"
        man_local.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[Drive] Enviando manifest.json")
        _ = self._upload_arquivo(man_local, raiz_id)

        return {"bytes_totais": total_bytes, "bytes_enviados": enviados, "pasta_drive_id": raiz_id}

def gerar_resposta_llm(pergunta: str, blocos: list) -> str:
    """Tenta Ollama; se falhar, tenta OpenAI; caso contrário, devolve síntese."""
    sistema = (
        "Você é uma IA privada. Responda APENAS com base nos trechos fornecidos. "
        "Cite como [n]. Se faltar evidência, diga que não há dados suficientes."
    )
    prompt = sistema + "\n\n"
    for i, b in enumerate(blocos, 1):
        prompt += f"[{i}] {b}\n\n"
    prompt += f"Pergunta: {pergunta}\n\nResposta objetiva, com referências [n]:"

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True, "options":{"temperature":0.2}},
                          timeout=120, stream=True)
        if r.ok:
            partes = []
            for ln in r.iter_lines(decode_unicode=True):
                if not ln: continue
                try:
                    j = json.loads(ln)
                    if "response" in j: partes.append(j["response"])
                except Exception:
                    pass
            return "".join(partes).strip()
    except Exception:
        pass

    if OPENAI_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            comp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sistema},{"role":"user","content":prompt}],
                temperature=0.2,
            )
            return comp.choices[0].message.content.strip()
        except Exception:
            pass

    out = ["[sem motor LLM ativo] Síntese dos trechos:"]
    for i,b in enumerate(blocos,1):
        tb = limpa_txt(b)
        out.append(f"[{i}] {tb[:400]}{'...' if len(tb)>400 else ''}")
    return "\n".join(out)

def montar_resposta(pergunta: str, evidencias: list) -> str:
    if not evidencias:
        return "Não encontrei evidências suficientes nos seus arquivos para responder com segurança."
    blocos = []
    fontes = []
    for i, e in enumerate(evidencias[:6], 1):
        blocos.append(limpa_txt(e["texto"]))
        origem = e.get("fonte") or e.get("arquivo") or "origem desconhecida"
        fontes.append(f"- [{i}] {origem} (score {e.get('score',0.0):.2f})")
    resp = gerar_resposta_llm(pergunta, blocos)
    resp += "\n\n[CITAÇÕES]\n" + "\n".join(fontes)
    return resp

class MotorConversa:
    def __init__(self, banco: BancoDeDadosConhecimento):
        self.banco = banco
        self.sessao_atual = None
        self._indice = None
        self._lock_ind = threading.Lock()

    def _indice_sem(self) -> IndiceSemantico:
        with self._lock_ind:
            if self._indice is None:
                try:
                    self._indice = IndiceSemantico(self.banco)
                except Exception:
                    self._indice = None
        return self._indice

    def iniciar_sessao(self, titulo=None):
        self.sessao_atual = self.banco.criar_sessao(titulo or "")

    def gerar_resposta(self, mensagem_usuario: str) -> str:
        msg = mensagem_usuario.strip()

        if msg.startswith("/lembrar "):
            conteudo = limpa_txt(msg[len("/lembrar "):])
            self.banco.aprender(conteudo, tipo="memoria_usuario")
            resp = "Anotado e aprendido na memória interna."
            if self.sessao_atual: self.banco.salvar_msg(self.sessao_atual, msg, resp)
            return resp

        if msg.startswith("/aprende "):
            pasta = limpa_txt(msg[len("/aprende "):].strip('"').strip("'"))
            cont = self._ingest_pasta(pasta)
            resp = f"Ingestão concluída: {cont} arquivos."
            if self.sessao_atual: self.banco.salvar_msg(self.sessao_atual, msg, resp)
            return resp

        if msg.startswith("/googledrive"):
          
            partes = msg.split()
            email = partes[1] if len(partes) >= 2 else None
            try:
                resp = self._enviar_google_drive(email)
            except Exception as e:
                resp = f"Falha no envio ao Google Drive: {e}"
            if self.sessao_atual: self.banco.salvar_msg(self.sessao_atual, msg, resp)
            return resp

        evid_sem = []
        ind = self._indice_sem()
        if ind:
            try:
                evid_sem = ind.buscar(msg, k=6)
            except Exception:
                evid_sem = []
        evid_lex = []
        try:
            brutos = self.banco.buscar_conhecimento_relevante(msg, limite=6) or []
            for (cont, fonte, tipo, tags, inter) in brutos:
                evid_lex.append({"texto": cont, "fonte": fonte, "arquivo": None, "score": 0.0})
        except Exception:
            pass

        evid = dedup_evidencias(evid_sem + evid_lex, limite=8)
        if not evid:
           
            resp = "Explorando… ainda não tenho material suficiente nos seus arquivos. Use /aprende \"pasta\" ou integre URLs no menu."
        else:
            resp = montar_resposta(msg, evid)

        if self.sessao_atual:
            self.banco.salvar_msg(self.sessao_atual, msg, resp)
            
            pacote = f"Pergunta: {msg}\nResposta: {resp}"
            self.banco.aprender(pacote, tipo="qa_conversa")
        return resp

    def _ingest_pasta(self, pasta: str) -> int:
        base = Path(pasta)
        if not base.exists():
            return 0
        cont = 0
        integ = IntegradorConhecimento(self.banco)
        for arq in base.rglob("*"):
            if not arq.is_file(): continue
            low = arq.suffix.lower()
            try:
                if low in [".txt",".md",".csv",".log"]:
                    txt = arq.read_text(encoding="utf-8", errors="ignore")
                    self.banco.aprender(txt, caminho_arquivo=str(arq), tipo="arquivo_texto")
                    cont += 1
                elif low in [".html",".htm"]:
                    raw = arq.read_text(encoding="utf-8", errors="ignore")
                    sopa = BeautifulSoup(raw, "lxml")
                    for e in sopa(["script","style","nav","header","footer","noscript"]): e.decompose()
                    txt = limpa_txt(sopa.get_text(" "))
                    self.banco.aprender(txt, caminho_arquivo=str(arq), tipo="html")
                    cont += 1
                elif low == ".pdf":
                    integ.processar_pdf(str(arq))
                    cont += 1
            except Exception:
                pass
       
        self._indice = None
        _ = self._indice_sem()
        return cont

    def _enviar_google_drive(self, email: str=None) -> str:
        """
        Dispara fluxo OAuth (1ª vez) e envia o “cérebro” para a pasta THOTH do Drive.
        O parâmetro 'email' é apenas informativo de UX; OAuth decide a conta.
        """
        cliente = GoogleDriveCliente(self.banco.caminho_base)
        resumo = cliente.enviar_cerebro()
        mb_total = resumo["bytes_totais"]/(1024**2)
        mb_env = resumo["bytes_enviados"]/(1024**2)
        return (f"Cérebro transferido para o Google Drive.\n"
                f"- Conta: {email or 'detectada via OAuth'}\n"
                f"- Pasta: THOTH (id {resumo['pasta_drive_id']})\n"
                f"- Tamanho total: {mb_total:.2f} MB | Enviado: {mb_env:.2f} MB\n"
                f"- Manifest salvo no Drive e local.")

def dedup_evidencias(items, limite=8):
    vistos = set()
    out = []
    for e in items:
        chave = hash(limpa_txt(e["texto"])[:420])
        if chave in vistos: 
            continue
        vistos.add(chave)
        out.append(e)
        if len(out) >= limite: break
    return out

class THOTH:
    def __init__(self):
        self.banco = BancoDeDadosConhecimento()
        self.integrador = IntegradorConhecimento(self.banco)
        self.chat = MotorConversa(self.banco)
        self.frases = [
            "O conhecimento é um rio que se renova a cada pergunta.",
            "A mente aberta cresce quanto mais se enche.",
            "Perguntas são chaves; cada uma abre uma sala de ideias.",
            "A sabedoria é a arte de fazer conexões improváveis."
        ]

    def cabeçalho(self):
        asciiart = r"""
████████╗██╗  ██╗ ██████╗ ████████╗██╗  ██╗
╚══██╔══╝██║  ██║██╔═══██╗╚══██╔══╝██║  ██║
   ██║   ███████║██║   ██║   ██║   ███████║
   ██║   ██╔══██║██║   ██║   ██║   ██╔══██║
   ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝

      Sistema de Conhecimento Integrado
"""
        print("\033[1;36m"+asciiart+"\033[0m")

    def saudacao(self):
        return random.choice(self.frases)

    def menu(self):
        while True:
            print("\n"+"="*70)
            print("\033[1;35mMENU PRINCIPAL - THOTH\033[0m")
            print("="*70)
            print("1. Integrar Conhecimento da Web")
            print("2. Processar Arquivo PDF")
            print("3. Iniciar Nova Conversa (tempo real + comandos)")
            print("4. Acessar Conversas Anteriores")
            print("5. Gerenciar Conhecimento")
            print("6. Gerenciar Armazenamento")
            print("7. Status do Sistema")
            print("8. Sair")
            print("="*70)
            esc = input("\nEscolha: ").strip()
            if esc == "1":
                self._menu_web()
            elif esc == "2":
                self._menu_pdf()
            elif esc == "3":
                self._conversa()
            elif esc == "4":
                self._anteriores()
            elif esc == "5":
                self._gerenciar_conhecimento()
            elif esc == "6":
                self._armazenamento()
            elif esc == "7":
                self._status()
            elif esc == "8":
                print("\nAté breve. O rio do conhecimento te espera.")
                break
            else:
                print("Opção inválida.")

    def _menu_web(self):
        print("\n— Integração Web —")
        print("1. Superfície (normal)")
        print("2. Tor (deep)")
        print("3. Voltar")
        op = input("Escolha: ").strip()
        if op in ["1","2"]:
            usar_tor = (op=="2")
            url = input("URL para estudar: ").strip()
            if url:
                self.integrador.integrar_url(url, usar_tor)

    def _menu_pdf(self):
        print("\n— PDF —")
        caminho = input("Caminho completo do PDF: ").strip()
        if os.path.exists(caminho) and caminho.lower().endswith(".pdf"):
            self.integrador.processar_pdf(caminho)
        else:
            print("Arquivo não encontrado ou inválido.")

    def _conversa(self):
        print("\n— Nova Conversa (tempo real) —")
        print('Comandos: /lembrar <texto> • /aprende "C:\\minha_pasta" • /googledrive meuemail@gmail.com')
        self.chat.iniciar_sessao(input("Título (opcional): ").strip())
        print("Digite 'sair' para encerrar.\n")
        while True:
            msg = input("Você: ").strip()
            if msg.lower() in ["sair","exit","quit"]:
                break
            if not msg:
                continue
            resp = self.chat.gerar_resposta(msg)
            print("\nTHOTH: ", end="", flush=True)
            for ch in resp:
                print(ch, end="", flush=True)
                time.sleep(0.0015)
            print("\n")

    def _anteriores(self):
        s = self.banco.listar_sessoes()
        if not s:
            print("Sem conversas armazenadas."); return
        for i, (sid, tit, cri, ult) in enumerate(s,1):
            print(f"{i}. {tit}  [{sid}]  Última: {ult}")
        try:
            k = int(input("Escolha nº (0 para voltar): "))
        except ValueError:
            return
        if not (1 <= k <= len(s)):
            return
        sid = s[k-1][0]
        hist = self.banco.historico(sid)
        print("\n— Histórico —")
        for mu, rt, ts in hist:
            print(f"\n[{ts}] Você: {mu}")
            print(f"[{ts}] THOTH: {rt}")

        print("\n— Continuar —  (digite 'sair' para voltar)")
        self.chat.sessao_atual = sid
        while True:
            msg = input("\nVocê: ").strip()
            if msg.lower() in ["sair","exit","quit"]:
                break
            resp = self.chat.gerar_resposta(msg)
            print(f"\nTHOTH: {resp}")

    def _gerenciar_conhecimento(self):
        while True:
            print("\n— Gerenciar Conhecimento —")
            print("1. Listar fontes integradas (com URL)")
            print("2. Voltar")
            op = input("Escolha: ").strip()
            if op == "1":
                c = self.banco.conexao.cursor()
                c.execute("""SELECT id, fonte_url, tipo_conteudo, data_processamento
                             FROM conhecimentos WHERE fonte_url IS NOT NULL
                             ORDER BY data_processamento DESC""")
                fts = c.fetchall()
                if not fts:
                    print("Nenhuma fonte com URL.")
                else:
                    for i,(fid,url,tipo,data) in enumerate(fts,1):
                        print(f"{i}. {url} ({tipo}) - {data}")
            else:
                break

    def _armazenamento(self):
        while True:
            print("\n— Armazenamento —")
            print("1. Adicionar pasta de conhecimento")
            print("2. Listar pastas e uso")
            print("3. Voltar")
            op = input("Escolha: ").strip()
            if op == "1":
                p = input("Caminho da pasta: ").strip()
                try:
                    cap = float(input("Capacidade total (GB): "))
                    ok = self.banco.adicionar_pasta_conhecimento(p, cap)
                    print("Adicionada." if ok else "Já existe.")
                except ValueError:
                    print("Capacidade inválida.")
            elif op == "2":
                lst = self.banco.listar_pastas_conhecimento()
                if not lst:
                    print("Nenhuma pasta.")
                for pid, caminho, cap, usado, data, uso_real in lst:
                    perc = (uso_real/cap*100) if cap>0 else 0
                    print(f"\nPasta: {caminho}\nCap: {cap:.1f} GB  Usado: {uso_real:.2f} GB ({perc:.1f}%)  Desde: {data}")
            else:
                break

    def _status(self):
        est = self.banco.estatisticas()
        pastas = self.banco.listar_pastas_conhecimento()
        print("\n=== STATUS THOTH ===")
        print(f"Livros: {est['livros_processados']}")
        print(f"Conhecimentos: {est['total_conhecimentos']}")
        print(f"Palavras: {est['total_palavras']:,}")
        print(f"Conversas: {est['total_conversas']}")
        if pastas:
            cap = sum(p[2] for p in pastas)
            usado = sum(p[5] for p in pastas)
            perc = (usado/cap*100) if cap>0 else 0
            print(f"Pastas: {len(pastas)} | Cap total: {cap:.1f} GB | Usado: {usado:.1f} GB ({perc:.1f}%)")
        db_path = self.banco.caminho_base/"conhecimento.db"
        if db_path.exists():
            print(f"DB: {(db_path.stat().st_size/(1024**2)):.1f} MB")
        print(f"Agora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

def main():
    app = THOTH()
    app.cabeçalho()
    print(f"\n\033[3m{app.saudacao()}\033[0m\n")
    app.menu()

if __name__ == "__main__":
    main()

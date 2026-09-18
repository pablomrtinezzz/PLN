"""
Práctica 1: Similitud de documentos con Bag-of-Words y Similitud del Coseno
"""

import re
import math
import docx
import torch
import torch.nn.functional as F
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# 0. ETIQUETAS DE LOS DOCUMENTOS (para identificarlos en los resultados)
# ─────────────────────────────────────────────────────────────────────────────
DOC_LABELS = {
    "doc01": "Móvil – Smartphone gama alta",
    "doc02": "Móvil – Batería smartphones",
    "doc03": "Móvil – Apps más descargadas",
    "doc04": "Eléctrico – Coches eléctricos",
    "doc05": "Eléctrico – Subvenciones",
    "doc06": "Móvil – Pantallas plegables",
    "doc07": "Móvil – Realidad aumentada",
    "doc08": "Eléctrico – EcoDrive económico",
    "doc09": "Eléctrico – Supercargadores",
    "doc10": "Móvil – Seguridad móvil",
    "doc11": "Mixto – Integración smartphone+coche",
    "doc12": "Mixto – App carga eléctrica",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. STOP WORDS en español (conjunto mínimo)
# ─────────────────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "es", "esta",
    "esas", "ese", "eso", "estas", "este", "estos", "fue", "han", "has",
    "hasta", "hay", "la", "las", "le", "les", "lo", "los", "más", "me", "mi",
    "mientras", "mismo", "muy", "na", "ni", "no", "nos", "o", "para", "pero",
    "por", "que", "quien", "quienes", "se", "ser", "si", "sin", "sobre",
    "son", "su", "sus", "también", "tan", "tanto", "te", "their", "this",
    "todo", "todos", "un", "una", "uno", "unos", "ya", "y", "yo",
    "ha", "he", "he", "les", "lo", "nos", "os", "su", "tuvo", "fue",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_document(path: str) -> str:
    """Extrae el texto completo de un archivo .docx."""
    doc = docx.Document(path)
    return " ".join(p.text for p in doc.paragraphs if p.text.strip())


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Preprocesamiento:
      1. Convierte a minúsculas.
      2. Elimina caracteres no alfabéticos (conserva letras con tilde/ñ).
      3. (Opcional) Elimina stop words.
    """
    text = text.lower()
    # Conservar letras del alfabeto español (incluye acentuadas y ñ)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def load_corpus(doc_ids: list[str], remove_stopwords: bool = True):
    """Carga y preprocesa todos los documentos. Devuelve lista de tokens por doc."""
    corpus_tokens = {}
    for doc_id in doc_ids:
        raw = load_document(f"{doc_id}.docx")
        corpus_tokens[doc_id] = preprocess(raw, remove_stopwords)
    return corpus_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCCIÓN DEL VOCABULARIO Y VECTORES BOW
# ─────────────────────────────────────────────────────────────────────────────

def build_vocabulary(corpus_tokens: dict) -> list[str]:
    """Crea un vocabulario global ordenado alfabéticamente."""
    vocab_set = set()
    for tokens in corpus_tokens.values():
        vocab_set.update(tokens)
    return sorted(vocab_set)


def tokens_to_bow_vector(tokens: list[str], vocab: list[str]) -> torch.Tensor:
    """Convierte una lista de tokens en un vector BoW (conteo de frecuencias)."""
    word2idx = {w: i for i, w in enumerate(vocab)}
    counts = Counter(tokens)
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for word, count in counts.items():
        if word in word2idx:
            vec[word2idx[word]] = int(count)
    return vec


def build_bow_matrix(corpus_tokens: dict, vocab: list[str]) -> tuple[torch.Tensor, list[str]]:
    """
    Construye la matriz BoW [N_docs × |vocab|].
    Retorna (matriz, lista_ids).
    """
    doc_ids = list(corpus_tokens.keys())
    vectors = [tokens_to_bow_vector(corpus_tokens[d], vocab) for d in doc_ids]
    return torch.stack(vectors), doc_ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def compute_tfidf_matrix(corpus_tokens: dict, vocab: list[str]) -> torch.Tensor:
    """
    Calcula la matriz TF-IDF [N_docs × |vocab|].
      TF(t,d)  = frecuencia del término t en el documento d (normalizada).
      IDF(t)   = log( N / df(t) )  donde df(t) = nº de docs que contienen t.
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    doc_ids  = list(corpus_tokens.keys())
    N        = len(doc_ids)
    V        = len(vocab)

    # Frecuencia de documento (df)
    df = torch.zeros(V, dtype=torch.float32)
    for tokens in corpus_tokens.values():
        unique_in_doc = set(tokens)
        for w in unique_in_doc:
            if w in word2idx:
                df[word2idx[w]] += 1.0

    idf = torch.log(torch.tensor(N, dtype=torch.float32) / (df + 1e-9))

    # TF normalizado por longitud del documento
    tfidf_rows = []
    for d in doc_ids:
        tokens = corpus_tokens[d]
        counts = Counter(tokens)
        tf_vec = torch.zeros(V, dtype=torch.float32)
        total  = len(tokens) if tokens else 1
        for w, c in counts.items():
            if w in word2idx:
                tf_vec[word2idx[w]] = c / total
        tfidf_rows.append(tf_vec * idf)

    return torch.stack(tfidf_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIMILITUD DEL COSENO (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calcula la matriz de similitud del coseno [N × N] para todos los pares
    de documentos usando torch.nn.functional.cosine_similarity.
    """
    n = matrix.shape[0]
    sim_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = F.cosine_similarity(
                matrix[i].unsqueeze(0),
                matrix[j].unsqueeze(0)
            )
    return sim_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISIS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def print_similarity_matrix(sim_matrix: torch.Tensor, doc_ids: list[str], title: str):
    """Imprime la matriz de similitud con formato tabular."""
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    short = [d.replace("doc", "D") for d in doc_ids]
    header = "       " + "  ".join(f"{s:>4}" for s in short)
    print(header)
    print("       " + "──────" * len(doc_ids))
    for i, d in enumerate(doc_ids):
        row = f"{short[i]:>5} │ " + "  ".join(f"{sim_matrix[i,j]:.2f}" for j in range(len(doc_ids)))
        print(row)


def top_pairs(sim_matrix: torch.Tensor, doc_ids: list[str], n: int = 5, highest: bool = True):
    """Devuelve los n pares con mayor (o menor) similitud (excluyendo diagonal)."""
    pairs = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            pairs.append((sim_matrix[i, j].item(), doc_ids[i], doc_ids[j]))
    pairs.sort(key=lambda x: x[0], reverse=highest)
    return pairs[:n]


def print_top_pairs(sim_matrix: torch.Tensor, doc_ids: list[str]):
    print("\n── Pares MÁS similares ──────────────────────────────────────────────")
    for score, a, b in top_pairs(sim_matrix, doc_ids, n=5, highest=True):
        print(f"  {a} ({DOC_LABELS[a]})  ↔  {b} ({DOC_LABELS[b]})")
        print(f"    Similitud: {score:.4f}")

    print("\n── Pares MENOS similares ────────────────────────────────────────────")
    for score, a, b in top_pairs(sim_matrix, doc_ids, n=5, highest=False):
        print(f"  {a} ({DOC_LABELS[a]})  ↔  {b} ({DOC_LABELS[b]})")
        print(f"    Similitud: {score:.4f}")


def compare_bow_vs_tfidf(bow_sim: torch.Tensor, tfidf_sim: torch.Tensor, doc_ids: list[str]):
    """Muestra los cambios más notables entre BoW y TF-IDF."""
    print("\n── Diferencias BoW vs TF-IDF (mayores cambios absolutos) ───────────")
    diffs = []
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            delta = abs(tfidf_sim[i, j].item() - bow_sim[i, j].item())
            diffs.append((delta, bow_sim[i,j].item(), tfidf_sim[i,j].item(),
                          doc_ids[i], doc_ids[j]))
    diffs.sort(reverse=True)
    for delta, bow_s, tfidf_s, a, b in diffs[:5]:
        print(f"  {a} ↔ {b}")
        print(f"    BoW={bow_s:.4f}  TF-IDF={tfidf_s:.4f}  Δ={delta:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    doc_ids = [f"doc{i:02d}" for i in range(1, 13)]

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Práctica: Similitud de documentos con BoW y Cosine Similarity  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── 2. Carga y preprocesamiento ──────────────────────────────────────────
    print("\n[1] Cargando y preprocesando documentos...")
    corpus_tokens = load_corpus(doc_ids, remove_stopwords=True)
    for d in doc_ids:
        print(f"  {d} ({DOC_LABELS[d]}): {len(corpus_tokens[d])} tokens")

    # ── 3. Vocabulario ───────────────────────────────────────────────────────
    vocab = build_vocabulary(corpus_tokens)
    print(f"\n[2] Vocabulario global: {len(vocab)} palabras únicas")

    # ── 4. Matriz BoW ────────────────────────────────────────────────────────
    print("\n[3] Construyendo matriz Bag-of-Words...")
    bow_matrix, ordered_ids = build_bow_matrix(corpus_tokens, vocab)
    print(f"  Forma de la matriz BoW: {bow_matrix.shape}")

    # ── 5. Similitud coseno BoW ──────────────────────────────────────────────
    print("\n[4] Calculando similitud del coseno (BoW)...")
    bow_sim = cosine_similarity_matrix(bow_matrix)
    print_similarity_matrix(bow_sim, ordered_ids, "Matriz de Similitud del Coseno — BoW")
    print_top_pairs(bow_sim, ordered_ids)

    # ── 6. TF-IDF ────────────────────────────────────────────────────────────
    print("\n\n[5] Calculando vectores TF-IDF...")
    tfidf_matrix = compute_tfidf_matrix(corpus_tokens, vocab)
    print(f"  Forma de la matriz TF-IDF: {tfidf_matrix.shape}")

    print("\n[6] Calculando similitud del coseno (TF-IDF)...")
    tfidf_sim = cosine_similarity_matrix(tfidf_matrix)
    print_similarity_matrix(tfidf_sim, ordered_ids, "Matriz de Similitud del Coseno — TF-IDF")
    print_top_pairs(tfidf_sim, ordered_ids)

    # ── 7. Comparación BoW vs TF-IDF ─────────────────────────────────────────
    compare_bow_vs_tfidf(bow_sim, tfidf_sim, ordered_ids)

    print("\n" + "═"*70)
    print("  Fin de la práctica.")
    print("═"*70)


if __name__ == "__main__":
    main()

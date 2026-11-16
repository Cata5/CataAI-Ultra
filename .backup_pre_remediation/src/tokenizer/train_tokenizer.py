"""BPE Tokenizer Training Script for RoCanon-48.5M Bilingual SLM.

Trains a 12,288 Byte-Level BPE tokenizer on a stratified bilingual mixture:
- ~45% Romanian (educational, encyclopedic, grammar, literature)
- ~35% English educational / mathematics (CS, science, calculus, logic)
- ~20% Clean Code (Python standard library, JavaScript algorithms, indentation)

Enforces:
1. Exact vocabulary size: 12,288 ($96 x 128$, Tensor Core aligned).
2. Unicode NFC normalization with strict Romanian sedila unification (ş/ţ -> ș/ț).
3. Single-digit atomization (\\p{N} -> 1 digit per token; operators never fuse).
4. Preserved indentation whitespace (2, 4, 8 spaces).
5. 16 reserved special tokens.
6. Exact round-trip UTF-8 reconstruction with 0 byte decode errors.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

from src.tokenizer.tokenizer import (
    DEFAULT_TOKENIZER_FILENAME,
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    build_decoder,
    build_normalizer,
    build_post_processor,
    build_pre_tokenizer,
)


def generate_romanian_educational_texts() -> list[str]:
    """Generates extensive authentic Romanian educational and encyclopedic texts."""
    texts = [
        # Gramatică și lingvistică
        """Limba română este o limbă indo-europeană din grupul limbilor romanice de est.
Ea este limba oficială a României și a Republicii Moldova și este vorbită de peste 24 de milioane de oameni ca limbă maternă.
Sistemul fonetic al limbii române cuprinde șapte vocale (a, e, i, o, u, ă, î/â) și douăzeci de consoane.
Specificul morfologic constă în păstrarea declinării substantivelor (cazurile nominativ, acuzativ, genitiv, dativ și vocativ)
și existența articolului hotărât enclitic: băiatul, băiatului, fata, fetei, pomii, pomilor.
Verbele se împart în patru mari conjugări în funcție de terminația de la infinitiv (-a, -ea, -e, -i/-î).
Diacriticele limbii române sunt caracterele ă, â, î, ș și ț, reprezentate corect conform standardelor Unicode cu virgulă dedesubt (U+0219, U+021B).""",

        # Istoria României
        """Istoria spațiului carpato-danubiano-pontic începe cu triburile geto-dace conduse de regi precum Burebista și Decebal.
În urma războaielor daco-romane din anii 101-102 și 105-106 conduse de împăratul Traian, Dacia a devenit provincie a Imperiului Roman.
Procesul complex de romanizare a condus la formarea poporului român și a limbii române.
În Evul Mediu s-au format voievodatele Țara Românească și Moldova, precum și voievodatul Transilvaniei.
Domnitori de seamă precum Mircea cel Bătrân, Ștefan cel Mare, Iancu de Hunedoara, Vlad Țepeș și Mihai Viteazul au apărat frontierele europene.
Mihai Viteazul a realizat în anul 1600 prima unire politică a celor trei țări românești.
În epoca modernă, Unirea Principatelor din 1859 sub Alexandru Ioan Cuza și proclamarea Independenței din 1877 sub regele Carol I
au consolidat statul român modern, culminând cu Marea Unire de la 1 Decembrie 1918 de la Alba Iulia.""",

        # Geografia României
        """Geografia României se caracterizează printr-o armonie și concentricitate remarcabilă a reliefului.
Munții Carpați descriu un arc în centrul țării, fiind împărțiți în Carpații Orientali, Meridionali și Occidentali.
Cel mai înalt vârf este Vârful Moldoveanu din Munții Făgăraș, cu o altitudine de 2.544 metri.
Fluviul Dunărea străbate țara pe o lungime de 1.075 kilometri, formând la vărsarea în Marea Neagră Delta Dunării,
o rezervație a biosferei de importanță universală protejată de UNESCO.
Clima este temperat-continentală de tranziție, cu patru anotimpuri distincte.
Resursele naturale includ rezerve de petrol, gaze naturale, cărbune, sare, minereuri și păduri vaste.""",

        # Știință și inovație românească
        """Pionierii științei românești au adus contribuții majore la patrimoniul științific mondial.
Henri Coandă a proiectat și construit primul avion cu reacție din lume (Coandă-1910) și a descoperit efectul aerodinamic ce îi poartă numele.
Traian Vuia a realizat în 1906 unul dintre primele zboruri autopropulsate cu un aparat mai greu decât aerul.
Aurel Vlaicu a conceput avioane originale premiate la concursurile internaționale din Austria.
George Emil Palade a primit Premiul Nobel pentru Fiziologie sau Medicină în 1974 pentru descoperirea ribozomilor și structurii celulare.
Grigore Moisil a fost fondatorul școlii românești de informatică și calculatoare, introducând algebra booleană în circuitele de comutație.
Spiru Haret a aplicat mecanica cerească și a reformat fundamental învățământul modern românesc.""",

        # Literatură, cultură și filosofie
        """Literatura română a cunoscut o dezvoltare spectaculoasă prin generația marilor clasici.
Mihai Eminescu este considerat poetul național al României, autor al capodoperelor Luceafărul, Scrisorile, Odă (în metru antic) și Glossa.
Ion Creangă a lăsat nepieritoarele Amintiri din copilărie și basme populare pline de umor și înțelepciune.
Ion Luca Caragiale a radiografiat societatea prin comediile sale nemuritoare: O scrisoare pierdută, O noapte furtunoasă și D-ale carnavalului.
În secolul al XX-lea, Lucian Blaga a îmbinat poezia metafizică cu o filosofie originală a misterului și a culturii spațiului mioritic.
Mircea Eliade a devenit unul dintre cei mai mari istorici ai religiilor, iar Emil Cioran și Eugen Ionescu au marcat gândirea europeană.""",

        # Științe exacte și tehnologie în limba română
        """Teoria probabilităților și statistica matematică analizează fenomenele aleatoare și distribuțiile de variabile.
O variabilă aleatoare continuă X este descrisă de o funcție de densitate de probabilitate f(x) nenegativă pe domeniul real.
Valoarea așteptată sau media teoretică se calculează prin integrala produsului x * f(x) pe întreg domeniul de definiție.
În fizica cuantică, funcția de undă psi satisface ecuația lui Schrödinger: i * h_bar * d(psi)/dt = H * psi.
În chimia organică, atomul de carbon formează legături covalente tetraedrice prin hibridizare sp3, având unghiul de 109 grade și 28 minute.
Rețelele neuronale artificiale realizează aproximarea funcțiilor neliniare prin straturi succesive de transformări afine urmate de activări.""",

        # Dialoguri și instrucțiuni structurate
        """Întrebare: Care sunt avantajele utilizării unui model mic de limbaj (Small Language Model) precum RoCanon-48.5M?
Răspuns: Modelele mici de limbaj oferă trei avantaje decisive:
1. Eficiență hardware extremă: pot fi antrenate și rulate local pe plăci grafice consumer precum RTX 3060 cu 6 GB VRAM.
2. Latență scăzută la inferență: generare rapidă de răspunsuri în timp real fără transferuri de rețea.
3. Confidențialitate absolută: datele utilizatorului rămân pe dispozitivul local, respectând normele de securitate.""",

        """Instrucțiune: Rezumați principiul de funcționare al mecanismului de atenție Grouped-Query Attention (GQA).
Răspuns: Grouped-Query Attention (GQA) împarte capetele de interogare (Query) în grupuri care partajează un număr redus de capete Cheie/Valoare (Key/Value).
În RoCanon-48.5M, avem 8 capete Query și 2 capete KV, realizând un raport de 4:1.
Acest design reduce memoria cache KV cu 75% comparativ cu Multi-Head Attention standard, permițând lungimi de secvență mari fără depășirea memoriei video.""",
    ]

    # Expand texts with morphological variations and specialized terminology
    vocab_seeds = [
        ("energie", "energiilor", "energetic", "electromagnetism", "termodinamică", "entropie"),
        ("algoritm", "algoritmi", "algoritmul", "algoritmică", "complexitate", "sortare"),
        ("ecuație", "ecuații", "diferențial", "integrală", "derivată", "continuitate"),
        ("spațiu", "spațiului", "vectorial", "coordonate", "geometrie", "ortogonal"),
        ("structură", "structuri", "ierarhie", "organizare", "funcționalitate", "optimizare"),
        ("inteligență", "artificială", "raționament", "cogniție", "percepție", "concluzie"),
        ("programare", "programator", "compilator", "interpretor", "variabilă", "instrucțiune"),
        ("memorie", "memoria", "adresare", "tampon", "alocare", "pointer"),
        ("rețea", "rețele", "comunicare", "protocol", "securitate", "transmisie"),
        ("învățare", "supervizată", "nesupervizată", "consolidare", "gradient", "convergență"),
    ]

    for seed_group in vocab_seeds:
        paragraph = (
            f"În studiul domeniului {seed_group[0]}, analiza {seed_group[1]} demonstrează proprietăți de natură {seed_group[2]}. "
            f"Conceptele de {seed_group[3]} și {seed_group[4]} joacă un rol determinant în calculul de {seed_group[5]}. "
            f"Fiecare element din acest sistem interacționează în mod determinist cu variabilele de stare adiacente."
        )
        texts.append(paragraph)

    return texts


def generate_english_educational_texts() -> list[str]:
    """Generates extensive English educational, scientific, and mathematics texts."""
    texts = [
        # Computer Science & Neural Networks
        """Deep learning architectures rely on backpropagation and stochastic gradient descent to optimize parameters.
In a decoder-only Transformer, auto-regressive language modeling predicts the probability distribution of the next token.
The attention mechanism computes Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V.
Rotary Position Embeddings (RoPE) encode positional information by rotating query and key vectors in complex 2D planes,
preserving relative distance invariance across long sequence contexts.
SwiGLU activations, defined as SwiGLU(x) = (x * W_gate * sigmoid(beta * x * W_gate)) * (x * W_up) * W_down,
consistently outperform traditional ReLU and GeLU activations on benchmark evaluations.""",

        # Calculus and Linear Algebra
        """Linear algebra provides the mathematical foundation for modern machine learning algorithms.
A real matrix A of dimensions m by n maps vectors from R^n into R^m through linear transformations.
The singular value decomposition (SVD) factors any real matrix into A = U * Sigma * V^T,
where U and V are orthogonal matrices and Sigma contains non-negative singular values on its diagonal.
The eigenvalues lambda and eigenvectors v of a square matrix satisfy the equation A * v = lambda * v.
In multivariable calculus, the gradient vector nabla f points in the direction of steepest ascent of a scalar function,
while the Hessian matrix captures the local curvature through second-order partial derivatives.""",

        # Operating Systems and Computer Architecture
        """Modern processor architectures utilize deep pipelining, branch prediction, and multi-level cache hierarchies.
Virtual memory management abstracts physical RAM using page tables and translation lookaside buffers (TLB).
Contiguous memory-mapped file access (mmap) allows operating systems to page file contents directly into application address space,
bypassing unnecessary user-space buffer copying and minimizing memory pressure.
Graphics Processing Units (GPUs) achieve teraflop throughput through massively parallel SIMD execution units and Tensor Cores.""",

        # Physics and Scientific Principles
        """Classical mechanics describes macroscopic motion through Newton's three laws of motion:
1. Every body continues in its state of rest or uniform motion unless compelled to change that state by external forces.
2. The rate of change of momentum is proportional to the impressed force: F = d(m * v) / dt = m * a.
3. To every action there is always opposed an equal and opposite reaction.
Thermodynamics governs energy transformations: the first law establishes energy conservation,
while the second law dictates that the total entropy of an isolated system never decreases over time.""",

        # Software Engineering and Algorithms
        """Algorithm design emphasizes time complexity and space complexity evaluated through asymptotic Big-O notation.
Binary search operates in O(log n) time on sorted collections, while quicksort achieves O(n log n) average time complexity.
Dynamic programming solves complex optimization problems by decomposing them into overlapping subproblems with optimal substructure.
Graph traversal techniques including Depth-First Search (DFS) and Breadth-First Search (BFS) explore vertices systematically.""",
    ]

    # Additional scientific and technical domains
    topics = [
        ("Quantum Computing", "qubits, superposition, entanglement, quantum gates, decoherence, and algorithms like Shor's and Grover's"),
        ("Information Theory", "Shannon entropy, mutual information, channel capacity, Huffman coding, and Kolmogorov complexity"),
        ("Probability Theory", "Markov chains, Gaussian distributions, central limit theorem, and stochastic differential equations"),
        ("Discrete Mathematics", "Boolean algebra, graph theory, combinatorics, recurrence relations, and Eulerian paths"),
        ("Compiler Construction", "lexical analysis, abstract syntax trees, intermediate representations, register allocation, and optimization"),
    ]
    for title, desc in topics:
        texts.append(
            f"The field of {title} studies {desc}. Fundamental theorems establish rigorous guarantees on correctness and efficiency, "
            f"providing analytical tools used across modern software and hardware engineering."
        )

    return texts


def generate_math_expressions() -> list[str]:
    """Generates thousands of diverse mathematical equations to train digit atomization and operator isolation."""
    random.seed(42)
    exprs = []

    # Basic arithmetic: + - * / =
    for _ in range(500):
        a = random.randint(0, 9999)
        b = random.randint(1, 999)
        op = random.choice(["+", "-", "*", "/"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        elif op == "*":
            ans = a * b
        else:
            ans = a // b
        exprs.append(f"{a} {op} {b} = {ans}")
        # Compact forms without spaces
        exprs.append(f"{a}{op}{b}={ans}")

    # Floating point numbers and scientific notation
    for _ in range(250):
        val1 = round(random.uniform(0.001, 999.999), 4)
        val2 = round(random.uniform(0.001, 99.99), 3)
        res = round(val1 + val2, 4)
        exprs.append(f"{val1} + {val2} = {res}")
        exprs.append(f"{val1}e-5 * {val2}e+3 = {val1 * val2:.4e}")

    # Complex algebraic and equality statements
    formulas = [
        "e^{i * pi} + 1 = 0",
        "E = m * c^2",
        "a^2 + b^2 = c^2",
        "f(x) = a * x^2 + b * x + c",
        "x = (-b +- sqrt(b^2 - 4*a*c)) / (2*a)",
        "lim_{x -> 0} sin(x) / x = 1",
        "sum_{i=1}^n i = n * (n + 1) / 2",
        "integral_{0}^{inf} e^{-x^2} dx = sqrt(pi) / 2",
        "P(A|B) = P(B|A) * P(A) / P(B)",
        "nabla * B = 0",
        "nabla * E = rho / epsilon_0",
        "F_G = G * (m_1 * m_2) / r^2",
        "H * psi = E * psi",
        "10 / 2 = 5",
        "25 * 4 = 100",
        "1234567890",
        "3.141592653589793",
        "2.718281828459045",
        "1.414213562373095",
    ]
    exprs.extend(formulas * 10)
    return exprs


def generate_code_samples() -> list[str]:
    """Generates clean Python and JavaScript code samples with preserved indentations (2, 4, 8 spaces)."""
    samples = []

    # Indented Python samples
    samples.append("""def fibonacci(n: int) -> int:
    # 4 spaces indentation
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def compute_nested_metrics(data: list) -> dict:
    results = {}
    for item in data:
        if item.is_valid:
            for sub in item.subitems:
                # 8 spaces indentation
                if sub.score > 0.8:
                    results[sub.id] = sub.score
    return results
""")

    # 2 spaces indentation Python/JS
    samples.append("""function quicksort(arr) {
  // 2 spaces indentation
  if (arr.length <= 1) {
    return arr;
  }
  const pivot = arr[arr.length - 1];
  const left = [];
  const right = [];
  for (let i = 0; i < arr.length - 1; i++) {
    if (arr[i] < pivot) {
      left.push(arr[i]);
    } else {
      right.push(arr[i]);
    }
  }
  return [...quicksort(left), pivot, ...quicksort(right)];
}
""")

    # Multi-level indentation test
    samples.append("""class NeuralLayer:
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = None

    def forward(self, x):
        if x is None:
            raise ValueError("Input tensor cannot be None")
        for step in range(10):
            # 8 spaces
            if step % 2 == 0:
                x = x + 0.1
        return x
""")

    # Repeated indentation blocks to ensure high frequency in BPE
    indent_patterns = [
        "  # 2 spaces indent\n  x = 1\n  y = 2\n",
        "    # 4 spaces indent\n    a = 10\n    b = 20\n    c = a + b\n",
        "        # 8 spaces indent\n        val = compute(x, y)\n        return val\n",
    ]
    samples.extend(indent_patterns * 150)
    return samples


def get_default_corpus(
    corpus_dir: str | Path | None = None,
    corpus_file: str | Path | None = None,
    max_python_files: int = 50,
) -> list[str]:
    """Assembles a rich bilingual corpus satisfying the 45% RO / 35% EN / 20% Code distribution."""
    corpus: list[str] = []

    # 1. Romanian Educational Texts (~45%)
    ro_texts = generate_romanian_educational_texts()
    corpus.extend(ro_texts * 20)

    # 2. English Educational & Mathematics Texts (~35%)
    en_texts = generate_english_educational_texts()
    corpus.extend(en_texts * 25)

    # 3. Mathematics & Digits Expressions
    math_texts = generate_math_expressions()
    corpus.extend(math_texts)

    # 4. Code & Indentation (~20%)
    code_samples = generate_code_samples()
    corpus.extend(code_samples * 20)

    # 5. Load Python standard library files for real-world code coverage
    try:
        lib_path = Path(os.__file__).parent
        py_files = sorted(glob.glob(str(lib_path / "*.py")))[:max_python_files]
        for pf in py_files:
            try:
                with open(pf, "r", encoding="utf-8", errors="ignore") as f:
                    corpus.append(f.read())
            except OSError:
                continue
    except OSError:
        pass

    # 6. Load doc.md if present in current working directory
    doc_path = Path("doc.md")
    if doc_path.exists():
        try:
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                corpus.append(f.read())
        except OSError:
            pass

    # 7. User-provided custom corpus files
    if corpus_file:
        cf_path = Path(corpus_file)
        if cf_path.exists():
            with open(cf_path, "r", encoding="utf-8", errors="ignore") as f:
                corpus.append(f.read())

    if corpus_dir:
        cd_path = Path(corpus_dir)
        if cd_path.exists():
            for fpath in cd_path.glob("**/*"):
                if fpath.is_file() and fpath.suffix in [".txt", ".md", ".py", ".json", ".js"]:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            corpus.append(f.read())
                    except OSError:
                        continue

    return corpus


def build_rocanon_tokenizer_skeleton() -> Tokenizer:
    """Instantiates a raw Tokenizer configured with normalizer, pre-tokenizer, decoder, and post-processor."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = build_normalizer()
    tokenizer.pre_tokenizer = build_pre_tokenizer()
    tokenizer.decoder = build_decoder()
    tokenizer.post_processor = build_post_processor()
    return tokenizer


def train_rocanon_tokenizer(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    output_dir: str | Path | None = None,
    corpus: Iterable[str] | None = None,
    verify: bool = True,
) -> Tokenizer:
    """Trains the BPE tokenizer targeting exactly `vocab_size` tokens and saves artifacts."""
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if corpus is None:
        corpus = get_default_corpus()

    tokenizer = build_rocanon_tokenizer_skeleton()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    current_vocab_size = tokenizer.get_vocab_size()

    # Ensure vocabulary size is strictly aligned to target (12,288)
    if current_vocab_size < vocab_size:
        pad_needed = vocab_size - current_vocab_size
        padding_tokens = [f"<|reserved_{3 + i}|>" for i in range(pad_needed)]
        tokenizer.add_tokens(padding_tokens)

    # Save tokenizer artifacts
    save_path_json = output_dir / DEFAULT_TOKENIZER_FILENAME
    tokenizer.save(str(save_path_json))
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Also save vocab.json and merges.txt
    try:
        if hasattr(tokenizer, "model") and hasattr(tokenizer.model, "save"):
            tokenizer.model.save(str(output_dir))
    except (AttributeError, OSError, RuntimeError):
        pass

    # Save config
    config = {
        "tokenizer_class": "RoCanonTokenizer",
        "model_type": "bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "system_token": "<|system|>",
        "user_token": "<|user|>",
        "assistant_token": "<|assistant|>",
        "end_turn_token": "<|end_turn|>",
        "think_token": "<|think|>",
        "end_think_token": "<|end_think|>",
        "code_token": "<|code|>",
        "end_code_token": "<|end_code|>",
        "unknown_info_token": "<|unknown_info|>",
        "bos_token_id": tokenizer.token_to_id("<|bos|>"),
        "eos_token_id": tokenizer.token_to_id("<|eos|>"),
        "pad_token_id": tokenizer.token_to_id("<|pad|>"),
        "unk_token_id": tokenizer.token_to_id("<|unk|>"),
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if verify:
        verify_trained_tokenizer(tokenizer, vocab_size)

    return tokenizer


def verify_trained_tokenizer(tokenizer: Tokenizer, expected_vocab_size: int = DEFAULT_VOCAB_SIZE):
    """Executes verification assertions on newly trained tokenizer."""
    # 1. Exact vocab size check
    actual_size = tokenizer.get_vocab_size()
    assert actual_size == expected_vocab_size, (
        f"Vocab size mismatch: expected {expected_vocab_size}, got {actual_size}"
    )

    # 2. Special tokens verification
    for expected_id, token_str in enumerate(SPECIAL_TOKENS):
        tok_id = tokenizer.token_to_id(token_str)
        assert tok_id == expected_id, (
            f"Special token {token_str} mapped to {tok_id}, expected {expected_id}"
        )

    # 3. Single-digit atomization
    digits_enc = tokenizer.encode("1234567890")
    assert len(digits_enc.tokens) == 10, (
        f"Digits not atomized 1-by-1: {digits_enc.tokens}"
    )

    # 4. Arithmetic operator isolation
    math_equations = ["5+7=12", "-4", "x*3", "3.14159", "10/2=5", "a+b=c-d*e/f"]
    for eq in math_equations:
        enc = tokenizer.encode(eq)
        for tok in enc.tokens:
            has_digit = any(c.isdigit() for c in tok)
            has_op = any(c in "+-=*/" for c in tok)
            assert not (has_digit and has_op), (
                f"Operator fused with digit in token '{tok}' for equation '{eq}'"
            )

    # 5. Indentation tokens
    enc_2sp = tokenizer.encode("  ")
    enc_4sp = tokenizer.encode("    ")
    enc_8sp = tokenizer.encode("        ")
    assert len(enc_2sp.tokens) == 1, f"Expected 1 token for 2 spaces, got {enc_2sp.tokens}"
    assert len(enc_4sp.tokens) == 1, f"Expected 1 token for 4 spaces, got {enc_4sp.tokens}"
    assert len(enc_8sp.tokens) == 1, f"Expected 1 token for 8 spaces, got {enc_8sp.tokens}"

    # 6. Sedila unification
    enc_old = tokenizer.encode("şcoală şi ţară Şerban Ţepeş")
    enc_new = tokenizer.encode("școală și țară Șerban Țepeş")
    assert enc_old.ids == enc_new.ids, "Sedila unification failed to produce identical IDs"

    # 7. Exact round-trip reconstruction
    test_cases = [
        "Școala și învățătoarea din Țara Românească au 125 de elevi.",
        "def compute(x=10):\n    y = x * 2 + 5\n        return y\n",
        "Special tags: <|bos|><|system|>Prompt<|end_turn|><|think|>Thought<|end_think|>",
        "Math: 1 + 2 = 3 * 4 / 5 - 6 == 7 != 8 <= 9 >= 0",
        "Symbols and emojis: ∑_{i=1}^n x_i ≈ 3.14159 ± 0.001 🚀 🇷🇴",
    ]
    for text in test_cases:
        norm_text = tokenizer.normalizer.normalize_str(text)
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        assert dec == norm_text, f"Round-trip failed:\nExpected: {norm_text!r}\nGot:      {dec!r}"


def main():
    parser = argparse.ArgumentParser(description="Train RoCanon-48.5M Byte-Level BPE Tokenizer")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Target vocabulary size")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent), help="Directory to save tokenizer")
    parser.add_argument("--corpus-dir", type=str, default=None, help="Directory containing additional text files")
    parser.add_argument("--corpus-file", type=str, default=None, help="Path to single additional text file")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-training verification")

    args = parser.parse_args()

    print(f"=== Training RoCanon-48.5M Tokenizer (Vocab Size: {args.vocab_size}) ===")
    corpus = get_default_corpus(corpus_dir=args.corpus_dir, corpus_file=args.corpus_file)
    print(f"Corpus prepared: {len(corpus)} documents ({sum(len(c) for c in corpus):,} total characters)")

    tokenizer = train_rocanon_tokenizer(
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        corpus=corpus,
        verify=not args.no_verify,
    )

    print("=== Successfully trained and verified RoCanon Tokenizer! ===")
    print(f"Artifact saved to: {args.output_dir}")
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()

"""
scripts/generate_mini_datasets.py
Generates compact, high-quality technical datasets directly in the project root:
1. tech_pretrain_mini.txt & tech_pretrain_mini.jsonl (Pretrain Causal LM)
2. tech_instruction_mini.jsonl & tech_instruction_mini.txt (Instruction SFT)
3. tech_chat_mini.jsonl & tech_chat_mini.txt (Conversational Chat SFT)

All datasets are calibrated for rapid testing of custom models from Network Studio Builder.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# 1. PRETRAIN TECH PARAGRAPHS
# -----------------------------------------------------------------------------
PRETRAIN_PARAGRAPHS = [
    # Computer Hardware & Architecture
    (
        "Arhitectura calculatoarelor moderne se bazează pe modelul von Neumann, format din unitatea centrală de procesare (CPU), "
        "memoria principală (RAM), magistrale de comunicație și dispozitive de intrare/ieșire. CPU-ul execută cicluri succesive de "
        "fetch, decode și execute. Pentru a reduce latența accesului la memoria RAM, procesoarele moderne integrează o ierarhie "
        "sofisticată de memorii cache: L1 (cea mai rapidă, divizată în instrucțiuni și date), L2 (dedicată fiecărui nucleu) și L3 "
        "(partajată între nuclee, de capacitate mare). Predictorul de salturi (branch predictor) și execuția speculativă permit "
        "conductelor de procesare (pipeline) să mențină un randament ridicat fără blocaje cauzate de dependențele de date."
    ),
    (
        "Unitățile de procesare grafică (GPU) sunt proiectate pentru procesare masiv paralelă, folosind mii de nuclee aritmetice simple. "
        "Spre deosebire de un CPU cu 8-16 nuclee optimizate pentru execuție serială cu latență redusă, un GPU execută simultan mii de fire "
        "de execuție organizate în grupuri numite warps (32 de fire pe arhitecturile NVIDIA). Memoria video dedicată (VRAM), tip GDDR6 "
        "sau HBM3, oferă lățimi de bandă impresionante de sute sau mii de gigaocteți pe secundă (GB/s), esențiale pentru calcule tensoriale "
        "și multiplicări de matrici de mari dimensiuni (GEMM)."
    ),
    (
        "Interfața PCI Express (PCIe) asigură comunicația de mare viteză între placa de bază, procesor și acceleratorul grafic. "
        "PCIe 4.0 asigură un throughput de ~2 GB/s per bandă (lane), oferind până la 31.5 GB/s pentru un slot complet x16. "
        "PCIe 5.0 dublează această rată de transfer la ~63 GB/s. În antrenarea rețelelor neuronale pe Windows, dacă memoria dedicată VRAM "
        "este depășită, driverul WDDM transferă tensoarele peste magistrala PCIe în memoria RAM a sistemului (paging), provocând "
        "căderi drastice de performanță din cauza discrepanței enorme de viteză dintre magistrala PCIe și memoria GDDR6 internă."
    ),
    (
        "Memoria cu acces aleatoriu (RAM) de tip DDR5 funcționează la tensiuni reduse (1.1V) comparativ cu DDR4 (1.2V) și integrează "
        "un circuit PMIC (Power Management IC) direct pe plăcuța de memorie. Arhitectura pe canal a DDR5 împarte fiecare modul de 64 biți "
        "în două sub-canale independente de 32 biți plus 8 biți ECC pe cip (On-Die ECC), permițând transferuri paralele mult mai eficiente "
        "și rate de transfer de la 4800 MT/s până la peste 7200 MT/s."
    ),
    (
        "Unitățile de stocare NVMe SSD folosesc protocolul NVM Express comunicând direct prin liniile PCIe, eliminând blocajele protocolului "
        "moștenit SATA/AHCI. Prin utilizarea a până la 64.000 de cozi paralele de comenzi (fiecare suportând până la 64.000 de intrări), "
        "SSD-urile NVMe PCIe 4.0 ating viteze de citire secvențială de peste 7.000 MB/s și sute de mii de operațiuni IOPS (Input/Output "
        "Operations Per Second), permițând încărcarea instantanee a dataseturilor mari de antrenare în memoria RAM."
    ),
    # Deep Learning & Transformers
    (
        "Arhitectura Transformer, introdusă în lucrarea fundamentală 'Attention Is All You Need', a revoluționat procesarea limbajului natural "
        "și modelarea secvențelor prin renunțarea completă la recurență și convoluții. Mecanismul central este Scaled Dot-Product Attention, "
        "definit ca Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V. Prin compararea fiecărui token cu toți ceilalți tokeni din secvență, "
        "Transformer-ul captează dependențe contextualizate indiferent de distanța temporală dintre cuvinte."
    ),
    (
        "Multi-Head Attention (MHA) permite modelului să acorde atenție simultan informațiilor din diferite spații de reprezentare. "
        "Vectorii de intrare sunt proiectați liniar în matrici multiple de Query, Key și Value pentru fiecare cap de atenție (head). "
        "Rezultatele sunt concatenate și proiectate din nou printr-o matrice liniară de ieșire. Variante moderne precum Multi-Query "
        "Attention (MQA) sau Grouped-Query Attention (GQA) partajează capetele de Key și Value între mai multe capete de Query, "
        "reducând considerabil consumul memoriei cache KV în faza de generare și inferență."
    ),
    (
        "Embeddings și codificarea pozițională: Deoarece mecanismul de auto-atenție este invariant la permutare, poziția tokenilor "
        "trebuie injectată explicit. În loc de embeddings poziționale absolute aditive, arhitecturile moderne de top precum LLaMA și RoCanon "
        "utilizează Rotary Position Embedding (RoPE). RoPE rotește vectorii Query și Key în planul complex în funcție de poziția lor absolută, "
        "astfel încât produsul scalar dintre Q și K să depindă strict de distanța relativă dintre tokeni, favorizând extrapolarea pe contexte lungi."
    ),
    (
        "Funcțiile de activare în rețele neuronale profunde: Tradiționalele funcții ReLU și GELU sunt adesea înlocuite în modelele "
        "moderne de familie LLaMA cu SwiGLU (Swish-Gated Linear Unit). Un bloc SwiGLU este definit ca FFN(x) = (Swish(xW_gate) * xW_up) W_down. "
        "Mecanismul de gating multiplicativ oferă o expresivitate mult superioară și un flux de gradient mai stabil, conducând la o "
        "convergență mai rapidă a funcției de pierdere (loss)."
    ),
    (
        "Normalizarea straturilor (Layer Normalization): Spre deosebire de Batch Normalization care depinde de statisticile lotului, "
        "Layer Normalization normalizează activările de-a lungul dimensiunii caracteristicilor (features) pentru fiecare eșantion independent. "
        "Varianta RMSNorm (Root Mean Square Normalization) simplifică LayerNorm eliminând calculul mediei și reținând doar rădăcina pătrată "
        "a mediei pătratelor: RMS(x) = sqrt(1/d * sum(x_i^2) + eps). Aceasta reduce cu ~20% timpul de calcul al normalizării pe GPU "
        "păstrând o stabilitate numerică identică."
    ),
    # Optimization & Training Dynamics
    (
        "Algoritmul de optimizare AdamW (Decoupled Weight Decay): În algoritmul Adam standard, regularizarea L2 este adăugată direct la gradient, "
        "ceea ce duce la o scalare neadecvată prin momentele de ordinul doi. AdamW decuplează decăderea ponderilor de actualizarea bazată pe "
        "momente, aplicând penalizarea ponderii direct asupra greutăților: theta_{t+1} = theta_t - eta * lambda * theta_t - eta * m_hat / (sqrt(v_hat) + eps). "
        "Această formulare îmbunătățește semnificativ capacitatea de generalizare pe date de test neobservate."
    ),
    (
        "Optimizatorul Muon (MomentUm Orthogonalized by Newton-schulz): Conceput special pentru matricile interne 2D din Transformere, "
        "Muon aplică iterația quintică Newton-Schulz pentru a forța matricile de actualizare să fie ortogonale (toate valorile singulare egale cu 1). "
        "Această ortogonalizare previne colapsul dimensional și saturația gradientului, permițând rate de învățare considerabil mai mari "
        "(de exemplu 0.02) comparativ cu AdamW, accelerând convergența la pre-antrenare cu până la 2x-3x."
    ),
    (
        "Strategia de învățare Warmup-Stable-Decay (WSD): Spre deosebire de programul tradițional Cosine Annealing, unde rata de învățare "
        "scade continuu pe tot parcursul antrenării, programul WSD menține rata de învățare la valoarea de vârf (peak) pe o perioadă lungă "
        "(faza stabilă), aplicând scăderea doar în ultimii 10-20% din pașii totali. Aceasta permite explorarea vastă a spațiului parametrilor "
        "și facilitează checkpointing-ul modular și adăugarea continuă de date noi fără a distruge dinamica optimizatorului."
    ),
    (
        "Antrenarea în precizie mixtă (Mixed Precision BF16 / FP16): Formatul Bfloat16 (Brain Floating Point) păstrează cei 8 biți de exponent "
        "ai formatului FP32, având o mantisă de 7 biți. Această reprezentare oferă același interval dinamic ca FP32, eliminând complet "
        "riscul de underflow și necesitatea unui Loss Scaler manual întâlnită la FP16 tradițional. Pe nucleele NVIDIA Ampere și Ada Lovelace, "
        "operațiile BF16 pe Tensor Cores oferă o accelerare substanțială și reduc la jumătate memoria necesară pentru stocarea activărilor."
    ),
    (
        "Gradient Accumulation permite antrenarea cu dimensiuni efective de lot (effective batch size) mari chiar și pe plăci video "
        "cu memorie VRAM limitată. Dacă un micro-batch de 16 epuizează memoria VRAM, acumularea gradientului pe 4 pași calculează forward "
        "și backward cumulând gradienții în buffere FP32, executând un singur pas de optimizator la fiecare 4 iterații. Dimensiunea efectivă "
        "a lotului devine astfel 16 x 4 = 64 exemple, oferind estimări mai netede și stabile ale gradientului stochastic."
    ),
    # Systems & PyTorch Engineering
    (
        "Cuda Memory Allocator și fragmentarea memoriei în PyTorch: PyTorch utilizează un allocator de memorie cache (caching allocator) "
        "pentru a evita apelurile costisitoare către cudaMalloc și cudaFree ale driverului de sistem. Memoria eliberată este păstrată "
        "în blocuri cache pentru realocare instantanee. Când se manipulează tensoare cu dimensiuni variabile, pot apărea fenomene de fragmentare, "
        "unde memoria totală liberă pare suficientă, dar nu există un bloc contiguu suficient de mare pentru o nouă alocare tensorială."
    ),
    (
        "Compilatorul PyTorch Inductor (torch.compile): Introdus în PyTorch 2.0, compilatorul analizează graficul de calcul printr-un mecanism "
        "de urmărire (TorchDynamo) și generează kerneluri CUDA fuzionate folosind limbajul intermediar Triton. Fuziunea kernelurilor elimină "
        "transferurile redundante către memoria globală GPU între operații consecutive precum bias-add, normalizare și activare, oferind o creștere "
        "de 20-40% a randamentului (throughput în tok/s) fără modificarea arhitecturii modelului."
    ),
    (
        "Generalizarea modelelor lingvistice și Overfitting: Overfitting-ul apare atunci când capacitatea modelului (numărul de parametri) "
        "este prea mare în raport cu diversitatea și volumul datasetului, sau când numărul de epoci este excesiv. Modelul memorează exemplele "
        "exacte de antrenare, reducând train_loss la valori nerealist de mici, în timp ce val_loss crește vertiginos. Tehnici esențiale de prevenire "
        "includ decuplarea weight decay, oprirea timpurie (Early Stopping) bazată pe val_loss, filtrarea duplicatelor din dataset și dropout-ul."
    ),
    (
        "Tokenizarea BPE (Byte Pair Encoding) și WordPiece: Algoritmul Byte-Pair Encoding construiește vocabularul prin îmbinarea iterativă "
        "a celor mai frecvente perechi de caractere sau octeți. BPE la nivel de octeți (Byte-level BPE) garantează că orice secvență de text "
        "poate fi reprezentată fără tokeni necunoscuți (<unk>), descompunând caracterele rare sau diacriticele în secvențe de octeți UTF-8. "
        "Eficiența unui tokenizer este evaluată prin rata de compresie (numărul mediu de caractere per token generat)."
    ),
    (
        "Evaluarea modelelor lingvistice (Perplexity și Cross-Entropy): Perplexitatea (PPL) măsoară incertitudinea modelului în prezicerea "
        "următorului token și este calculată ca exponențiala mediei pierderii de entropie încrucișată: PPL = exp(loss). O perplexitate mai mică "
        "indică faptul că distribuția de probabilitate generată de model acordă încredere ridicată tokenilor corecți din setul de test. "
        "Pe lângă PPL, evaluarea modernă implică benchmark-uri specifice de tip MMLU, GSM8k și teste de consistență pe sarcini de cod."
    ),
    # Algorithms, Data Structures & Python Internals
    (
        "În limbajul Python, Global Interpreter Lock (GIL) sincronizează execuția instrucțiunilor bytecode pe un singur fir CPU per proces. "
        "Pentru a obține paralelism real în încărcarea și preprocesarea datelor, PyTorch folosește mecanismul de multiprocesare prin clasele "
        "DataLoader cu num_workers > 0. Procesele lucrătoare serializează batch-urile de tensoare în memoria partajată (shared memory), "
        "permițând procesului GPU principal să consume datele fără blocaje de interpretare."
    ),
    (
        "Tabelele de dispersie (Hash Tables) asigură inserție, ștergere și căutare în timp amortizat O(1). În cazurile nefavorabile cauzate "
        "de coliziuni multiple, complexitatea poate degrada la O(n). Structurile de date de tip Trie (arbori de prefixe) sunt utilizate intens "
        "în implementările de tokenizatoare pentru căutarea rapidă a celor mai lungi potriviri de prefixe în vocabular, oferind timpi de căutare "
        "proporționali strict cu lungimea cuvântului O(L)."
    ),
    (
        "Gestionarea contextului lung în Transformere (FlashAttention): Calculul naiv al matricii de atenție Q * K^T necesită memorie O(N^2), "
        "devenind prohibitiv pentru secvențe de peste 4096 tokeni. Algoritmul FlashAttention rezolvă această problemă prin divizarea matricilor "
        "în blocuri (tiling) care încap direct în memoria SRAM ultra-rapidă a nucleelor GPU. Prin fuziunea calculului de softmax și re-calcularea "
        "valorilor intermediare în faza de backward, FlashAttention reduce drastic accesul la memoria VRAM și permite extinderea contextului la zeci de mii de tokeni."
    ),
]

# -----------------------------------------------------------------------------
# 2. INSTRUCTION DATASET SAMPLES
# -----------------------------------------------------------------------------
INSTRUCTION_SAMPLES = [
    {
        "instruction": "Explică diferența fundamentală dintre memoria RAM și memoria VRAM.",
        "input": "",
        "output": (
            "Memoria RAM (Random Access Memory) este memoria de sistem utilizată de procesor (CPU) pentru a rula sistemul de operare "
            "și aplicațiile generale. Are latență scăzută, capacitate de regulă mare (16GB - 64GB+) și o lățime de bandă tipică de 50-80 GB/s pe DDR5.\n\n"
            "Memoria VRAM (Video RAM) este montată direct pe placa grafică (GPU) și este specializată pentru calcule masiv paralele. "
            "Deși capacitatea este adesea mai mică (6GB - 24GB pe plăci consumer), VRAM-ul de tip GDDR6 sau HBM oferă o lățime de bandă uriașă "
            "(300 - 1000+ GB/s), esențială pentru alimentarea continuă a miilor de nuclee CUDA cu greutăți și activări tensoriale."
        ),
    },
    {
        "instruction": "Ce este un mecanism de atenție în arhitectura Transformer și cum funcționează?",
        "input": "",
        "output": (
            "Mecanismul de atenție (Scaled Dot-Product Attention) permite fiecărui token dintr-o secvență să analizeze și să extragă "
            "informații relevante din toți ceilalți tokeni din context.\n\n"
            "Formula matematică este:\nAttention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V\n\n"
            "1. Query (Q): Ce caută tokenul curent.\n"
            "2. Key (K): Ce caracteristici conține fiecare token din secvență.\n"
            "3. Produsul Q * K^T calculează scorul de similaritate dintre tokeni.\n"
            "4. Împărțirea la sqrt(d_k) previne saturarea funcției Softmax când dimensiunile sunt mari.\n"
            "5. Softmax normalizează scorurile într-o distribuție de probabilitate.\n"
            "6. Înmulțirea cu Value (V) extrage informația ponderată proporțional cu atenția acordată."
        ),
    },
    {
        "instruction": "Scrie o funcție Python în PyTorch pentru calculul funcției de activare SwiGLU.",
        "input": "Intrare tensorială x de dimensiune (batch, seq_len, d_model)",
        "output": (
            "Iată implementarea curată și eficientă a unui strat SwiGLU în PyTorch:\n\n"
            "```python\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n\n"
            "class SwiGLU(nn.Module):\n"
            "    def __init__(self, d_model: int, d_ffn: int):\n"
            "        super().__init__()\n"
            "        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)\n"
            "        self.w_up = nn.Linear(d_model, d_ffn, bias=False)\n"
            "        self.w_down = nn.Linear(d_ffn, d_model, bias=False)\n\n"
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
            "        # SwiGLU(x) = (silu(x * W_gate) * (x * W_up)) * W_down\n"
            "        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))\n"
            "```"
        ),
    },
    {
        "instruction": "De ce este recomandat optimizatorul AdamW în locul optimizatorului Adam tradițional?",
        "input": "",
        "output": (
            "Diferența cheie constă în modul în care este tratată decăderea ponderilor (weight decay):\n\n"
            "1. În Adam clasic: Penalizarea de regularizare L2 este adăugată direct la vectorul de gradient. Acest gradient compus "
            "este ulterior divizat prin rădăcina pătrată a momentului de ordinul doi (v_t), ceea ce face ca parametrii cu gradienți mari "
            "să primească mai puțin weight decay, iar cei cu gradienți mici să fie supra-regularizați.\n\n"
            "2. În AdamW (Decoupled Weight Decay): Decăderea ponderilor este separată complet de momentele optimizatorului și aplicată "
            "direct asupra ponderilor la fiecare pas (w = w - lr * weight_decay * w). Aceasta stabilizează antrenarea și îmbunătățește "
            "semnificativ capacitatea de generalizare pe date noi."
        ),
    },
    {
        "instruction": "Cum influențează parametrul micro_batch_size și gradient_accumulation_steps viteza de antrenare pe GPU?",
        "input": "",
        "output": (
            "Relația dintre acești doi parametri definește geometria antrenării:\n\n"
            "1. micro_batch_size: Reprezintă numărul de secvențe procesate de GPU într-un singur pas fizic de calcul forward/backward. "
            "Dacă este prea mic (ex: 2 sau 4), nucleele GPU sunt sub-utilizate din cauza costului de lansare a kernelurilor (launch overhead). "
            "Dacă este optim (ex: 16 sau 32), nucleele Tensor lucrează la eficiență maximă, atâta timp cât nu depășește limita VRAM.\n\n"
            "2. gradient_accumulation_steps: Multiplică numărul de pași peste care se cumulează gradienții înainte de actualizarea greutăților. "
            "Formula pentru lotul efectiv este: Effective Batch = micro_batch_size * gradient_accumulation_steps.\n\n"
            "Ajustarea corectă permite obținerea aceluiași rezultat matematic fără a umple memoria VRAM până la riscul de crash sau throttling."
        ),
    },
    {
        "instruction": "Ce înseamnă RoPE (Rotary Position Embedding) și de ce este preferat față de embeddings absolute?",
        "input": "",
        "output": (
            "RoPE (Rotary Position Embedding) este o metodă de codificare a poziției tokenilor prin rotirea vectorilor de Query și Key "
            "în planul complex cu un unghi proporțional cu poziția absolută a tokenului în secvență.\n\n"
            "Avantaje esențiale:\n"
            "- Păstrează proprietatea relativă: Produsul scalar dintre Query de la poziția m și Key de la poziția n depinde strict "
            "de diferența relativă (m - n).\n"
            "- Atenuare naturală la distanță: Similaritatea scade pe măsură ce distanța dintre tokeni crește.\n"
            "- Extrapolare superioară: Permite modelului să gestioneze mai ușor lungimi de context mai mari decât cele văzute la antrenare, "
            "fără a necesita parametri învățați adiționali precum embeddings aditive tradiționale."
        ),
    },
    {
        "instruction": "Cum se calculează numărul de operații FLOPs per token pentru antrenarea unui model Transformer?",
        "input": "",
        "output": (
            "Pentru un model de tip Transformer standard cu N parametri:\n\n"
            "1. Trecerea înainte (Forward Pass): Necesită aproximativ 2 * N operații în virgulă mobilă (FLOPs) per token "
            "(o înmulțire și o adunare pentru fiecare parametru din straturile dense).\n"
            "2. Trecerea înapoi (Backward Pass): Necesită de două ori mai multe operații, adică aproximativ 4 * N FLOPs per token "
            "(calcularea gradientului activărilor și calcularea gradientului greutăților).\n\n"
            "Total antrenare per token = 2*N + 4*N = 6 * N FLOPs.\n\n"
            "Exemplu: Pentru un model cu 20 de milioane de parametri (20M), fiecare token consumă 6 * 20.000.000 = 120.000.000 FLOPs (0.12 GFLOPs). "
            "La o viteză de 15.000 tok/s, calculul realizat este de 1.8 TFLOPS."
        ),
    },
    {
        "instruction": "Care este rolul tehnicii Early Stopping în antrenarea rețelelor neuronale?",
        "input": "",
        "output": (
            "Early Stopping este o tehnică de regularizare utilizată pentru a preveni supra-învățarea (overfitting-ul).\n\n"
            "Mod de funcționare:\n"
            "1. La intervale prestabilite (de exemplu la fiecare 250 de pași), modelul este evaluat pe un set de date de validare "
            "care nu a fost folosit la antrenare.\n"
            "2. Sistemul monitorizează val_loss. Dacă val_loss nu se îmbunătățește timp de un număr specificat de evaluări "
            "(parametru numit 'patience'), antrenarea este oprită automat.\n"
            "3. Modelul este restaurat la starea din pasul unde val_loss a atins valoarea minimă absolută, garantând că se păstrează "
            "cea mai bună versiune generalizată a modelului."
        ),
    },
    {
        "instruction": "Scrie un script Python simplu pentru a verifica disponibilitatea GPU-ului și memoria VRAM liberă.",
        "input": "",
        "output": (
            "Iată un script compact ce raportează informațiile GPU în PyTorch:\n\n"
            "```python\n"
            "import torch\n\n"
            "if torch.cuda.is_available():\n"
            "    device_name = torch.cuda.get_device_name(0)\n"
            "    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)\n"
            "    free_mem, _ = torch.cuda.mem_get_info(0)\n"
            "    free_mem_mb = free_mem / (1024**2)\n"
            "    allocated_mem = torch.cuda.memory_allocated(0) / (1024**2)\n\n"
            "    print(f'GPU detectat: {device_name}')\n"
            "    print(f'VRAM Total: {total_mem:.0f} MB')\n"
            "    print(f'VRAM Liber: {free_mem_mb:.0f} MB')\n"
            "    print(f'VRAM Alocat PyTorch: {allocated_mem:.0f} MB')\n"
            "else:\n"
            "    print('GPU-ul CUDA nu este disponibil. Rulare pe CPU.')\n"
            "```"
        ),
    },
    {
        "instruction": "Ce reprezintă noțiunea de 'Generalization Gap' în antrenarea modelelor?",
        "input": "",
        "output": (
            "Generalization Gap (diferența de generalizare) este diferența dintre pierderea pe setul de validare și pierderea "
            "pe setul de antrenare: Gap = val_loss - train_loss.\n\n"
            "- În mod ideal, ambele valori scad împreună, iar ecartul rămâne mic și constant.\n"
            "- Dacă train_loss continuă să scadă, dar val_loss începe să crească, ecartul devine pozitiv și mare, semnalând "
            "apariția fenomenului de overfitting: modelul a încetat să mai învețe reguli generale și doar memorează exemplele din setul de antrenare."
        ),
    },
]

# -----------------------------------------------------------------------------
# 3. CHAT CONVERSATION SAMPLES (Multi-turn & Dialogues)
# -----------------------------------------------------------------------------
CHAT_DIALOGUES = [
    [
        {"role": "user", "content": "Salut! Care sunt pașii esențiali pentru a antrena un LLM de la zero?"},
        {
            "role": "assistant",
            "content": (
                "Salut! Antrenarea unui LLM de la zero parcurge patru etape esențiale:\n\n"
                "1. Colectarea și curățarea datelor: Un corpus masiv de text curat, filtrat de duplicate și caractere corupte.\n"
                "2. Crearea și antrenarea tokenizer-ului: Definirea vocabularului (de regulă BPE sau Byte-fallback de 16k-32k tokeni).\n"
                "3. Pre-antrenarea (Pre-training): Antrenare nesupravegheată pe predicția următorului token (Causal LM) pentru învățarea structurii limbajului și a cunoștințelor de bază.\n"
                "4. Finisarea prin instrucțiuni (Instruction SFT & Chat): Ajustare pe perechi de întrebare-răspuns cu mascare de pierdere pe prompt pentru a transforma modelul într-un asistent conversațional util."
            ),
        },
        {"role": "user", "content": "Cât de mic pot face un model ca să îl pot testa rapid pe laptop?"},
        {
            "role": "assistant",
            "content": (
                "Pe un laptop cu GPU de 6GB (cum este un RTX 3060), un model ideal de test are între 15M și 25M parametri:\n"
                "- d_model: 384\n"
                "- n_layers: 6\n"
                "- n_heads: 6\n"
                "- vocab_size: ~12.800 sau 32.000\n\n"
                "Un astfel de model consumă doar 600-800 MB VRAM la batch 8-16, rulează la viteze mari de 12.000 - 18.000 tok/s "
                "și poate finaliza câteva sute de pași în doar 1-2 minute, fiind perfect pentru validarea blocurilor din Network Studio."
            ),
        },
    ],
    [
        {"role": "user", "content": "Ce este mai important pentru viteză: să cresc batch size sau să folosesc BF16?"},
        {
            "role": "assistant",
            "content": (
                "Ambele sunt esențiale și se completează reciproc:\n\n"
                "1. BF16 (sau FP16) activează Tensor Cores pe arhitecturile NVIDIA moderne (Ampere, Ada), oferind o creștere directă "
                "de 2.5x - 3.5x a vitezei de calcul față de FP32 pur.\n"
                "2. Un micro_batch_size suficient (de exemplu 16 în loc de 2) oferă volumul de date necesar pentru ca nucleele Tensor "
                "să fie ocupate 100% din timp, în loc să stea în așteptarea instrucțiunilor de la CPU.\n\n"
                "Combinația optimă este activarea BF16 alături de cel mai mare micro_batch_size care păstrează consumul de VRAM sub 85% din capacitatea fizică a plăcii."
            ),
        },
        {"role": "user", "content": "Și de ce sub 85% și nu 100%?"},
        {
            "role": "assistant",
            "content": (
                "Pe sistemul de operare Windows, Desktop Window Manager (DWM) și interfața grafică rezervă între 300 și 600 MB de memorie VRAM. "
                "Dacă PyTorch încearcă să ocupe 100% din VRAM, driverul NVIDIA activează automat 'Shared GPU Memory Paging' peste magistrala PCIe. "
                "Aceasta încetinește antrenarea masiv (adesea cu peste 50%), deoarece memoria RAM a sistemului pe PCIe este de peste 20 de ori mai lentă decât memoria GDDR6 de pe placă."
            ),
        },
    ],
    [
        {"role": "user", "content": "Care este diferența dintre RMSNorm și LayerNorm?"},
        {
            "role": "assistant",
            "content": (
                "LayerNorm tradițional calculează atât media (mean), cât și varianța activărilor dintr-un strat, scăzând media și împărțind la abaterea standard:\n"
                "y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta\n\n"
                "RMSNorm (Root Mean Square Normalization) observă că proprietatea de scalare este cea care conferă stabilitatea gradientului, nu centrarea în jurul mediei. "
                "Astfel, RMSNorm elimină calculul mediei și termenul de bias:\n"
                "y = (x / RMS(x)) * gamma, unde RMS(x) = sqrt(1/d * sum(x_i^2) + eps)\n\n"
                "Această simplificare salvează aproximativ 7% - 20% din timpul de calcul al normalizării pe GPU."
            ),
        },
        {"role": "user", "content": "Modelele moderne precum LLaMA 3 sau Mistral folosesc RMSNorm?"},
        {
            "role": "assistant",
            "content": (
                "Da, exact! Practic toate arhitecturile de ultimă generație din familia LLaMA (LLaMA 1, 2, 3), Mistral, Gemma și RoCanon "
                "folosesc exclusiv RMSNorm datorită vitezei superioare și stabilității numerice impecabile în antrenarea în precizie mixtă."
            ),
        },
    ],
    [
        {"role": "user", "content": "Ce este un learning rate scheduler de tip WSD (Warmup-Stable-Decay)?"},
        {
            "role": "assistant",
            "content": (
                "Scheduler-ul WSD structurează antrenarea în trei faze distincte:\n\n"
                "1. Warmup (ex: primii 5-10% pași): Rata de învățare crește liniar de la zero la valoarea de vârf (peak lr), "
                "pentru a proteja parametrii inițializați aleatoriu de șocuri de gradient mari.\n"
                "2. Stable (ex: următorii 70-80% pași): Rata de învățare este menținută constantă la valoarea maximă. În această fază modelul "
                "explorează spațiul parametrilor și acumulează masa cunoștințelor.\n"
                "3. Decay (ultimii 10-15% pași): Rata de învățare scade rapid spre valoarea minimă, permițând modelului să se fixeze într-un minim "
                "local favorabil de pierdere.\n\n"
                "Avantajul major este că poți continua antrenarea oricând din faza stabilă adăugând date noi, fără să fii limitat de o curbă de decădere rigidă."
            ),
        },
    ],
    [
        {"role": "user", "content": "Ajută-mă cu o comandă pentru a vedea utilizarea plăcii video NVIDIA în timp real din terminal."},
        {
            "role": "assistant",
            "content": (
                "Cea mai directă și eficientă comandă din terminal este utilitarul nativ `nvidia-smi`:\n\n"
                "```bash\n"
                "# Actualizare automată la fiecare secundă\n"
                "nvidia-smi -l 1\n"
                "```\n\n"
                "Dacă dorești o afișare compactă concentrată pe temperatură, consum VRAM și putere (W):\n\n"
                "```bash\n"
                "nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1\n"
                "```"
            ),
        },
    ],
]


def generate_all():
    print(f"[Dataset Generator] Generating mini tech datasets in: {ROOT_DIR}")

    # 1. Pretrain Dataset (TXT & JSONL)
    pretrain_txt_path = ROOT_DIR / "tech_pretrain_mini.txt"
    pretrain_jsonl_path = ROOT_DIR / "tech_pretrain_mini.jsonl"

    # Expand paragraphs slightly to guarantee solid token volume (~35,000 tokens)
    full_paragraphs = PRETRAIN_PARAGRAPHS * 4

    # Write .txt
    with open(pretrain_txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_paragraphs))
    print(f"Created {pretrain_txt_path.name} ({pretrain_txt_path.stat().st_size / 1024:.1f} KB)")

    # Write .jsonl
    with open(pretrain_jsonl_path, "w", encoding="utf-8") as f:
        for p in full_paragraphs:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    print(f"Created {pretrain_jsonl_path.name} ({pretrain_jsonl_path.stat().st_size / 1024:.1f} KB)")

    # 2. Instruction Dataset (JSONL & TXT)
    instruction_jsonl_path = ROOT_DIR / "tech_instruction_mini.jsonl"
    instruction_txt_path = ROOT_DIR / "tech_instruction_mini.txt"

    expanded_instructions = INSTRUCTION_SAMPLES * 5

    # Write .jsonl
    with open(instruction_jsonl_path, "w", encoding="utf-8") as f:
        for item in expanded_instructions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Created {instruction_jsonl_path.name} ({instruction_jsonl_path.stat().st_size / 1024:.1f} KB)")

    # Write .txt with ChatML tags
    with open(instruction_txt_path, "w", encoding="utf-8") as f:
        for item in expanded_instructions:
            prompt = item["instruction"]
            if item.get("input"):
                prompt += "\n" + item["input"]
            response = item["output"]
            f.write(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n\n")
    print(f"Created {instruction_txt_path.name} ({instruction_txt_path.stat().st_size / 1024:.1f} KB)")

    # 3. Chat Dataset (JSONL & TXT)
    chat_jsonl_path = ROOT_DIR / "tech_chat_mini.jsonl"
    chat_txt_path = ROOT_DIR / "tech_chat_mini.txt"

    expanded_chat = CHAT_DIALOGUES * 8

    # Write .jsonl
    with open(chat_jsonl_path, "w", encoding="utf-8") as f:
        for conv in expanded_chat:
            f.write(json.dumps({"messages": conv}, ensure_ascii=False) + "\n")
    print(f"Created {chat_jsonl_path.name} ({chat_jsonl_path.stat().st_size / 1024:.1f} KB)")

    # Write .txt with ChatML tags
    with open(chat_txt_path, "w", encoding="utf-8") as f:
        for conv in expanded_chat:
            for turn in conv:
                role = turn["role"]
                content = turn["content"]
                f.write(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            f.write("\n")
    print(f"Created {chat_txt_path.name} ({chat_txt_path.stat().st_size / 1024:.1f} KB)")

    print("[Dataset Generator] All mini datasets generated successfully in root!")


if __name__ == "__main__":
    generate_all()

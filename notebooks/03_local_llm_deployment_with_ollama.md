# Notebook 03: Local LLM Deployment with Ollama

**CABS AI Productivity Series**
Workshop: _LangChain, LangGraph & Local LLM Deployment: Building AI Agent Systems That Keep Your Data Safe_

---

## Why Run a Model Locally?

In Notebooks 01 and 02, every question you asked and every piece of data you
sent was processed on Google's servers via the Gemini API. For public literature
and public gene names, this is fine. But consider these scenarios:

- You are analyzing unpublished gene expression data from a clinical trial
- You are querying proprietary compound structures before a patent filing
- Your institutional data use agreement prohibits sending data to third-party servers
- You simply want to use an LLM on an airplane with no internet

**Local deployment means the LLM runs on your own machine.** Your prompts, your
data, and the model's reasoning never leave your computer. The tradeoff is that
local models are smaller and less capable than cloud models like GPT-4 or Gemini.

This guide walks you through:

1. Installing Ollama on a Mac
2. Understanding model sizes and hardware limits
3. Running a local model from the command line
4. Calling the local model from Python (same pattern as Notebooks 01 and 02)
5. Running the gene lookup agent from Notebook 02 — but fully local

---

## Part 1: Install Ollama on Mac

### Step 1: Download Ollama

Go to [ollama.com](https://ollama.com) and click **Download for macOS**.

This downloads a .zip file. Unzip it and drag the Ollama app into your
Applications folder — same as installing any other Mac app.

### Step 2: Launch Ollama

Open the Ollama app from Applications. You will see a small llama icon
appear in your menu bar (top right of your screen). That means Ollama
is running as a background service.

You can now use it from the Terminal.

### Step 3: Pull your first model

Open Terminal and run:

    ollama pull llama3.2

This downloads the Llama 3.2 model (about 2 GB for the 3B version).
It may take a few minutes depending on your internet speed. You only
need to do this once — the model is saved locally.

### Step 4: Test it

Run the model interactively:

    ollama run llama3.2

You will see a prompt where you can type. Try:

    >>> What does the TP53 gene do?

The model will respond. Press Ctrl+D to exit.

### Step 5: Verify what you have installed

    ollama list

Expected output (your list may differ):

    NAME                       ID              SIZE      MODIFIED
    llama3.2:latest            a80c4f17acd5    2.0 GB    just now

## That's it. You now have a working local LLM.

## Part 2: Model Sizes, Hardware Limits, and Speed

This is the most important section for making practical decisions about
local deployment. The key constraint is **RAM** (memory), not CPU or disk.

### How model size works

Model names include a parameter count: 1B, 3B, 7B, 8B, 14B, 70B. This is
the number of parameters (weights) in the neural network. More parameters
generally means smarter, but also bigger and slower.

When you download a model through Ollama, it comes **quantized** — the
weights are compressed from 16-bit floats to 4-bit integers. This is
why a "7B" model doesn't actually take 14 GB of RAM.

### The rule of thumb

**A quantized model needs roughly 1 GB of RAM per 1 billion parameters,
plus some overhead.**

| Model size | RAM needed (approx) | Example models                                     |
| ---------- | ------------------- | -------------------------------------------------- |
| 1-3B       | 2-4 GB              | llama3.2:1b, llama3.2:3b, gemma2:2b                |
| 7-8B       | 5-8 GB              | llama3.1:8b, mistral:7b, gemma2:9b                 |
| 13-14B     | 10-14 GB            | qwen2.5:14b, codellama:13b                         |
| 32-34B     | 20-24 GB            | qwen2.5:32b, codellama:34b                         |
| 70B        | 40-48 GB            | llama3.1:70b (needs Mac Studio / high-end desktop) |

### What does this mean for your Mac Mini?

Check your RAM: Apple menu → About This Mac → Memory.

| Your Mac Mini RAM | What you can run                                | Practical speed          |
| ----------------- | ----------------------------------------------- | ------------------------ |
| 8 GB              | Up to 3B models comfortably, 7-8B models slowly | 8-15 tokens/sec for 3B   |
| 16 GB             | 7-8B models well, 14B models possible           | 15-25 tokens/sec for 8B  |
| 24 GB             | 14B models comfortably                          | 12-18 tokens/sec for 14B |
| 32 GB+            | 32B models, maybe 70B (tight)                   | Varies                   |

**Apple Silicon (M1/M2/M4) is very efficient** for local inference because
the CPU and GPU share the same unified memory. Ollama automatically uses
the GPU (Metal acceleration) — you don't need to configure anything.

### Speed: what to expect

"Tokens per second" is how fast the model generates text. A token is
roughly 3/4 of a word.

- **25+ tokens/sec** = feels like a fast typist, responsive
- **10-20 tokens/sec** = usable, slight delay
- **5-10 tokens/sec** = noticeably slow, but works
- **< 5 tokens/sec** = painful for interactive use

If the model doesn't fit in RAM, your Mac will use disk swap, and
speed drops to < 1 token/sec. At that point, use a smaller model.

### Quality vs. size tradeoff

This is a real tradeoff, not a marketing difference:

- **3B models**: Can follow simple instructions, summarize short text,
  answer basic questions. Often makes mistakes on complex reasoning.
  Tool-calling (agent behavior) is unreliable.
- **7-8B models**: Decent general capability. Can do tool-calling if
  the prompt is well-structured. Good for RAG and simple agents.
  This is the sweet spot for most local use cases.
- **14B+ models**: Noticeably better reasoning and instruction-following.
  More reliable tool-calling. Worth it if your hardware supports it.
- **70B+ models**: Approaching cloud model quality. Requires high-end
  hardware (64 GB+ RAM).

**For this workshop, we recommend llama3.1:8b** as the default. If your
machine only has 8 GB RAM, use llama3.2:3b instead.

---

## Part 3: Useful Ollama Commands

Here are the commands you'll use most often.

### Pull (download) a model

    ollama pull llama3.1:8b

### List installed models

    ollama list

### Run a model interactively (chat mode)

    ollama run llama3.1:8b

Then type your prompt and press Enter. Ctrl+D to exit.

### Run a one-off prompt (no interactive session)

    ollama run llama3.1:8b "What does BRCA1 do? Answer in 2 sentences."

Expected output (will vary — the model is probabilistic):

    BRCA1 is a tumor suppressor gene that plays a critical role in DNA
    repair, specifically in the homologous recombination pathway. Mutations
    in BRCA1 are strongly associated with increased risk of breast and
    ovarian cancer.

### Check if Ollama is running

    curl http://localhost:11434

Expected output:

    Ollama is running

### See which models are currently loaded in memory

    ollama ps

### Remove a model you no longer need (frees disk space)

    ollama rm llama3.2:latest

---

## Part 4: Calling Ollama from Python

Ollama runs a local API server at http://localhost:11434. You can call it
from Python the same way you call any API. LangChain has a built-in
integration.

The key point: **switching from a cloud model to a local model is a
one-line change** in your code.

Before running the code below, make sure:

1. Ollama is running (check the llama icon in menu bar)
2. You have pulled a model: `ollama pull llama3.1:8b`

### Install the LangChain Ollama integration

If you're running this locally (not in Colab), open Terminal:

    pip install langchain-ollama langchain-core requests

---

## Part 5: Python Script — Local Gene Lookup Agent

Below is a complete Python script that replicates the agent from
Notebook 02, but runs entirely on your machine. No API key. No
cloud. Your queries stay local.

The only things that go over the internet are the UniProt and
AlphaFold API calls — which send a public gene name, not your
private data. The LLM reasoning happens locally.

Save this as `local_agent.py` and run it with `python local_agent.py`,
or copy the cells into a local Jupyter notebook.

# ============================================================

    # LOCAL GENE LOOKUP AGENT — runs on your machine with Ollama
    # ============================================================
    # This script does the same thing as Notebook 02, but uses a
    # local LLM instead of Google Gemini. No API key needed.
    #
    # Requirements:
    #   - Ollama installed and running
    #   - A model pulled: ollama pull llama3.1:8b
    #   - Python packages: pip install langchain-ollama langchain-core requests langgraph
    #
    # What goes where:
    #   - Your question → stays on your machine (processed by local LLM)
    #   - LLM reasoning → stays on your machine
    #   - Gene name → sent to UniProt/AlphaFold public APIs (public info only)
    #   - API results → come back to your machine → processed by local LLM
    # ============================================================

    import requests
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent


    # ============================================================
    # STEP 1: Connect to your local Ollama model
    # ============================================================
    # This is the ONE LINE that's different from Notebook 02.
    # Instead of:
    #   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # We use:
    #   llm = ChatOllama(model="llama3.1:8b")
    #
    # Everything else — tools, agent, prompts — stays exactly the same.

    llm = ChatOllama(model="llama3.1:8b")

    # Quick test
    response = llm.invoke("Say hello in one sentence.")
    print("Local LLM says:", response.content)
    print("✅ Local model is working!\n")


    # ============================================================
    # STEP 2: Define the tools (identical to Notebook 02)
    # ============================================================
    # These tools call public APIs — the gene names are public info.
    # Your private data (your question, your analysis) stays local.

    @tool
    def lookup_gene_uniprot(gene_name: str) -> str:
        """Look up a gene/protein in the UniProt database.
        Use this tool when the user asks about gene function, protein
        information, disease associations, or subcellular location.
        Input should be a standard gene symbol like TP53, BRCA1, EGFR.
        Returns protein name, function, subcellular location, and
        disease involvement."""

        search_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"(gene:{gene_name}) AND (organism_id:9606) AND (reviewed:true)",
            "format": "json",
            "size": 1,
            "fields": "accession,gene_names,protein_name,cc_function,cc_subcellular_location,cc_disease"
        }

        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            return f"Error: UniProt API returned status {response.status_code}"

        data = response.json()
        results = data.get("results", [])

        if not results:
            return f"No results found for gene '{gene_name}' in UniProt."

        entry = results[0]
        accession = entry.get("primaryAccession", "N/A")

        protein_name = "N/A"
        prot_desc = entry.get("proteinDescription", {})
        rec_name = prot_desc.get("recommendedName", {})
        if rec_name:
            protein_name = rec_name.get("fullName", {}).get("value", "N/A")

        gene_names = [g.get("geneName", {}).get("value", "")
                      for g in entry.get("genes", [])]

        function_texts = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "FUNCTION":
                for text in comment.get("texts", []):
                    function_texts.append(text.get("value", ""))

        location_texts = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "SUBCELLULAR LOCATION":
                for loc in comment.get("subcellularLocations", []):
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        location_texts.append(loc_val)

        disease_texts = []
        for comment in entry.get("comments", []):
            if comment.get("commentType") == "DISEASE":
                disease = comment.get("disease", {})
                if disease:
                    disease_texts.append(disease.get("diseaseId", ""))

        output = f"""UniProt Entry: {accession}
    Gene: {', '.join(gene_names) if gene_names else 'N/A'}
    Protein: {protein_name}
    Function: {' '.join(function_texts) if function_texts else 'No function annotation available.'}
    Subcellular Location: {', '.join(location_texts) if location_texts else 'Not annotated.'}
    Disease Associations: {', '.join(disease_texts) if disease_texts else 'None annotated.'}
    UniProt URL: https://www.uniprot.org/uniprot/{accession}"""

        return output


    @tool
    def lookup_alphafold(uniprot_id: str) -> str:
        """Look up a protein's predicted 3D structure in the AlphaFold database.
        Use this tool when the user asks about protein structure, 3D structure,
        folding, or structural predictions.
        Input should be a UniProt accession ID like P04637 or Q9Y6K9.
        If you don't have the UniProt ID, use lookup_gene_uniprot first to find it.
        Returns structure prediction confidence and a link to view the 3D model."""

        url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        response = requests.get(url)

        if response.status_code == 404:
            return f"No AlphaFold prediction found for UniProt ID '{uniprot_id}'."
        if response.status_code != 200:
            return f"Error: AlphaFold API returned status {response.status_code}"

        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
        else:
            return f"Unexpected response format from AlphaFold."

        entry_id = entry.get("entryId", "N/A")
        gene = entry.get("gene", "N/A")
        organism = entry.get("organismScientificName", "N/A")
        confidence = entry.get("globalMetricValue", "N/A")
        pdb_url = entry.get("pdbUrl", "N/A")

        output = f"""AlphaFold Prediction: {entry_id}
    Gene: {gene}
    Organism: {organism}
    Global Confidence (pLDDT): {confidence}
    View 3D Structure: https://alphafold.ebi.ac.uk/entry/{uniprot_id}
    Download PDB: {pdb_url}
    Note: pLDDT > 90 = high confidence, 70-90 = good, 50-70 = low, < 50 = unreliable."""

        return output


    # ============================================================
    # STEP 3: Create the agent (identical pattern to Notebook 02)
    # ============================================================

    tools = [lookup_gene_uniprot, lookup_alphafold]
    agent = create_react_agent(model=llm, tools=tools)

    print("✅ Local agent created with 2 tools")
    print("   Model: llama3.1:8b (running on your machine)")
    print("   Tools: UniProt lookup, AlphaFold lookup\n")


    # ============================================================
    # STEP 4: Helper function to run the agent and show its reasoning
    # ============================================================

    def ask_agent(question):
        print(f"❓ Question: {question}")
        print("=" * 60)

        inputs = {"messages": [{"role": "user", "content": question}]}

        for step in agent.stream(inputs, stream_mode="values"):
            messages = step["messages"]
            last_msg = messages[-1]

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"\n🔧 Calling tool: {tc['name']}")
                    print(f"   Input: {tc['args']}")

            elif last_msg.type == "tool":
                content = last_msg.content
                if len(content) > 500:
                    content = content[:500] + "..."
                print(f"\n📋 Tool result (truncated):\n{content}")

        final = step["messages"][-1].content
        print(f"\n{'=' * 60}")
        print(f"💬 Final Answer:\n{final}")


    # ============================================================
    # STEP 5: Test the local agent
    # ============================================================

    print("\n--- Test 1: Gene function (should call UniProt) ---\n")
    ask_agent("What does the BRCA1 gene do?")

    print("\n\n--- Test 2: Gene + structure (should chain both tools) ---\n")
    ask_agent("Tell me about TP53 — its function and AlphaFold structure confidence.")

---

## Part 6: Side-by-Side Comparison — Cloud vs. Local

Run the same question through both models and compare:

|                           | Cloud (Gemini, Notebook 02) | Local (Ollama, this guide)      |
| ------------------------- | --------------------------- | ------------------------------- |
| Model                     | gemini-2.5-flash            | llama3.1:8b                     |
| Where your prompt goes    | Google's servers            | Your machine                    |
| API key needed            | Yes                         | No                              |
| Internet needed for LLM   | Yes                         | No                              |
| Internet needed for tools | Yes (UniProt, AlphaFold)    | Yes (UniProt, AlphaFold)        |
| Response quality          | Higher — larger model       | Lower — smaller model           |
| Speed                     | Fast (cloud GPU)            | Depends on your hardware        |
| Cost                      | Free tier has rate limits   | Free forever (your electricity) |
| Code change needed        | -                           | One line: swap the LLM object   |

The architecture is identical. The only difference is where the LLM runs.

### The one-line swap

Cloud version (Notebook 02):

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

Local version (this guide):

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1:8b")

Everything downstream — tools, agent, prompts — is unchanged. This is
the power of LangChain's abstraction: you can swap the brain without
rewiring the body.

---

## Part 7: Data Flow — What Goes Where

This is the most important diagram of the workshop. When your agent
runs locally, here is exactly what stays on your machine and what
goes over the internet:

STAYS ON YOUR MACHINE (private):
├── Your question / prompt
├── LLM reasoning ("I should call UniProt for this gene")
├── LLM's final answer generation
└── Any private context you add to the prompt

GOES OVER THE INTERNET (public):
├── Gene name → UniProt API (e.g., "TP53")
├── UniProt accession → AlphaFold API (e.g., "P04637")
└── API responses come back to your machine

The gene names sent to UniProt and AlphaFold are public identifiers —
the same as typing them into a web browser. But your reasoning, your
hypotheses, and your analysis context never leave your computer.

If you need even the gene names to stay private (e.g., proprietary
engineered sequences), you could replace the API tools with a local
database. The architecture supports this — just swap the tool function.

---

## Part 8: Practical Tips and Troubleshooting

### "ollama: command not found"

Make sure the Ollama app is open (check for the llama icon in the
menu bar). The CLI commands only work when the app is running.

### Model is very slow

Your model might not fit in RAM. Check with `ollama ps` to see
memory usage. If the model is swapping to disk, use a smaller one:

    ollama run llama3.2:3b

### Agent doesn't call the tools correctly

Smaller local models are less reliable at tool-calling than cloud
models. This is expected. Mitigations:

- Use llama3.1:8b or larger — the 3B models struggle with tool-calling
- Keep tool docstrings simple and specific
- If tool-calling fails consistently, this is a real limitation of
  small local models. Be honest about it in your work.

### Which model should I use?

| Your situation                 | Recommended model |
| ------------------------------ | ----------------- |
| 8 GB RAM, just want to try it  | llama3.2:3b       |
| 16 GB RAM, general use         | llama3.1:8b       |
| 16 GB RAM, need tool-calling   | llama3.1:8b       |
| 24+ GB RAM, best local quality | qwen2.5:14b       |
| Coding tasks specifically      | qwen2.5-coder:14b |

### Can I use Ollama for the RAG workflow from Notebook 01?

Yes. You would also need local embeddings. Ollama supports embedding
models:

    ollama pull nomic-embed-text

Then in Python, swap GoogleGenerativeAIEmbeddings for OllamaEmbeddings.
This gives you a fully offline RAG pipeline — no internet needed at all.

---

## Summary: What You Learned in This Workshop

Across three notebooks, you built the same kind of AI systems that
commercial "AI for biology" tools use internally:

**Notebook 01 (RAG):** LLM reads your documents and answers questions.
Pattern: retrieve → read → answer.

**Notebook 02 (Agent):** LLM decides which tools to call and chains
them together. Pattern: think → act → observe → repeat.

**Notebook 03 (Local):** Same agent, but the LLM runs on your machine.
Your data stays private. Pattern: identical, just swap the model.

### The three decisions you now know how to make:

1. **Do I need RAG or an agent?**
   RAG = you have documents to search. Agent = you need to call
   external tools or APIs. You can combine both.

2. **Cloud or local?**
   Cloud = better quality, needs internet and API key, data goes to
   a third party. Local = weaker but private, no cost, no internet
   needed for LLM.

3. **Which model size?**
   Depends on your RAM, your quality needs, and whether you need
   reliable tool-calling. 8B is the practical sweet spot for local.

### Next steps:

- Browse models at [ollama.com/library](https://ollama.com/library)
- Try adding your own tools (PubMed search, BLAST, pathway lookup)
- Look into Open WebUI for a ChatGPT-like interface on top of Ollama
- Read the LangChain docs: [python.langchain.com](https://python.langchain.com)

Full workshop materials:
👉 [github.com/lecaibio/cabs-workshop-llm-agents](https://github.com/lecaibio/cabs-workshop-llm-agents)

---

## Part 9: Fully Offline Deployment — When You Can't Even pip install

Some environments are completely air-gapped: no internet, no pip install,
no pulling models from Ollama's registry. This is common in clinical
research settings, pharmaceutical companies, and government labs where
machines handling sensitive data are physically disconnected from the
internet.

In these cases, you prepare everything on a connected machine, verify
it works, then physically transport the packaged environment to the
offline machine through approved internal transfer channels (secure
file transfer, USB with clearance, internal network share).

### The tool: conda-pack

conda-pack takes an entire conda environment — Python, all installed
packages, all dependencies — and compresses it into a single .tar.gz
file that can be unpacked and used on another machine without any
installation step.

### Step 1: Build and test the environment on a connected machine

    # Create a clean conda environment
    conda create -n llm-agent python=3.11 -y
    conda activate llm-agent

    # Install everything the scripts need
    pip install langchain langchain-ollama langchain-core langgraph requests

    # Run your script, confirm everything works
    python local_agent.py

### Step 2: Pack the environment

    # Install conda-pack (only needed on the connected machine)
    conda install conda-pack -y

    # Pack the environment into a single file
    conda pack -n llm-agent -o llm-agent-env.tar.gz

This produces a single file (typically 500 MB - 1 GB) containing the
entire Python environment.

### Step 3: Transfer the Ollama model file

Ollama stores downloaded models in ~/.ollama/models/. You need to
copy this directory as well. For llama3.1:8b, this is about 4-5 GB.

Also download the Ollama installer itself (.dmg for Mac) from
ollama.com while you have internet access.

Your transfer package is:

    llm-agent-env.tar.gz       (~500 MB - 1 GB)
    ollama-darwin.dmg           (~100 MB)
    ~/.ollama/models/           (~4-5 GB for one 8B model)
    local_agent.py              (your script)

Transfer these via whatever secure channel your institution approves.

### Step 4: Unpack on the offline machine

    # Install Ollama from the .dmg (normal Mac app install)

    # Copy the models directory into place
    cp -r models/ ~/.ollama/models/

    # Verify Ollama sees the model
    ollama list

    # Unpack the conda environment
    mkdir -p ~/envs/llm-agent
    tar -xzf llm-agent-env.tar.gz -C ~/envs/llm-agent

    # Activate it (no conda needed on target machine)
    source ~/envs/llm-agent/bin/activate

    # Fix path prefixes (required by conda-pack)
    conda-unpack

    # Run your script
    python local_agent.py

### Important notes

- The connected machine and the offline machine must be the same OS
  and architecture (e.g., both macOS ARM64). conda-pack is not
  cross-platform.
- Test thoroughly on the connected machine before transferring.
  Debugging on an air-gapped machine is painful.
- If your offline machine doesn't have conda installed, that's fine —
  conda-pack environments are self-contained. You just need to source
  the activate script.
- For the agent script: remember that UniProt and AlphaFold API calls
  also need internet. On an air-gapped machine, you would need to
  replace those tools with a local database or pre-downloaded data
  files. The agent architecture stays the same — you just swap the
  tool functions.

---

## A Note on Troubleshooting: Use AI to Learn AI

If you run into errors following this guide — a package version conflict,
a model that won't load, a tool-calling format your local model doesn't
understand — that is normal. Local environments vary: different OS versions,
different Python versions, different RAM, different chip architectures. A
workflow that runs perfectly on one machine may need small adjustments on
another.

When this happens, copy the error message and ask an AI (ChatGPT, Claude,
Gemini — whichever you have access to). Describe what you were trying to
do, paste the full traceback, and ask it to explain what went wrong and
how to fix it. This is one of the highest-value uses of LLMs right now:
not generating code from scratch, but explaining why existing code broke
and what to do about it.

This is also a good learning habit in general. When the AI gives you a
fix, don't just apply it — ask it to explain _why_ it works. "Why did
changing this import path fix the error?" or "What does this version
conflict mean?" These follow-up questions are where real understanding
builds up.

Use AI to learn AI. The tools you built in this workshop are the same
tools that can help you debug and improve them.

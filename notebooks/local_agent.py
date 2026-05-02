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

# ============================================================
# STEP 6: work around with local ollama's limitation
# ============================================================

# ask_agent("What is the UniProt accession ID for TP53?")
# # read the answer: P04637
# ask_agent("Look up the AlphaFold structure for P04637")
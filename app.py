import io
import json
import textwrap
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

st.set_page_config(
    page_title="MoleculeLab | Chemoinformatics Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------- Design system ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172033; --muted:#667085; --line:#e6eaf0; --teal:#0f766e; --mint:#dff7ef; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7fbfa 0%,#f7f8fc 100%); color:var(--ink); }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1400px; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
.hero { padding:2rem 2.2rem; border-radius:24px; background:linear-gradient(120deg,#102a43,#0f766e); color:white; margin-bottom:1.3rem; box-shadow:0 18px 45px rgba(16,42,67,.14); }
.hero h1 { color:white; font-size:2.45rem; margin:.2rem 0 .5rem 0; }
.hero p { color:#d7f5ee; font-size:1.02rem; margin:0; }
.eyebrow { text-transform:uppercase; letter-spacing:.15em; font-size:.72rem; font-weight:700; color:#a7f3d0; }
.panel { background:white; border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.25rem; box-shadow:0 5px 20px rgba(16,24,40,.035); }
.kpi { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
.kpi-label { color:#667085; font-size:.8rem; font-weight:600; }
.kpi-value { color:#101828; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.55rem; margin-top:.25rem; }
.small-muted { color:#667085; font-size:.88rem; }
.pill { display:inline-block; background:#dff7ef; color:#0f766e; padding:.25rem .65rem; border-radius:999px; font-size:.75rem; font-weight:700; }
div[data-testid="stSidebar"] { background:#101d2b; }
div[data-testid="stSidebar"] * { color:#edf7f5; }
div[data-testid="stSidebar"] .stRadio label { padding:.18rem 0; }
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:700; border:1px solid #c9d7d5; }
.stButton>button[kind="primary"] { background:#0f766e; border-color:#0f766e; }
hr { border-color:#e6eaf0; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Curated teaching dataset: SMILES are included locally, so core features work without an API.
MOLECULES = [
    {"name":"Paracetamol", "category":"Analgesic", "smiles":"CC(=O)NC1=CC=C(O)C=C1O", "use":"Pain and fever relief", "note":"An acetanilide derivative used as an analgesic and antipyretic."},
    {"name":"Ibuprofen", "category":"NSAID", "smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "use":"Pain and inflammation", "note":"A propionic-acid NSAID; its stereochemistry can affect pharmacology."},
    {"name":"Aspirin", "category":"NSAID", "smiles":"CC(=O)OC1=CC=CC=C1C(=O)O", "use":"Pain relief; antiplatelet use", "note":"Acetylsalicylic acid contains an ester and a carboxylic acid."},
    {"name":"Naproxen", "category":"NSAID", "smiles":"COC1=CC2=CC=C(C=C2C=C1)C(C)C(=O)O", "use":"Pain and inflammation", "note":"A propionic-acid NSAID with a methoxy-substituted aromatic system."},
    {"name":"Metformin", "category":"Antidiabetic", "smiles":"CN(C)C(=N)NC(=N)N", "use":"Type 2 diabetes management", "note":"A biguanide with high polarity and multiple nitrogen atoms."},
    {"name":"Amoxicillin", "category":"Antibiotic", "smiles":"CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](C3=CC=C(O)C=C3)N)C(=O)N2[C@H]1C(=O)O", "use":"Bacterial infections", "note":"A penicillin-class antibiotic; beta-lactam integrity is important to activity."},
    {"name":"Ciprofloxacin", "category":"Antibiotic", "smiles":"O=C(O)C1=CN(C2CC2)C2=CC(N3CCNCC3)=CC=C2C1=O", "use":"Selected bacterial infections", "note":"A fluoroquinolone scaffold; use is guided by susceptibility and clinical advice."},
    {"name":"Doxycycline", "category":"Antibiotic", "smiles":"CN(C)C1C(O)=C(C2=CC=CC=C2)C(=O)C2=C(O)C3=C(C(=O)C(C(N)=O)=C3O)C1=O", "use":"Selected bacterial infections", "note":"A tetracycline-class molecule with multiple oxygen-containing groups."},
    {"name":"Cetirizine", "category":"Antihistamine", "smiles":"O=C(O)COCCN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1", "use":"Allergy symptoms", "note":"A piperazine-containing antihistamine with a carboxylic acid group."},
    {"name":"Loratadine", "category":"Antihistamine", "smiles":"CCOC(=O)N1CCC(=C2C3=CC=CC=C3CC3=CC=CC=C23)CC1", "use":"Allergy symptoms", "note":"A tricyclic antihistamine with an ester and a carbamate."},
    {"name":"Omeprazole", "category":"Proton-pump inhibitor", "smiles":"COC1=CC2=C(C=C1)N=CN=C2S(=O)CC3=NC4=CC(OC)=C(OC)C=C4N3", "use":"Acid-related disorders", "note":"A substituted benzimidazole sulfoxide; chemical stability depends on conditions."},
    {"name":"Famotidine", "category":"H2-receptor antagonist", "smiles":"NC(=N)NS(=O)(=O)CC1=CSC(N)=N1", "use":"Acid-related disorders", "note":"A sulfur-rich molecule with several nitrogen-containing functional groups."},
    {"name":"Acyclovir", "category":"Antiviral", "smiles":"Nc1nc(=O)c2ncn(COCCO)c2n1", "use":"Selected herpesvirus infections", "note":"A guanine analogue with a flexible acyclic side chain."},
    {"name":"Fluconazole", "category":"Antifungal", "smiles":"OC(CN1C=NC=N1)(CN1C=NC=N1)C1=CC=C(F)C=C1F", "use":"Selected fungal infections", "note":"Contains two triazole rings and a difluorophenyl group."},
    {"name":"Amlodipine", "category":"Cardiovascular", "smiles":"CCOC(=O)C1=C(COCCN)NC(C)=C(C1C2=CC=CC=C2Cl)C(=O)OC", "use":"Hypertension and angina", "note":"A dihydropyridine calcium-channel blocker scaffold."},
    {"name":"Losartan", "category":"Cardiovascular", "smiles":"CCCC1=NC(=C(N1CC1=CC=C(C=C1)C2=CC=CC=C2C3=NN=NN3)Cl)CO", "use":"Hypertension", "note":"An angiotensin II receptor blocker with a tetrazole group."},
    {"name":"Atorvastatin", "category":"Lipid-lowering", "smiles":"CC(C)C1=C(C(=C(N1CC[C@@H](O)C[C@@H](O)CC(=O)O)C2=CC=CC=C2)C3=CC=CC=C3)C4=CC=CC=C4", "use":"Cholesterol management", "note":"A statin scaffold containing a hydroxy-acid side chain."},
    {"name":"Lidocaine", "category":"Local anesthetic", "smiles":"CCN(CC)CC(=O)NC1=C(C)C=CC=C1C", "use":"Local anaesthesia", "note":"An amide local anaesthetic with a tertiary amine."},
    {"name":"Caffeine", "category":"Other bioactive", "smiles":"Cn1c(=O)c2c(ncn2C)n(C)c1=O", "use":"Stimulant found in common beverages", "note":"A methylxanthine containing fused nitrogen heterocycles."},
    {"name":"Nicotine", "category":"Other bioactive", "smiles":"CN1CCC[C@H]1C1=CC=CN=C1", "use":"Research example", "note":"An alkaloid with pyrrolidine and pyridine rings; addictive and toxic."},
    {"name":"Vitamin C (Ascorbic acid)", "category":"Vitamin", "smiles":"O=C1OC(C(O)CO)C(O)=C1O", "use":"Nutrient", "note":"A water-soluble vitamin with an enediol-containing lactone structure."},
    {"name":"Fluoxetine", "category":"Antidepressant", "smiles":"CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1", "use":"Selected mental-health conditions", "note":"A selective serotonin reuptake inhibitor; included for structure learning only."},
    {"name":"Melatonin", "category":"Hormone-related", "smiles":"COc1ccc2[nH]cc(CCNC(C)=O)c2c1", "use":"Sleep-wake biology research", "note":"An indole derivative with an acetamide side chain."},
    {"name":"Sulfamethoxazole", "category":"Antibiotic", "smiles":"Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", "use":"Selected bacterial infections", "note":"A sulfonamide antibiotic often studied in combination therapy."},
    {"name":"Hydroxychloroquine", "category":"Antimalarial / immunomodulatory", "smiles":"CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12", "use":"Specific clinician-directed indications", "note":"A quinoline derivative; included for chemical descriptor comparison."},
    {"name":"Benzocaine", "category":"Local anesthetic", "smiles":"CCOC(=O)c1ccc(N)cc1", "use":"Topical local anaesthesia", "note":"An aromatic ester with a para-amino substituent."},
    {"name":"Warfarin", "category":"Anticoagulant", "smiles":"CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "use":"Anticoagulation under monitoring", "note":"A coumarin derivative; clinical use requires careful monitoring."},
    {"name":"Prednisone", "category":"Corticosteroid", "smiles":"C[C@]12CC[C@H]3[C@@H]([C@H]1CC[C@@]2(C(=O)CO)O)CCC4=CC(=O)C=C[C@]34C", "use":"Inflammatory and immune conditions", "note":"A steroid scaffold with multiple carbonyl and hydroxyl-related features."},
    {"name":"Montelukast", "category":"Respiratory", "smiles":"CC(C)(C)SCC1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC(=CC=C3)CC(O)C(=O)O", "use":"Asthma/allergy management", "note":"A lipophilic molecule containing a carboxylic acid and sulfur-containing group."},
    {"name":"Donepezil", "category":"Neurology", "smiles":"COC1=CC2=C(C=C1)C(=O)C(C3=CC=CC=C3)C2CCN4CCCCC4", "use":"Selected dementia symptoms", "note":"A piperidine-containing compound with aromatic and carbonyl features."},
]

THEORY = {
    "SMILES notation": "SMILES (Simplified Molecular Input Line Entry System) encodes a molecular graph as a text string. Atoms are represented by symbols, branches by parentheses, ring closures by matching digits, and bonds by symbols or implied defaults.",
    "Molecular descriptors": "Descriptors are numerical or categorical features calculated from a molecular structure. Examples include molecular weight, topological polar surface area (TPSA), hydrogen-bond donors/acceptors, rotatable bonds and calculated logP. They summarize selected structural properties; they do not by themselves establish safety or efficacy.",
    "Lipinski's Rule of Five": "A common oral-drug-likeness heuristic considers molecular weight ≤500 Da, calculated logP ≤5, hydrogen-bond donors ≤5 and acceptors ≤10. It is a heuristic, not a universal rule and not a prediction of whether a drug will work.",
    "2D and 3D representations": "A 2D depiction shows atom connectivity and bond arrangement. A 3D conformer is a computational geometry generated under assumptions; it is not automatically the experimentally observed bioactive conformation.",
    "Structure–property relationships": "Molecular structure influences physicochemical properties. Comparisons are most informative when the molecules and descriptors are chosen carefully and the limitations of calculated values are stated.",
}

def make_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def descriptors(mol):
    return {
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular weight (Da)": round(Descriptors.MolWt(mol), 2),
        "Exact mass (Da)": round(Descriptors.ExactMolWt(mol), 4),
        "cLogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
        "H-bond donors": int(Lipinski.NumHDonors(mol)),
        "H-bond acceptors": int(Lipinski.NumHAcceptors(mol)),
        "Rotatable bonds": int(Lipinski.NumRotatableBonds(mol)),
        "Heavy atoms": int(mol.GetNumHeavyAtoms()),
        "Aromatic rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "Fraction sp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }

def structure_png(mol, size=(700, 450)):
    copy = Chem.Mol(mol)
    AllChem.Compute2DCoords(copy)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 2.0
    drawer.DrawMolecule(copy)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def get_molecule(name):
    return next(item for item in MOLECULES if item["name"] == name)

def molecule_frame():
    rows = []
    for item in MOLECULES:
        mol = make_mol(item["smiles"])
        if mol:
            row = {"Name": item["name"], "Category": item["category"], **descriptors(mol)}
            rows.append(row)
    return pd.DataFrame(rows)

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "registration" not in st.session_state:
    st.session_state.registration = ""

with st.sidebar:
    st.markdown("## 🧬 MoleculeLab")
    st.caption("CHEMOINFORMATICS • VIRTUAL LAB")
    page = st.radio(
        "WORKSPACE",
        ["🏠 Home", "📚 Theory", "🧪 Molecular Studio", "📊 Properties Report",
         "🔬 Structure–Property Analysis", "📝 Assessment", "📄 Project Report"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("### 🎓 Student details")
    st.session_state.student_name = st.text_input("Student name", value=st.session_state.student_name, placeholder="Enter your name")
    st.session_state.registration = st.text_input("Registration number", value=st.session_state.registration, placeholder="Enter your registration number")
    st.markdown("---")
    st.markdown("### ✨ Quick facts")
    st.caption(f"{len(MOLECULES)} curated molecule records")
    st.caption("Local structure calculations")
    st.caption("Educational use only")
    st.markdown("---")
    st.caption("Built with Python · Streamlit · RDKit")

# Header / page routing
st.markdown("""
<div class="hero">
  <div class="eyebrow">Interactive learning environment</div>
  <h1>🧬 MoleculeLab</h1>
  <p>Explore drug-like molecular structures, calculate descriptors, and investigate structure–property relationships.</p>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Home":
    st.markdown("### Welcome to your chemoinformatics workspace")
    st.write("A hands-on virtual lab for learning how molecular structures can be represented, visualized, and described computationally.")
    c1,c2,c3,c4 = st.columns(4)
    for col, label, value in zip([c1,c2,c3,c4], ["Molecule records","Chemical categories","Core tools","Learning modules"], [len(MOLECULES), len(set(x["category"] for x in MOLECULES)), "RDKit", "5"]):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    left,right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🔎 Start exploring")
        query = st.text_input("Search molecules", placeholder="Try paracetamol, caffeine, antibiotic…", key="home_search")
        cats = ["All categories"] + sorted({x["category"] for x in MOLECULES})
        cat = st.selectbox("Filter by category", cats)
        matches = [x for x in MOLECULES if (query.lower() in (x["name"]+" "+x["category"]+" "+x["use"]).lower()) and (cat=="All categories" or x["category"]==cat)]
        st.caption(f"{len(matches)} molecule(s) match your filters")
        for item in matches[:8]:
            st.markdown(f'**{item["name"]}**  ·  <span class="pill">{item["category"]}</span><br><span class="small-muted">{item["use"]}</span>', unsafe_allow_html=True)
            st.markdown("---")
        if len(matches)>8:
            st.caption("Showing first 8 results. Use Molecular Studio to inspect any molecule.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🧭 Learning pathway")
        st.markdown("1. **Theory** — learn SMILES and descriptors.\n2. **Molecular Studio** — inspect structures and properties.\n3. **Properties Report** — filter and download a dataset.\n4. **Structure–Property Analysis** — compare descriptors visually.\n5. **Assessment** — test your understanding.\n6. **Project Report** — generate a summary for your lab record.")
        st.info("The listed medicines are examples for chemical-structure education, not treatment recommendations.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("#### Featured molecules")
    featured = ["Paracetamol","Metformin","Caffeine","Fluconazole"]
    cols = st.columns(4)
    for col, name in zip(cols, featured):
        item = get_molecule(name)
        mol = make_mol(item["smiles"])
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.image(structure_png(mol, (320,220)), use_container_width=True)
            st.markdown(f"**{name}**")
            st.caption(item["category"])
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "📚 Theory":
    st.markdown("### Foundations of chemoinformatics")
    st.write("Use these short notes as a starting point; add your class notes and references for the final submission.")
    for title, body in THEORY.items():
        with st.expander(title, expanded=(title=="SMILES notation")):
            st.write(body)
    st.markdown("#### Mini example: SMILES")
    st.code("Paracetamol: CC(=O)NC1=CC=C(O)C=C1O", language="text")
    st.caption("A SMILES string is a representation of connectivity, not a 2D picture itself.")

elif page == "🧪 Molecular Studio":
    st.markdown("### Molecular studio")
    search = st.text_input("Find a molecule", placeholder="Type a molecule or category", key="studio_search")
    options = [x["name"] for x in MOLECULES if search.lower() in (x["name"]+" "+x["category"]).lower()]
    if not options:
        st.warning("No molecule matches that search.")
    else:
        selected = st.selectbox("Choose molecule", options)
        item = get_molecule(selected)
        mol = make_mol(item["smiles"])
        if mol is None:
            st.error("This structure could not be parsed. Please review its SMILES.")
        else:
            left,right = st.columns([1.05,1], gap="large")
            with left:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown(f"#### {item['name']}")
                st.caption(item["category"])
                st.image(structure_png(mol), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Molecular identity")
                d = descriptors(mol)
                st.markdown(f"**Molecular formula:** `{d['Molecular formula']}`")
                st.markdown(f"**SMILES:** `{Chem.MolToSmiles(mol)}`")
                st.markdown(f"**Educational context:** {item['use']}")
                st.write(item["note"])
                st.markdown("#### Computed descriptors")
                st.dataframe(pd.DataFrame({"Descriptor":list(d.keys()),"Value":list(d.values())}), hide_index=True, use_container_width=True)
                fav_label = "★ Remove from favourites" if selected in st.session_state.favorites else "☆ Add to favourites"
                if st.button(fav_label, key=f"fav_{selected}"):
                    if selected in st.session_state.favorites:
                        st.session_state.favorites.remove(selected)
                    else:
                        st.session_state.favorites.append(selected)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("#### Structure input")
            with st.expander("Inspect or validate a custom SMILES string"):
                custom = st.text_input("SMILES", placeholder="Paste a SMILES string")
                if st.button("Validate custom structure"):
                    custom_mol = make_mol(custom.strip())
                    if not custom.strip():
                        st.warning("Enter a SMILES string first.")
                    elif custom_mol is None:
                        st.error("RDKit could not parse this SMILES string.")
                    else:
                        st.success("SMILES parsed successfully.")
                        st.image(structure_png(custom_mol, (700,400)), use_container_width=True)
                        st.dataframe(pd.DataFrame({"Descriptor":list(descriptors(custom_mol).keys()),"Value":list(descriptors(custom_mol).values())}), hide_index=True, use_container_width=True)
            with st.expander("3D conformer (experimental)"):
                if st.button("Generate 3D conformer"):
                    with st.spinner("Generating a conformer…"):
                        m3 = Chem.AddHs(Chem.Mol(mol))
                        status = AllChem.EmbedMolecule(m3, randomSeed=42)
                        if status == 0:
                            AllChem.MMFFOptimizeMolecule(m3, maxIters=200)
                            conf = m3.GetConformer()
                            lines = [f"{m3.GetNumAtoms()}","MoleculeLab educational conformer"]
                            for atom in m3.GetAtoms():
                                pos = conf.GetAtomPosition(atom.GetIdx())
                                lines.append(f"{atom.GetSymbol():2s} {pos.x: .5f} {pos.y: .5f} {pos.z: .5f}")
                            xyz = "\n".join(lines)
                            st.success("A computed conformer was generated. It is a model, not an experimental structure.")
                            st.download_button("Download XYZ coordinates", xyz, file_name=f"{selected.lower().replace(' ','_')}.xyz", mime="text/plain")
                        else:
                            st.warning("The conformer could not be generated for this structure.")

elif page == "📊 Properties Report":
    st.markdown("### Molecular properties report")
    df = molecule_frame()
    f1,f2 = st.columns(2)
    with f1:
        cat_filter = st.multiselect("Categories", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    with f2:
        name_filter = st.text_input("Search names", placeholder="Optional name filter")
    filtered = df[df["Category"].isin(cat_filter)]
    if name_filter:
        filtered = filtered[filtered["Name"].str.contains(name_filter, case=False, na=False)]
    st.caption(f"{len(filtered)} of {len(df)} valid molecule records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", csv, file_name="molecule_properties_report.csv", mime="text/csv")
    st.markdown("#### Rule of Five screening (heuristic only)")
    df["Ro5 criteria met"] = (
        (df["Molecular weight (Da)"] <= 500) &
        (df["cLogP"] <= 5) &
        (df["H-bond donors"] <= 5) &
        (df["H-bond acceptors"] <= 10)
    )
    st.dataframe(df[["Name","Molecular weight (Da)","cLogP","H-bond donors","H-bond acceptors","Ro5 criteria met"]], hide_index=True, use_container_width=True)
    st.caption("This heuristic does not establish absorption, safety, effectiveness, or suitability for any patient.")

elif page == "🔬 Structure–Property Analysis":
    st.markdown("### Structure–property analysis")
    df = molecule_frame()
    a,b,c = st.columns(3)
    with a:
        x = st.selectbox("X-axis descriptor", ["Molecular weight (Da)","cLogP","TPSA (Å²)","Heavy atoms","Rotatable bonds","Aromatic rings"], index=0)
    with b:
        y = st.selectbox("Y-axis descriptor", ["cLogP","TPSA (Å²)","Molecular weight (Da)","H-bond donors","H-bond acceptors","Fraction sp3"], index=0)
    with c:
        hue = st.selectbox("Colour by", ["Category","Name"])
    fig = px.scatter(df, x=x, y=y, color=hue, hover_name="Name", hover_data=["Molecular formula","Category"], title=f"{y} vs {x}")
    fig.update_layout(template="plotly_white", legend_title_text=hue, margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Compare two molecules")
    names = [x["name"] for x in MOLECULES]
    p1,p2 = st.columns(2)
    with p1:
        first = st.selectbox("Molecule A", names, index=0)
    with p2:
        second = st.selectbox("Molecule B", names, index=1)
    d1 = descriptors(make_mol(get_molecule(first)["smiles"]))
    d2 = descriptors(make_mol(get_molecule(second)["smiles"]))
    compare = pd.DataFrame({"Descriptor":d1.keys(), first:d1.values(), second:d2.values()})
    st.dataframe(compare, hide_index=True, use_container_width=True)
    numeric = ["Molecular weight (Da)","Exact mass (Da)","cLogP","TPSA (Å²)","H-bond donors","H-bond acceptors","Rotatable bonds","Heavy atoms","Aromatic rings","Fraction sp3"]
    chart_df = pd.DataFrame({"Descriptor":numeric, first:[d1[k] for k in numeric], second:[d2[k] for k in numeric]})
    st.bar_chart(chart_df.set_index("Descriptor")[[first,second]])

elif page == "📝 Assessment":
    st.markdown("### Knowledge check")
    st.write("Choose the best answer for each question, then submit to see your score.")
    questions = [
        ("What does SMILES primarily encode?", ["A molecule's connectivity as text","Only its colour","Its clinical dose","Its measured melting point"], 0, "SMILES encodes molecular connectivity as a text representation."),
        ("Which tool is used here to parse SMILES and calculate descriptors?", ["RDKit","PowerPoint","HTML only","A spreadsheet formula"], 0, "RDKit provides molecular parsing and descriptor calculations."),
        ("What does TPSA stand for?", ["Topological polar surface area","Total protein safety assessment","Thermal pressure standard amount","Target prediction sequence analysis"], 0, "TPSA is a calculated descriptor related to polar surface area."),
        ("The Rule of Five is best described as…", ["A heuristic, not a guarantee","A proof a medicine is safe","A dosing guideline","A way to identify every side effect"], 0, "It is a drug-likeness heuristic and has important limitations."),
        ("A generated 3D conformer is…", ["A computational model","Always an experimentally observed structure","A guarantee of biological activity","A clinical recommendation"], 0, "A computed conformer is a model and may not match an experimental bioactive conformation."),
    ]
    with st.form("quiz"):
        answers = []
        for i,(q,opts,correct,explanation) in enumerate(questions):
            answers.append(st.radio(f"{i+1}. {q}", opts, key=f"q{i}"))
        submitted = st.form_submit_button("Submit assessment", type="primary")
    if submitted:
        score = sum(1 for ans, q in zip(answers, questions) if ans == q[1][q[2]])
        st.success(f"Your score: {score}/{len(questions)} ({score/len(questions)*100:.0f}%)")
        for i,((q,opts,correct,explanation),ans) in enumerate(zip(questions,answers)):
            if ans == opts[correct]:
                st.markdown(f"✅ **Q{i+1}: Correct.** {explanation}")
            else:
                st.markdown(f"❌ **Q{i+1}: Correct answer: {opts[correct]}.** {explanation}")

elif page == "📄 Project Report":
    st.markdown("### Project report builder")
    st.write("Create a concise, downloadable summary of the application and your selected molecule. Add your course-specific requirements before submission.")
    names = [x["name"] for x in MOLECULES]
    chosen = st.multiselect("Molecules to include", names, default=["Paracetamol","Metformin","Caffeine"])
    objectives = st.text_area("Your project objectives", value="1. Represent molecules using SMILES.\n2. Visualize molecular structures.\n3. Calculate and compare molecular descriptors.\n4. Explore structure–property relationships.")
    observations = st.text_area("Your observations", placeholder="Write observations from your own analysis…")
    if st.button("Generate report", type="primary"):
        report_lines = [
            "# Chemoinformatics Virtual Laboratory — Project Report",
            f"**Student:** {st.session_state.student_name or 'Not entered'}",
            f"**Registration number:** {st.session_state.registration or 'Not entered'}",
            f"**Generated:** {datetime.now().strftime('%d %B %Y, %H:%M')}",
            "",
            "## Aim",
            "To explore molecular representation, visualization, calculated physicochemical descriptors, and structure–property relationships using a Python-based virtual laboratory.",
            "",
            "## Objectives",
            objectives,
            "",
            "## Methodology",
            "Molecular SMILES were parsed using RDKit. 2D depictions and selected molecular descriptors were generated computationally. Descriptor comparisons were visualized using Plotly.",
            "",
            "## Selected molecules",
        ]
        for name in chosen:
            item = get_molecule(name)
            mol = make_mol(item["smiles"])
            d = descriptors(mol)
            report_lines += [f"### {name}", f"- Category: {item['category']}", f"- SMILES: `{Chem.MolToSmiles(mol)}`", f"- Formula: {d['Molecular formula']}", f"- Molecular weight: {d['Molecular weight (Da)']} Da", f"- cLogP: {d['cLogP']}", f"- TPSA: {d['TPSA (Å²)']} Å²", f"- Context: {item['use']}", ""]
        report_lines += ["## Observations", observations or "Add your own observations based on the results.", "", "## Limitations", "Calculated descriptors are model-dependent and do not establish clinical safety, efficacy, dosage, or suitability. The dataset is educational and should be verified against authoritative chemical references before formal submission.", "", "## Conclusion", "The application demonstrates how cheminformatics tools can represent and compare molecular structures and selected physicochemical descriptors."]
        report = "\n".join(report_lines)
        st.markdown("#### Preview")
        st.markdown(report)
        st.download_button("⬇️ Download Markdown report", report, file_name="cheminformatics_project_report.md", mime="text/markdown")

st.markdown("---")
st.markdown('<div class="small-muted">MoleculeLab is an educational cheminformatics demonstration. Molecular descriptors are calculated estimates and are not medical advice or clinical predictions.</div>', unsafe_allow_html=True)

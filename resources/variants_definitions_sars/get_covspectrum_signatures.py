import requests
import pandas as pd

def get_variant_mutations(variant_name, 
                           min_proportion=0.80,
                           min_count=100):
    """
    Query CovSpectrum LAPIS API for mutations defining a variant.
    
    Parameters
    ----------
    variant_name   : str    e.g. 'B.1.617.2', 'BA.2'
    min_proportion : float  minimum fraction of sequences with mutation
    min_count      : int    minimum number of sequences supporting mutation
    
    Returns
    -------
    pd.DataFrame   mutations with proportion and count
    """
    url = "https://lapis.cov-spectrum.org/open/v2/sample/nucleotideMutations"

    params = {
        'nextcladePangoLineage':  variant_name + '*',
        'minProportion': min_proportion,
        'limit':         10000

    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    data = response.json()['data']
    df   = pd.DataFrame(data)
    
    # Filter by minimum count
    if 'count' in df.columns:
        df = df[df['count'] >= min_count]
    
    return df


# ── Query all variants ────────────────────────────────────────────────────────
variants = [
# ── Early variants ──────────────────────────────
    "B.1.1.7",      # Alpha
    #"B.1.351",      # Beta
    #"P.1",          # Gamma
    "B.1.617.2",    # Delta

    # ── Omicron ─────────────────────────────────────
    "BA.1",         # Omicron BA.1
    "BA.2",         # Omicron BA.2
    "BA.4",         # Omicron BA.4
    "BA.5",         # Omicron BA.5
    "BA.2.75",      # Omicron BA.2.75 (Centaurus)

    # ── Omicron  ──────────────────────────
    "BQ.1.1",       # Omicron BQ.1.1
    "XBB.1.5",      # Omicron XBB.1.5 (Kraken)
    "XBB.1.9",      # Omicron XBB.1.9
    "XBB.1.16",     # Omicron XBB.1.16 (Arcturus)
    "XBB.2.3",      # Omicron XBB.2.3
    "EG.5",         # Omicron EG.5 (Eris)
    #"BA.2.86",      # Omicron BA.2.86 (Pirola)
    "JN.1",         # Omicron JN.1 (Pirola descendant)
    #"KP.2",         # Omicron KP.2
    #"KP.3",         # Omicron KP.3
]

all_mutations = {}

for lineage in variants:
    print(f"Querying {lineage} ...")
    df = get_variant_mutations(lineage,
                               min_proportion=0.80,
                               min_count=100)
    # ── Keep substitutions only ───────────────────────────────────────
    valid_nucleotides = {'A', 'C', 'G', 'T'}
    df = df[
        df['mutationFrom'].isin(valid_nucleotides) &
        df['mutationTo'].isin(valid_nucleotides)
    ]
    all_mutations[lineage] = df
    print(f"  → {len(df)} mutations found")


# ── Build binary signature matrix ─────────────────────────────────────────────
# Collect all unique mutations
all_mut_names = sorted(set(
    mut 
    for df in all_mutations.values() 
    for mut in df['mutation'].tolist()
))

# Build binary matrix
W_true = pd.DataFrame(0,
                       index=all_mut_names,
                       columns=list(variants))





for lineage, df in all_mutations.items():
    W_true.loc[df['mutation'], lineage] = 1

print(f"\nSignature matrix shape: {W_true.shape}")
print(W_true.sum(axis=0))   # mutations per variant

# Save
W_true.to_csv('../../resources/variants_definitions_sars/variant_signatures_covspectrum_2026_08_27.csv')
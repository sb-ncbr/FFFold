import requests

def valid_pH(ph):
    if ph is None:
        return 7.0, True
    try:
        ph = float(ph)
    except ValueError:
        return ph, False
    if not 0 <= ph <= 14:
        return ph, False
    return ph, True

def valid_alphafold_request(code):
    # check whether UniProt code is valid, ping AlphaFold website
    response = requests.head(f'https://alphafold.ebi.ac.uk/files/AF-{code}-F1-model_v4.pdb')
    return response.status_code == 200

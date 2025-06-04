import pandas as pd
import datetime
import os


def variables_uniques(df, exclude):
    """Affiche les valeurs uniques des colonnes sauf celles exclues."""
    for col in df.columns:
        if col not in exclude:
            uniques = ', '.join(map(str, df[col].unique()))
            print(f"{col}: {uniques}")

def load_data(path_dict):
    """Charge les fichiers CSV des datasets avec un dict {name: filepath}."""
    data = {}
    for name, path in path_dict.items():
        data[name] = pd.read_csv(path, sep=",")
    return data

def preprocess_caract(caract):
    """Nettoyage et catégorisation du dataset 'caract'."""
    caract = caract.drop(columns=["adr", "com"])

    # Conversion horaire en catégorie
    caract["hrmn"] = caract["hrmn"].astype(str).str[:2].astype(int)
    def categoriser_heure(h):
        if 6 <= h < 12:
            return 'matin'
        elif 12 <= h < 18:
            return 'après-midi'
        elif 18 <= h < 24:
            return 'soir'
        else:
            return 'nuit'
    caract['heure_categorie'] = caract['hrmn'].apply(categoriser_heure)
    caract.drop(columns=['hrmn'], inplace=True)

    # Catégorisation jour semaine/weekend
    def categoriser_jour(j, mois):
        date = datetime.date(2023, mois, j)
        return 'semaine' if date.weekday() < 5 else 'weekend'
    caract['jour_categorie'] = caract.apply(lambda r: categoriser_jour(r['jour'], r['mois']), axis=1)
    caract.drop(columns=['jour'], inplace=True)

    # Catégorisation saison
    def categoriser_mois(m):
        if m in [12, 1, 2]:
            return 'hiver'
        elif 3 <= m <= 5:
            return 'printemps'
        elif 6 <= m <= 8:
            return 'été'
        else:
            return 'automne'
    caract['mois_categorie'] = caract['mois'].apply(categoriser_mois)

    # Catégorisation départementale
    def categoriser_dep(dep):
        # Paris
        if dep == "75":
            return "Paris"
        # Grandes villes
        elif dep in ["69", "31", "13", "33", "59", "92", "93", "94"]:
            return "Grandes Villes"
        # Île-de-France hors Paris
        elif dep in ["77", "78", "91", "95"]:
            return "Île-de-France"
        # Sud (PACA, Occitanie, Corse élargie)
        elif dep in ["06", "13", "30", "34", "66", "83", "84", "2A", "2B",
                     "04", "05", "07", "09", "11", "12", "81", "82"]:
            return "Sud"
        # Nord
        elif dep in ["59", "62", "80", "02", "60", "08"]:
            return "Nord"
        # Est (Grand Est + Bourgogne/Franche-Comté)
        elif dep in ["67", "68", "88", "54", "55", "57", "52", "10",
                     "70", "90", "25", "39", "21", "71", "89", "58"]:
            return "Est"
        # Ouest (Bretagne + Pays de la Loire élargi)
        elif dep in ["29", "22", "35", "56", "44", "49", "53", "72",
                     "85", "50", "61", "14"]:
            return "Ouest"
        # Centre
        elif dep in ["45", "28", "41", "18", "36", "37", "03", "23",
                     "19", "15", "43", "63", "87"]:
            return "Centre"
        # Sud-Ouest
        elif dep in ["16", "17", "24", "40", "47", "64", "79", "86", "87"]:
            return "Sud-Ouest"
        # DROM
        elif dep in ["971", "972", "973", "974", "976"]:
            return "Outre-Mer"
        else:
            return "Autre"
    caract["dep_cat"] = caract["dep"].apply(categoriser_dep)

    # Remplacement valeurs manquantes et aberrantes pour 'col' et 'atm'
    for col in ['col', 'atm']:
        med = caract[col][caract[col] != -1].median()
        caract[col] = caract[col].replace(-1, med).fillna(med).astype(int)

    return caract

def preprocess_lieux(lieux):
    """Nettoyage du dataset 'lieux'."""
    lieux = lieux.drop(columns=["voie", 'v1', 'v2', 'vosp', 'pr', 'pr1'])
    lieux = lieux[~lieux.duplicated()]

    # Remplacement de virgules par points et conversion numérique
    for col in ["lartpc", "larrout", "nbv"]:
        lieux[col] = lieux[col].astype(str).str.replace(',', '.')
        lieux[col] = pd.to_numeric(lieux[col], errors='coerce')

    lieux["nbv"].fillna(-1, inplace=True)
    lieux = lieux.drop(columns=["lartpc"])  # trop de NaN

    return lieux

def preprocess_vehicules(vehicules):
    """Nettoyage du dataset 'vehicules'."""
    vehicules = vehicules.drop(columns=["senc", "num_veh"])

    def sous_categoriser_catv(catv):
        if catv in [1, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43]:
            return '2 roues'
        elif catv in [7, 10]:
            return 'Véhicules légers'
        elif catv in [13, 14, 15, 16, 17]:
            return 'Poids lourds'
        elif catv in [37, 38]:
            return 'Transports en commun'
        elif catv in [20, 21]:
            return 'Véhicules spéciaux'
        elif catv in [50, 60, 80]:
            return 'Engins de déplacement personnel'
        elif catv in [39, 40]:
            return 'Rails (Train/Tramway)'
        elif catv in [0, 99]:
            return 'Autre/Indéterminé'
        else:
            return 'Inconnu'

    vehicules['catv_categorie'] = vehicules['catv'].apply(sous_categoriser_catv)
    vehicules = vehicules.drop(columns=["catv", "occutc"])

    return vehicules

def preprocess_usagers(usagers):
    """Nettoyage du dataset 'usagers'."""
    usagers = usagers.drop(columns=["num_veh", "trajet", "secu2", "secu3", "etatp"])

    # Remplacement des valeurs -1 et '-1' par NA
    for col in usagers.columns:
        if usagers[col].dtype == object:
            usagers[col] = usagers[col].replace('-1', pd.NA)
        else:
            usagers[col] = usagers[col].replace(-1, pd.NA)
    usagers = usagers.dropna()

    return usagers

def save_clean_data(data_dict, output_dir):
    """Sauvegarde des datasets nettoyés dans output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in data_dict.items():
        df.to_csv(os.path.join(output_dir, f"{name}_clean.csv"), index=False)

if __name__ == "__main__":
    paths = {
        "caract": "data/raw/caract-2023.csv",
        "lieux": "data/raw/lieux-2023.csv",
        "vehicules": "data/raw/vehicules-2023.csv",
        "usagers": "data/raw/usagers-2023.csv"
    }

    data = load_data(paths)
    data["caract"] = preprocess_caract(data["caract"])
    data["lieux"] = preprocess_lieux(data["lieux"])
    data["vehicules"] = preprocess_vehicules(data["vehicules"])
    data["usagers"] = preprocess_usagers(data["usagers"])

    save_clean_data(data, "data/processed")
    print("Pré-traitement terminé, fichiers sauvegardés dans data/processed/")

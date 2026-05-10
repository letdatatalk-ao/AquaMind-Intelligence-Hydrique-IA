import pandas as pd

# Chargement du fichier
df = pd.read_parquet("data/master_dataset.parquet")
cust_a = df[df["customer"] == "CustomerA"]

print("=== DIAGNOSTIC CUSTOMER A ===")
print("1. Répartition des catégories :")
print(cust_a["sub_category"].value_counts())

print("\n2. Chaque capteur (device_id) gère-t-il plusieurs types d'équipements ?")
equipements_par_device = cust_a.groupby("device_id")["sub_category"].nunique()
print(equipements_par_device.value_counts())
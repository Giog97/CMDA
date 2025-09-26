import pandas as pd
import matplotlib.pyplot as plt

# Legge i dati dal file CSV che hai salvato
data = pd.read_csv('loss_l2_source_target.csv')

# Estrae le colonne per il grafico
iterazioni = data['iterazione']
loss = data['loss']

# Crea il grafico
plt.figure(figsize=(12, 7))
plt.plot(iterazioni, loss, marker='.', linestyle='-', color='royalblue')

# Aggiunge titoli e etichette per chiarezza
plt.title('Andamento di loss_l2_source_target', fontsize=16)
plt.xlabel('Iterazioni', fontsize=12)
plt.ylabel('Valore della Loss', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.yscale('log') # Usa la scala logaritmica per vedere meglio i piccoli valori
plt.tight_layout()

# Salva il grafico in un file PNG
nome_file_grafico = 'grafico_lossL2_source_target.png'
plt.savefig(nome_file_grafico)

print(f"Grafico salvato con successo come '{nome_file_grafico}'")
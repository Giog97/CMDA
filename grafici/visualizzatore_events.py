# Script per visualizzare eventi
# esegure con python visualizzatore_events.py ./data/DSEC_Night/zurich_city_09_c/events/events_vg/000658.npy --output-dir ./risultati_analisi/
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import datetime

def visualize_npy_without_modification(npy_path, output_dir=None):
    """
    Visualizza un file .npy senza modificare l'originale
    Crea nuove immagini e file in una directory separata
    """
    
    try:
        # Carica i dati (sola lettura)
        data = np.load(npy_path)
        print(f"File caricato: {npy_path}")
        print(f"Shape originale: {data.shape}")
        print(f"Tipo dati: {data.dtype}")
        print(f"Valore min: {data.min():.6f}, max: {data.max():.6f}")
        
        # Crea directory di output se non specificata
        if output_dir is None:
            # Crea directory 'visualizations' nella stessa posizione del file originale
            original_dir = os.path.dirname(npy_path)
            output_dir = os.path.join(original_dir, "visualizations")
        
        # Crea la directory di output (e tutte le sottodirectory necessarie)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory output: {output_dir}")
        
        # Base name per i file di output
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        
        # 1. Salva i dati normalizzati in un NUOVO file .npy
        data_squeezed = data.squeeze()
        normalized_data = (data_squeezed - data_squeezed.min()) / (data_squeezed.max() - data_squeezed.min() + 1e-8)
        
        normalized_npy_path = os.path.join(output_dir, f"{base_name}_normalized.npy")
        np.save(normalized_npy_path, normalized_data)
        print(f"✓ File normalizzato salvato: {normalized_npy_path}")
        
        # 2. Crea immagine visualizzabile
        img_data = (normalized_data * 255).astype(np.uint8)
        
        # Salva immagine PNG
        img_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Immagine
        plt.subplot(1, 2, 1)
        if len(img_data.shape) == 2:
            plt.imshow(img_data, cmap='viridis')
            plt.title(f'Visualizzazione\n{img_data.shape}')
        elif len(img_data.shape) == 3:
            if img_data.shape[2] == 3:
                plt.imshow(img_data)
                plt.title(f'Immagine RGB\n{img_data.shape}')
            else:
                plt.imshow(img_data[:,:,0], cmap='viridis')
                plt.title(f'Primo canale\n{img_data.shape}')
        
        plt.axis('off')
        plt.colorbar()
        
        # Subplot 2: Istogramma
        plt.subplot(1, 2, 2)
        plt.hist(data_squeezed.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribuzione Valori Originali')
        plt.xlabel('Valore')
        plt.ylabel('Frequenza')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', dpi=120)
        plt.close()
        
        print(f"✓ Immagine salvata: {img_path}")
        
        # 3. Salva metadati in un file di testo
        metadata_path = os.path.join(output_dir, f"{base_name}_info.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Analisi del file: {base_name}.npy\n")
            f.write(f"Data analisi: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Path originale: {npy_path}\n")
            f.write(f"Path output: {output_dir}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Shape originale: {data.shape}\n")
            f.write(f"Shape dopo squeeze: {data_squeezed.shape}\n")
            f.write(f"Tipo dati: {data.dtype}\n")
            f.write(f"Valore minimo: {data.min():.6f}\n")
            f.write(f"Valore massimo: {data.max():.6f}\n")
            f.write(f"Media: {data.mean():.6f}\n")
            f.write(f"Deviazione standard: {data.std():.6f}\n")
            f.write("=" * 50 + "\n")
            f.write(f"File creati:\n")
            f.write(f"- Normalizzato: {os.path.basename(normalized_npy_path)}\n")
            f.write(f"- Immagine: {os.path.basename(img_path)}\n")
            f.write(f"- Metadati: {os.path.basename(metadata_path)}\n")
        
        print(f"✓ Metadati salvati: {metadata_path}")
        print(f"✓ Analisi completata! File salvati in: {output_dir}")
        
        return {
            'original_path': npy_path,
            'normalized_npy': normalized_npy_path,
            'image_path': img_path,
            'metadata_path': metadata_path
        }
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

def process_multiple_files(input_dir, output_dir=None):
    """Elabora tutti i file .npy in una directory"""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"❌ Directory non trovata: {input_dir}")
        return
    
    # Crea directory di output se non specificata
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_visualizations"
    
    npy_files = list(input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"❌ Nessun file .npy trovato in: {input_dir}")
        return
    
    print(f"Trovati {len(npy_files)} file .npy")
    print(f"Output directory: {output_dir}")
    
    # Crea directory output
    os.makedirs(output_dir, exist_ok=True)
    
    for i, npy_file in enumerate(npy_files):
        print(f"\n--- [{i+1}/{len(npy_files)}] Elaborando {npy_file.name} ---")
        visualize_npy_without_modification(str(npy_file), str(output_dir))
    
    print(f"\n🎯 Elaborazione completata! Tutti i file salvati in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualizza file .npy senza modificare l\'originale')
    parser.add_argument('input_path', help='Percorso del file .npy o directory')
    parser.add_argument('--output-dir', '-o', help='Directory di output personalizzata')
    parser.add_argument('--batch', '-b', action='store_true', help='Elabora tutti i file nella directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"❌ Percorso non trovato: {args.input_path}")
        return
    
    if args.batch:
        # Modalità batch: elabora tutti i file in una directory
        process_multiple_files(args.input_path, args.output_dir)
    else:
        # Modalità singolo file
        if os.path.isdir(args.input_path):
            print("❌ Percorso è una directory. Usa --batch per elaborare tutti i file")
            return
        visualize_npy_without_modification(args.input_path, args.output_dir)

if __name__ == "__main__":
    main()
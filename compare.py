import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

    # WYKRES 1: Słupkowy (MAE i RMSE)
    # MAE (Mean Absolute Error) mówi o średnim błędzie.
    # RMSE (Root Mean Squared Error) bardziej karze duże pomyłki - 
    # jeśli RMSE jest dużo wyższe od MAE, oznacza to, że model zalicza "wtopy" na niektórych obiektach.
    
    # WYKRES 2: Scatter Plot (Predykcja vs Prawda)
    # Linia przerywana (y=x) to ideał. 
    # Im punkty są bliżej linii, tym model jest dokładniejszy.
    # Skupienie punktów w lewym dolnym rogu (małe MOID) jest kluczowe dla bezpieczeństwa planetarnego.

# AUTOMATYCZNE WYSZUKIWANIE PLIKÓW
def find_file(filename, root_folder="."):
    """Przeszukuje katalog główny i podkatalogi w poszukiwaniu pliku."""
    for dirpath, _, filenames in os.walk(root_folder):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

print("Szukam plików modeli i danych w obecnym katalogu i podkatalogach...")

PINN_MODEL_PATH = find_file('best_neo_pinn_model.pth')
BLACKBOX_MODEL_PATH = find_file('best_neo_blackbox_model.pth')
DATA_CSV_PATH = find_file('NEO_Curated.csv')

# Sprawdzanie czy wszystkie pliki zostały znalezione
missing_files = []
if not PINN_MODEL_PATH: missing_files.append("best_neo_pinn_model.pth")
if not BLACKBOX_MODEL_PATH: missing_files.append("best_neo_blackbox_model.pth")
if not DATA_CSV_PATH: missing_files.append("NEO_Curated.csv")

if missing_files:
    print("\nBŁĄD: Nie znaleziono następujących plików:")
    for mf in missing_files:
        print(f" - {mf}")
    print("\nUpewnij się, że uruchamiasz ten skrypt w nadrzędnym folderze, który zawiera oba projekty.")
    sys.exit(1)

print(f"Znaleziono model PINN: {PINN_MODEL_PATH}")
print(f"Znaleziono model Black Box: {BLACKBOX_MODEL_PATH}")
print(f"Znaleziono plik danych: {DATA_CSV_PATH}\n")

# definicje architektury
class NeoKeplerPINN(nn.Module):
    def __init__(self, input_size):
        super(NeoKeplerPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2) # moid i q
        )
    def forward(self, x): return self.net(x)

class NeoBlackBox(nn.Module):
    def __init__(self, input_size):
        super(NeoBlackBox, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1) # tylko moid
        )
    def forward(self, x): return self.net(x)

# logika porownawcza
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def prepare_test_data():
    """Odtwarza ten sam zbiór testowy co w engine.py"""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    df = pd.read_csv(DATA_CSV_PATH, low_memory=False)
    
    # przetwarzanie cech
    df['i_rad'], df['om_rad'], df['w_rad'] = np.radians(df['i']), np.radians(df['om']), np.radians(df['w'])
    df['i_sin'], df['i_cos'] = np.sin(df['i_rad']), np.cos(df['i_rad'])
    df['om_sin'], df['om_cos'] = np.sin(df['om_rad']), np.cos(df['om_rad'])
    df['w_sin'], df['w_cos'] = np.sin(df['w_rad']), np.cos(df['w_rad'])
    df = df[(df['H'] > -1) & (df['a'] > 0)].copy()
    df['H_log'], df['a_log'] = np.log1p(df['H']), np.log1p(df['a'])

    features_cols = ['H_log', 'e', 'a_log', 'i_sin', 'i_cos', 'om_sin', 'om_cos', 'w_sin', 'w_cos']
    target_cols = ['moid_ld'] 
    
    required_cols = features_cols + target_cols + ['a', 'i', 'om', 'w', 'ma', 'n', 'epoch']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])

    X = df[features_cols].values
    y = df[target_cols].values

    scaler_X = StandardScaler()
    scaler_y = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # odwzorowanie tych samych podziałów
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    return device, X_test_t, y_test, scaler_y, len(features_cols)

def main():
    print("Preparing data and loading identical test set...")
    device, X_test_t, y_test_scaled, scaler_y, input_size = prepare_test_data()
    y_true = scaler_y.inverse_transform(y_test_scaled)[:, 0]

    # proces myslowy PINN
    print("Evaluating PINN Model...")
    pinn_model = NeoKeplerPINN(input_size).to(device)
    pinn_model.load_state_dict(torch.load(PINN_MODEL_PATH, map_location=device))
    pinn_model.eval()
    with torch.no_grad():
        pinn_pred_scaled = pinn_model(X_test_t).cpu().numpy()
        pinn_moid_scaled = pinn_pred_scaled[:, 0].reshape(-1, 1)
        pinn_pred = scaler_y.inverse_transform(pinn_moid_scaled)[:, 0]

    # proces myslowy black box
    print("Evaluating Black Box Model...")
    bb_model = NeoBlackBox(input_size).to(device)
    bb_model.load_state_dict(torch.load(BLACKBOX_MODEL_PATH, map_location=device))
    bb_model.eval()
    with torch.no_grad():
        bb_pred_scaled = bb_model(X_test_t).cpu().numpy()
        bb_pred = scaler_y.inverse_transform(bb_pred_scaled)[:, 0]

    # obliczanie metryk
    pinn_mae = np.mean(np.abs(pinn_pred - y_true))
    pinn_rmse = np.sqrt(np.mean((pinn_pred - y_true)**2))
    
    bb_mae = np.mean(np.abs(bb_pred - y_true))
    bb_rmse = np.sqrt(np.mean((bb_pred - y_true)**2))

    print("\n" + "="*40)
    print("           RESULTS SUMMARY")
    print("="*40)
    print(f"PINN Model      - MAE: {pinn_mae:.4f} LD  |  RMSE: {pinn_rmse:.4f} LD")
    print(f"Black Box Model - MAE: {bb_mae:.4f} LD  |  RMSE: {bb_rmse:.4f} LD")
    print("="*40)

    # wizualizacja
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['MAE (Lower is better)', 'RMSE (Lower is better)']
    pinn_scores = [pinn_mae, pinn_rmse]
    bb_scores = [bb_mae, bb_rmse]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, pinn_scores, width, label='PINN', color='royalblue', edgecolor='black')
    ax1.bar(x + width/2, bb_scores, width, label='Black Box', color='tomato', edgecolor='black')
    ax1.set_ylabel('Błąd [Lunar Distances]')
    ax1.set_title('Porównanie skuteczności modeli (Zbiór testowy)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    sample_size = min(1000, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    
    ax2.scatter(y_true[indices], pinn_pred[indices], alpha=0.5, label='PINN', color='royalblue', s=15)
    ax2.scatter(y_true[indices], bb_pred[indices], alpha=0.5, label='Black Box', color='tomato', s=15)
    
    min_val = min(np.min(y_true[indices]), np.min(pinn_pred[indices]), np.min(bb_pred[indices]))
    max_val = max(np.max(y_true[indices]), np.max(pinn_pred[indices]), np.max(bb_pred[indices]))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Idealna predykcja')
    
    ax2.set_xlabel('Prawdziwe MOID [LD]')
    ax2.set_ylabel('Przewidziane MOID [LD]')
    ax2.set_title(f'Predykcja vs Prawda (Próbka {sample_size} obiektów)')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Biblioteka 'matplotlib' nie jest zainstalowana. Uruchom: pip install matplotlib")
        sys.exit(1)
        
    main()
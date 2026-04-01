import os
import random
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from model import NeoKeplerPINN, kepler_pinn_loss
from visualizer import visualize_animated_neos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'best_neo_pinn_model.pth')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data():
    """Ładuje dane z CSV (lub mock) i przygotowuje tensory."""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    csv_path = os.path.join(BASE_DIR, 'data', 'NEO_Curated.csv')
    
    if not os.path.exists(csv_path):
        print(f"WARN: File not found {csv_path}. Loading mock data...")
        np.random.seed(42)
        mock_data = {
            'full_name': [f"Asteroid-{i}" for i in range(1000)],
            'H': np.random.uniform(15, 25, 1000), 'e': np.random.uniform(0.1, 0.9, 1000),
            'a': np.random.uniform(0.8, 3.0, 1000), 'i': np.random.uniform(0, 30, 1000),
            'om': np.random.uniform(0, 360, 1000), 'w': np.random.uniform(0, 360, 1000),
            'ma': np.random.uniform(0, 360, 1000), 'n': np.random.uniform(0.1, 1.0, 1000),
            'epoch': np.full(1000, 2459000.5),
            'moid_ld': np.random.uniform(0.001, 0.5, 1000)
        }
        df = pd.DataFrame(mock_data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path, low_memory=False)

    df['display_name'] = df['full_name'].fillna("Unknown object").str.strip() if 'full_name' in df.columns else "NEO"
    df['i_rad'], df['om_rad'], df['w_rad'] = np.radians(df['i']), np.radians(df['om']), np.radians(df['w'])
    # Transformacja kątów na sin/cos: zapobiega problemowi "skoku" (np. 359st i 1st są blisko, ale numerycznie dzieli je duża różnica). Sin/Cos zapewnia ciągłość matematyczną.
    df['i_sin'], df['i_cos'] = np.sin(df['i_rad']), np.cos(df['i_rad'])
    df['om_sin'], df['om_cos'] = np.sin(df['om_rad']), np.cos(df['om_rad'])
    df['w_sin'], df['w_cos'] = np.sin(df['w_rad']), np.cos(df['w_rad'])

    df = df[(df['H'] > -1) & (df['a'] > 0)].copy()
    # Logarytmowanie H (jasność) i a (półoś wielka): H i a często mają rozkład długoogonowy.
    # log1p (log(1+x)) stabilizuje proces uczenia i zmniejsza wpływ ekstremalnych wartości.
    df['H_log'], df['a_log'] = np.log1p(df['H']), np.log1p(df['a'])

    features_cols = ['H_log', 'e', 'a_log', 'i_sin', 'i_cos', 'om_sin', 'om_cos', 'w_sin', 'w_cos']
    target_cols = ['moid_ld'] 
    
    required_cols = features_cols + target_cols + ['a', 'i', 'om', 'w', 'ma', 'n', 'epoch']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])

    X = df[features_cols].values
    y = df[target_cols].values
    indices = df.index.values 

    scaler_X = StandardScaler()
    # Użycie RobustScaler dla celu (y): Odporny na outliery (bardzo odległe obiekty), 
    # co pozwala modelowi skupić się na "normalnym" zakresie odległości.
    scaler_y = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(X_scaled, y_scaled, indices, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(X_temp, y_temp, idx_temp, test_size=0.5, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    X_train_real = scaler_X.inverse_transform(X_train)
    e_train_real = torch.tensor(X_train_real[:, 1], dtype=torch.float32, device=device)
    a_train_real = torch.tensor(np.expm1(X_train_real[:, 2]), dtype=torch.float32, device=device) 
    
    X_val_real = scaler_X.inverse_transform(X_val)
    e_val_real = torch.tensor(X_val_real[:, 1], dtype=torch.float32, device=device)
    a_val_real = torch.tensor(np.expm1(X_val_real[:, 2]), dtype=torch.float32, device=device)

    return {
        'device': device, 'df': df, 'features_cols': features_cols,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'train': (X_train_t, y_train_t, a_train_real, e_train_real),
        'val': (X_val_t, y_val_t, a_val_real, e_val_real),
        'test': (X_test_t, y_test, idx_test)
    }

def train_model(data, epochs=5000, patience=30):
    """Trenuje model PINN i zapisuje najlepsze wagi."""
    device = data['device']
    X_train_t, y_train_t, a_train_real, e_train_real = data['train']
    X_val_t, y_val_t, a_val_real, e_val_real = data['val']
    scaler_y = data['scaler_y']

    model = NeoKeplerPINN(input_size=len(data['features_cols'])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n--- PINN Training Start on {device} ---")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_t)
        loss_train, _, _ = kepler_pinn_loss(y_pred_train, y_train_t, a_train_real, e_train_real, scaler_y, lambda_physics=0.5)
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_t)
            loss_val, _, _ = kepler_pinn_loss(y_pred_val, y_val_t, a_val_real, e_val_real, scaler_y, lambda_physics=0.5)
        
        current_val_loss = loss_val.item()
        scheduler.step(current_val_loss)
        
        if current_val_loss < best_val_loss - 0.0001:
            best_val_loss = current_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch: {epoch+1}!")
            break 
            
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                y_pred_val_real = scaler_y.inverse_transform(y_pred_val[:, [0]].cpu().numpy())
                y_true_val_real = scaler_y.inverse_transform(y_val_t.cpu().numpy())
                mae_moid = np.mean(np.abs(y_pred_val_real[:, 0] - y_true_val_real[:, 0]))
            print(f"Epoch [{epoch+1}/{epochs}] | Val Loss: {current_val_loss:.4f} | MOID Error (MAE): {mae_moid:.2f} LD")
    
    print("Training Complete! Model saved.")

def evaluate_model(data):
    """Ładuje model i ewaluuje go na zbiorze testowym."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Error: Model not found. Please train it first.")
        return

    device = data['device']
    X_test_t, y_test, _ = data['test']
    scaler_y = data['scaler_y']

    model = NeoKeplerPINN(input_size=len(data['features_cols'])).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    with torch.no_grad():
        y_pred_test = model(X_test_t)
        y_pred_real_moid = scaler_y.inverse_transform(y_pred_test[:, [0]].cpu().numpy())
        y_true_test_real_moid = scaler_y.inverse_transform(y_test)
        
        final_mae = np.mean(np.abs(y_pred_real_moid[:, 0] - y_true_test_real_moid[:, 0]))
        final_rmse = np.sqrt(np.mean((y_pred_real_moid[:, 0] - y_true_test_real_moid[:, 0])**2))
        
        print(f"\n--- Final metrics (Test) ---")
        print(f"Mean Absolute Error (MAE): {final_mae:.2f} Lunar Distances (LD)")
        print(f"Root Mean Square Error (RMSE): {final_rmse:.2f} Lunar Distances (LD)\n")

def generate_visualization(data, top_n=10):
    """Generuje animację Plotly dla top N najgroźniejszych obiektów z test setu."""
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Error: Model not found. Please train it first.")
        return

    device, df = data['device'], data['df']
    X_test_t, _, idx_test = data['test']
    scaler_y = data['scaler_y']

    model = NeoKeplerPINN(input_size=len(data['features_cols'])).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    with torch.no_grad():
        y_pred_test = model(X_test_t)
        y_pred_real_moid = scaler_y.inverse_transform(y_pred_test[:, [0]].cpu().numpy())

    sorted_indices = np.argsort(np.abs(y_pred_real_moid[:, 0]))
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]
    
    neo_plot_list = []
    for idx in top_indices:
        row = df.loc[idx_test[idx]]
        moid_val = np.abs(y_pred_real_moid[idx, 0])
        neo_plot_list.append({
            'name': f"{row['display_name']} (MOID: {moid_val:.2f} LD)",
            'params': [row['a'], row['e'], row['i'], row['om'], row['w']],
            'time_params': {'ma': row.get('ma', 0.0), 'n': row.get('n', 0.0), 'epoch': row.get('epoch', 2459000.5)},
            'moid': moid_val
        })

    print(f"\nRendering Plotly animation for {len(neo_plot_list)} NEOs...")
    visualize_animated_neos(neo_plot_list, epoch_start=2459000.5, days_to_simulate=365, frames_count=180)

def predict_single(data, name_query):
    """Przewiduje MOID dla konkretnego wpisanego obiektu z całego DF."""
    df = data['df']
    # Wyszukujemy obiekt po nazwie (case-insensitive)
    matches = df[df['display_name'].str.contains(name_query, case=False, na=False)]
    
    if matches.empty:
        print(f"No object found matching '{name_query}'.")
        return
    
    row = matches.iloc[0]
    print(f"\nFound object: {row['display_name']}")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print("Error: Model not found. Train the model first to make predictions.")
        return

    # Skalujemy cechy tego jednego obiektu
    features = row[data['features_cols']].values.reshape(1, -1)
    X_scaled = data['scaler_X'].transform(features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=data['device'])

    model = NeoKeplerPINN(input_size=len(data['features_cols'])).to(data['device'])
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    with torch.no_grad():
        y_pred = model(X_tensor)
        predicted_moid_ld = data['scaler_y'].inverse_transform(y_pred[:, [0]].cpu().numpy())[0, 0]
    
    print(f"Predicted MOID: {predicted_moid_ld:.2f} Lunar Distances (LD)")
    print(f"Actual MOID (from database): {row['moid_ld']:.2f} LD")
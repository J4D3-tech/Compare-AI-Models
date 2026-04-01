import torch
import torch.nn as nn
import torch.nn.functional as F # funkcje bezstanowe

class NeoKeplerPINN(nn.Module):
    def __init__(self, input_size):
        super(NeoKeplerPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # wyjscie ma 2 neurony: 
            # [0] to skalowany MOID (uczony z danych) (Skalowany MOID (regresja z danych))
            # [1] to q (uczony TYLKO z praw fizyki) (Parametr q (peryhelium), wymuszany przez fizykę w funkcji straty)
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        return self.net(x)

def kepler_pinn_loss(y_pred, y_true_scaled, a_real, e_real, scaler_y, lambda_physics=0.5):
    
    # y_pred ma kształt [N, 2], y_true_scaled ma teraz kształt [N, 1] (zawiera tylko MOID)
    #Funkcja straty typu PINN (Physics-Informed Neural Network).
    #Łączy błąd średniokwadratowy (MSE) z ograniczeniami fizycznymi.
    
    moid_pred_scaled = y_pred[:, 0]
    q_pred_real = y_pred[:, 1] # Sieć bezpośrednio wypluwa q w prawdziwych jednostkach
    
    moid_true_scaled = y_true_scaled[:, 0]
    
    #  DANGER-WEIGHTED LOSS Wyższa kara za błędy dla obiektów o małym MOID (bliskich Ziemi)
    danger_weights = torch.exp(-moid_true_scaled).clamp(max=10.0)
    mse_loss_per_element = (moid_pred_scaled - moid_true_scaled) ** 2
    mse_weighted = torch.mean(mse_loss_per_element * danger_weights)
    
    # KARA FIZYCZNA Weryfikacja czy wyjście q sieci zgadza się z prawem Keplera: q = a(1-e)
    # To działa jak zaawansowana regularyzacja, która "pilnuje" modelu.
    q_physics_ideal = a_real * (1.0 - e_real)
    
    # Zoptymalizowane wyliczenie MSELoss
    physics_penalty = F.mse_loss(q_pred_real, q_physics_ideal)
    
    # Całkowita strata
    total_loss = mse_weighted + lambda_physics * physics_penalty
    return total_loss, mse_weighted, physics_penalty
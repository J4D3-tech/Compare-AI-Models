# For the project to work as intended you need to download https://drive.google.com/file/d/1qZbRrY0va7xsfj1w-kk6C03Vo4LoD4ef/view?usp=drive_link and put the NEO_Curated.csv in data/ folder
# Quick Start Guide

1. **Install dependencies:** Ensure you have all required libraries installed:  
   `pip install torch pandas numpy scikit-learn matplotlib plotly`

2. **Train the models:** Run `app.py` to train the PINN model and `BB_app.py` for the Black Box baseline.

3. **Verify saved models:** Ensure that both training processes finished and the `.pth` files are present in the `models/` directory.

4. **Compare results:** Run `compare.py` to generate a detailed visual comparison and performance metrics (MAE, RMSE) for both approaches.

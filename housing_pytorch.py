import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np

# Chargement du fichier local
data = pd.read_csv('housing.csv')

columns_to_keep = ['median_income', 'housing_median_age', 'total_rooms']
target_column = 'median_house_value'

data_clean = data[columns_to_keep + [target_column]].dropna()

X = data_clean[columns_to_keep]
y = data_clean[[target_column]]

# 3. Normalisation Min-Max par BROADCASTING
X_min = X.min()
X_max = X.max()
X_norm = (X - X_min) / (X_max - X_min)

print("Statistiques Min/Max de X_norm :")
print(X_norm.describe().loc[['min', 'max']])

# ==================================
# 1. Préparation des Tensors PyTorch
# ==================================
X_tensor = torch.tensor(X_norm.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# ======================================
# 2. Définition de l'architecture du MLP
# ======================================
class PricePredictorMLP(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictorMLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 16)  # Couche cachée 1
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 8)          # Couche cachée 2
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(8, 1)            # Couche de sortie

    def forward(self, x):
        x = self.relu1(self.hidden1(x))
        x = self.relu2(self.hidden2(x))
        x = self.output(x)
        return x

model = PricePredictorMLP(input_dim=X_tensor.shape[1])

# ==================================
# 3. Fonction de Coût (Loss) et Optimiseur
# =====================================
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# ======================
#  Boucle d'entraînement
# ======================
epochs = 100
loss_history = []

print("--- Début de l'entraînement ---")
for epoch in range(epochs):
    model.train()
    
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    optimizer.zero_grad() 
    loss.backward()       
    optimizer.step()      
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Époque [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =================================
#  Visualisation avec Matplotlib (Sécurisée)
# =================================
model.eval()
with torch.no_grad():
    final_predictions = model(X_tensor).numpy()

# Sécurité Windows : Sous-échantillonnage de 500 points aléatoires
np.random.seed(42)
indices = np.random.choice(len(y), size=500, replace=False)
y_sub = y.values[indices]
preds_sub = final_predictions[indices]

plt.figure(figsize=(12, 5))

# Graphique 1 : Courbe de Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue', label='Train Loss')
plt.title('Évolution de la Perte (Loss)')
plt.xlabel('Époques')
plt.ylabel('MSE')
plt.grid(True)
plt.legend()

# Graphique 2 : Nuage de points optimisé
plt.subplot(1, 2, 2)
plt.scatter(y_sub, preds_sub, alpha=0.6, color='orange', edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title('Prix Réels vs Prix Prédits (Échantillon 500 pts)')
plt.xlabel('Prix Réels')
plt.ylabel('Prix Prédits')
plt.grid(True)

plt.tight_layout()
# Sauvegarde locale immédiate
plt.savefig('performances_mlp.png', dpi=300)
print("\n--> [SUCCÈS] Graphique sauvegardé sous 'performances_mlp.png'")

# Force la fenêtre à s'ouvrir et bloque le script pour vous laisser observer
plt.show(block=True)

# =====================================================================
# ÉTAPE 3 : CRÉATION DE L'INTERFACE INTERACTIVE GRADIO
# =====================================================================
print("\n3. Configuration et lancement de l'interface Gradio...")

X_min_np = X_min.values
X_max_np = X_max.values

def predict_real_estate(income, age, rooms):
    raw_inputs = np.array([income, age, rooms], dtype=np.float32)
    norm_inputs = (raw_inputs - X_min_np) / (X_max_np - X_min_np)
    tensor_inputs = torch.tensor(norm_inputs, dtype=torch.float32).unsqueeze(0) 
    
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_inputs).item()
        
    return f"${prediction:,.2f}"

# ORDRE STRICT DES SLIDERS : income -> age -> rooms
demo = gr.Interface(
    fn=predict_real_estate,
    inputs=[
        gr.Slider(minimum=float(X_min['median_income']), maximum=float(X_max['median_income']), value=5.0, label="Revenu Médian du Quartier (en dizaines de milliers $)"),
        gr.Slider(minimum=float(X_min['housing_median_age']), maximum=float(X_max['housing_median_age']), value=25.0, label="Âge Médian de la Maison"),
        gr.Slider(minimum=float(X_min['total_rooms']), maximum=float(X_max['total_rooms'] / 10), value=500.0, label="Nombre Total de Pièces (Échantillon)")
    ],
    outputs=gr.Textbox(label="Prix Médian Prédit de la Propriété"),
    title="Prédicteur de Prix Immobilier Interactif (MLP)",
    description="Bougez les sliders pour voir l'impact en temps réel sur la prédiction."
)

demo.launch()
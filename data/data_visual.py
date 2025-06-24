import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

proplist = [
    'HeavyAtomCount',
    'NumHAcceptors',
    'NumHDonors',
    'NumHeteroatoms',
    'NumRotatableBonds',
    'NumValenceElectrons',
    'NumAromaticRings',
    'NumSaturatedRings',
    'NumAliphaticRings',
    'RingCount'
]

sol = pd.read_csv("data/solubility-dataset.csv")
sol["NumHAcceptors"] = sol["NumHAcceptors"] + 1 # Add 1 to avoid division by zero

# Divide featrues by molecular weight to get molar properties
X = np.array([list(sol[prop] / sol['MolWt']) for prop in proplist]) 
X = np.insert(X, 0, list(np.log(sol['MolWt'])), axis=0) 
Y = np.array(sol['Solubility']); 

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(Y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Solubility Values')
ax1.set_xlabel('Solubility')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(sol['MolWt'], Y, alpha=0.6, color='coral')
ax2.set_title('Molecular Weight vs Solubility')
ax2.set_xlabel('Molecular Weight')
ax2.set_ylabel('Solubility')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[2, 0])
feature_names = ['Log_MolWt'] + proplist
correlation_matrix = np.corrcoef(X)
im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
ax3.set_title('Feature Correlation Heatmap')
ax3.set_xticks(range(len(feature_names)))
ax3.set_yticks(range(len(feature_names)))
ax3.set_xticklabels(feature_names, rotation=45, ha='right')
ax3.set_yticklabels(feature_names)
plt.colorbar(im, ax=ax3)

ax4 = fig.add_subplot(gs[:, 1])
key_props = ['HeavyAtomCount', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds']
box_data = [sol[prop] for prop in key_props]
bp = ax4.boxplot(box_data, labels=key_props, patch_artist=True)
ax4.set_title('Boxplots of Some Molecular Properties')
ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-5, 200)


colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout()
plt.show()

print("\n=== Dataset Summary ===")
print(f"Total samples: {len(Y)}")
print(f"Solubility range: {Y.min():.3f} to {Y.max():.3f}")
print(f"Mean solubility: {Y.mean():.3f}")
print(f"Molecular weight range: {sol['MolWt'].min():.1f} to {sol['MolWt'].max():.1f}")
print(f"Mean molecular weight: {sol['MolWt'].mean():.1f}")

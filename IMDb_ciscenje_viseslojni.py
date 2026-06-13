import os
import pandas as pd
import numpy as np
import re

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
#%%

podaci = pd.read_csv("IMDB Dataset.csv", header=0)

vektori = "vectors.txt"

podaci['ocena'] = (podaci['sentiment'] == 'positive').astype(int)

deo_za_trening = 0.8
#%%
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()

def load_word_vectors(filename):

    word_vectors = {}
    word2id = {}
    id2word = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        
        first_line = f.readline().strip()
        vocab_size, embedding_dim = map(int, first_line.split())
        
        for idx, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            
            word_vectors[word] = vector
            word2id[word] = idx
            id2word[idx] = word
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2id.items():
        embedding_matrix[idx] = word_vectors[word]
    
    return {
        'word_vectors': word_vectors,     
        'embedding_matrix': embedding_matrix, 
        'word2id': word2id,               
        'id2word': id2word,               
        'embedding_dim': embedding_dim,
        'vocab_size': vocab_size
    }

#%%
totalni_broj_tokena = 0
totalni_unc = 0
#%%
podaci['recenzija_tokeni'] = podaci['review'].apply(clean_and_tokenize)
vektorisane_reci = load_word_vectors(vektori)
#%%
def filter_and_replace_unknown_words(tokens):
    tokeni_izlaz = []
    global totalni_unc,totalni_broj_tokena
    for token in tokens:
        if token in vektorisane_reci['word2id'].keys():
            tokeni_izlaz.append(token)#ovo zakomentarisem
        else:
            totalni_unc += 1
            tokeni_izlaz.append('UNK')
        totalni_broj_tokena += 1
    
    return tokeni_izlaz


podaci['filterisano'] = podaci['recenzija_tokeni'].apply(filter_and_replace_unknown_words)

#%%

gener = np.random.default_rng(14)

podaci_pozitivni = podaci.loc[podaci['ocena'] == 1,:]
podaci_negativni = podaci.loc[podaci['ocena'] == 0,:]

podaci_pozitvni_trening = podaci_pozitivni.sample(frac = deo_za_trening, random_state = gener.bit_generator) 
podaci_negativni_trening = podaci_negativni.sample(frac = deo_za_trening,  random_state = gener.bit_generator) 

podaci_pozitvni_validacija = podaci_pozitivni.drop(index = podaci_pozitvni_trening.index)
podaci_negativni_validacija = podaci_negativni.drop(index = podaci_negativni_trening.index)


podaci_trening = pd.concat([podaci_pozitvni_trening, podaci_negativni_trening], ignore_index=True)[[
        "filterisano",
        "ocena"
    ]]
podaci_validacija = pd.concat([podaci_pozitvni_validacija, podaci_negativni_validacija], ignore_index=True)[[
        "filterisano",
        "ocena"
    ]]

podaci_trening = podaci_trening.sample(frac = 1, random_state= 14).reset_index(drop = True)
podaci_validacija = podaci_validacija.sample(frac = 1, random_state= 14).reset_index(drop = True)
print(f"U treningu je ukupno: {podaci_trening["ocena"].count()}, od toga pozitivno: {podaci_trening["ocena"].sum()}")
print(f"U validaciji je ukupno: {podaci_validacija["ocena"].count()}, od toga pozitivno: {podaci_validacija["ocena"].sum()}")

#%%


def prebaci_tokene_u_srednji_vektor(tokeni):
    pocetni_vektor = np.zeros(shape= (vektorisane_reci['embedding_dim'], ) )
    
    for token in tokeni:
        pocetni_vektor += vektorisane_reci['word_vectors'][token]

    return pocetni_vektor / len(tokeni)


podaci_ulaz_trening = podaci_trening['filterisano'].apply(prebaci_tokene_u_srednji_vektor)
podaci_ulaz_validacija = podaci_validacija['filterisano'].apply(prebaci_tokene_u_srednji_vektor)
podaci_izlaz_trening = podaci_trening.get("ocena")
podaci_izlaz_validacija = podaci_validacija.get("ocena")

#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#%%
def generisi_batches_ff(podaci_df, word_vectors, embedding_dim=200, batch_size=32):
    nula_vektor = np.zeros(embedding_dim, dtype=np.float32)
    broj_redova = podaci_df.shape[0]

    while True:
        for start in range(0, broj_redova, batch_size):
            end = min(start + batch_size, broj_redova)
            batch_podaci = podaci_df.iloc[start:end]

            batch_x = []
            batch_y = []

            for _, red in batch_podaci.iterrows():
                tokeni = red['filterisano']
                ocena = red['ocena']

                if len(tokeni) > 0:
                    vektori_tokena = [word_vectors.get(t, nula_vektor) for t in tokeni]
                    avg_vektor = np.mean(vektori_tokena, axis=0).astype(np.float32)
                else:
                    avg_vektor = nula_vektor

                batch_x.append(avg_vektor)
                batch_y.append([1, 0] if ocena == 0 else [0, 1])

            yield np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

BATCH_SIZE = 32
EMB_DIM = vektorisane_reci['embedding_dim']
steps_per_epoch = len(podaci_trening) // BATCH_SIZE
validation_steps = len(podaci_validacija) // BATCH_SIZE

#%%
# Model 1: Smanjujuca arhitektura (128 -> 64 -> 2)
model1 = Sequential([
    Dense(128, activation='relu', input_shape=(EMB_DIM,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax') 
], name='Model_1_smanjujuci')

model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.F1Score(name='f1_score', average='micro')
    ]
)
model1.summary()

#%%
# Model 2: Luk arhitektura (128 -> 256 -> 128 -> 2)
model2 = Sequential([
    Dense(128, activation='relu', input_shape=(EMB_DIM,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
], name='Model_2_luk')

model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.F1Score(name='f1_score', average='micro')
    ]
)
model2.summary()

#%%
# Model 3: Piramidalna arhitektura (256 -> 128 -> 64 -> 32 -> 2)
model3 = Sequential([
    Dense(256, activation='relu', input_shape=(EMB_DIM,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
], name='Model_3_piramida')

model3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.F1Score(name='f1_score', average='micro')
    ]
)
model3.summary()

#%%
# Cuvanje arhitekture modela
tf.keras.utils.plot_model(model1, to_file=os.path.join(SCRIPT_DIR, 'arhitektura_model1.png'), show_shapes=True, show_layer_activations=True, dpi=150)
tf.keras.utils.plot_model(model2, to_file=os.path.join(SCRIPT_DIR, 'arhitektura_model2.png'), show_shapes=True, show_layer_activations=True, dpi=150)
tf.keras.utils.plot_model(model3, to_file=os.path.join(SCRIPT_DIR, 'arhitektura_model3.png'), show_shapes=True, show_layer_activations=True, dpi=150)

#%%
print(" Trening Model 1: 128 -> 64 -> 2 ")
istorija1 = model1.fit(
    generisi_batches_ff(podaci_trening, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=generisi_batches_ff(podaci_validacija, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    validation_steps=validation_steps,
    verbose=1
)

#%%
print(" Trening Model 2: 128 -> 256 -> 128 -> 2 ")
istorija2 = model2.fit(
    generisi_batches_ff(podaci_trening, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=generisi_batches_ff(podaci_validacija, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    validation_steps=validation_steps,
    verbose=1
)

#%%
print(" Trening Model 3: 256 -> 128 -> 64 -> 32 -> 2 ")
istorija3 = model3.fit(
    generisi_batches_ff(podaci_trening, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=generisi_batches_ff(podaci_validacija, vektorisane_reci['word_vectors'], EMB_DIM, BATCH_SIZE),
    validation_steps=validation_steps,
    verbose=1
)

#%%
# Grafik 1: accuracy i loss
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Feed-Forward neuronska mreza - poredenje arhitektura', fontsize=14, fontweight='bold')

axes[0, 0].plot(istorija1.history['accuracy'], label='Trening', color='blue')
axes[0, 0].plot(istorija1.history['val_accuracy'], label='Validacija', color='orange')
axes[0, 0].set_title('Model 1 - Tacnost')
axes[0, 0].set_xlabel('Epoha')
axes[0, 0].set_ylabel('Tacnost')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(istorija1.history['loss'], label='Trening', color='blue')
axes[0, 1].plot(istorija1.history['val_loss'], label='Validacija', color='orange')
axes[0, 1].set_title('Model 1 - Greska')
axes[0, 1].set_xlabel('Epoha')
axes[0, 1].set_ylabel('Greska')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(istorija2.history['accuracy'], label='Trening', color='green')
axes[1, 0].plot(istorija2.history['val_accuracy'], label='Validacija', color='red')
axes[1, 0].set_title('Model 2 - Tacnost')
axes[1, 0].set_xlabel('Epoha')
axes[1, 0].set_ylabel('Tacnost')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(istorija2.history['loss'], label='Trening', color='green')
axes[1, 1].plot(istorija2.history['val_loss'], label='Validacija', color='red')
axes[1, 1].set_title('Model 2 - Greska')
axes[1, 1].set_xlabel('Epoha')
axes[1, 1].set_ylabel('Greska')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[2, 0].plot(istorija3.history['accuracy'], label='Trening', color='purple')
axes[2, 0].plot(istorija3.history['val_accuracy'], label='Validacija', color='brown')
axes[2, 0].set_title('Model 3 - Tacnost')
axes[2, 0].set_xlabel('Epoha')
axes[2, 0].set_ylabel('Tacnost')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(istorija3.history['loss'], label='Trening', color='purple')
axes[2, 1].plot(istorija3.history['val_loss'], label='Validacija', color='brown')
axes[2, 1].set_title('Model 3 - Greska')
axes[2, 1].set_xlabel('Epoha')
axes[2, 1].set_ylabel('Greska')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modeli_accuracy_loss.png', dpi=150, bbox_inches='tight')
#plt.show()

#%%
# Grafik 2: Poredjenje validacione tacnosti
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(istorija1.history['val_accuracy'], label='Model 1', marker='o')
ax2.plot(istorija2.history['val_accuracy'], label='Model 2', marker='s')
ax2.plot(istorija3.history['val_accuracy'], label='Model 3', marker='^')
ax2.set_title('Poredjenje validacione tacnosti')
ax2.set_xlabel('Epoha')
ax2.set_ylabel('Validaciona tacnost')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('poredjenje_validacije.png', dpi=150, bbox_inches='tight')
#plt.show()

#%%
# Grafik 3: Poredjenje validacionog gubitka
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(istorija1.history['val_loss'], label='Model 1', marker='o')
ax3.plot(istorija2.history['val_loss'], label='Model 2', marker='s')
ax3.plot(istorija3.history['val_loss'], label='Model 3', marker='^')
ax3.set_title('Poredjenje validacionog gubitka')
ax3.set_xlabel('Epoha')
ax3.set_ylabel('Validacioni gubitak')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('poredjenje_validacionog_gubitka.png', dpi=150, bbox_inches='tight')
#plt.show()

#%%
# Grafik 4: F1
f1_model1 = np.mean(istorija1.history['val_f1_score'][-1])
f1_model2 = np.mean(istorija2.history['val_f1_score'][-1])
f1_model3 = np.mean(istorija3.history['val_f1_score'][-1])

fig4, ax4 = plt.subplots(figsize=(8, 5))
modeli_nazivi = ['Model 1', 'Model 2', 'Model 3']
f1_vrednosti = [f1_model1, f1_model2, f1_model3]
boje = ['steelblue', 'darkorange', 'mediumseagreen']

stubici = ax4.bar(modeli_nazivi, f1_vrednosti, color=boje, width=0.5)
ax4.set_title('Poredjenje F1 metrike - poslednja epoha')
ax4.set_ylabel('F1 Score')
ax4.set_ylim(0, 1)
ax4.grid(True, alpha=0.3, axis='y')

for stub, vrednost in zip(stubici, f1_vrednosti):
    ax4.text(stub.get_x() + stub.get_width() / 2, stub.get_height() + 0.01,
             f'{vrednost:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('poredjenje_f1.png', dpi=150, bbox_inches='tight')
#plt.show()

#%%
# Grafik 5: Tabela sa parametrima modela
fig5, ax5 = plt.subplots(figsize=(12, 3))
ax5.axis('off')

kolone = ['Model', 'Arhitektura', 'Tip', 'Dropout', 'Skriveni slojevi', 'Max val. tacnost', 'Max trening tacnost']
redovi = [
    ['Model 1', '128->64->2', 'Smanjujuca', '0.3 (2x)', '2',
     f"{max(istorija1.history['val_accuracy']):.4f}",
     f"{max(istorija1.history['accuracy']):.4f}"],
    ['Model 2', '128->256->128->2', 'Luk (gore-dole)', '0.3 (3x)', '3',
     f"{max(istorija2.history['val_accuracy']):.4f}",
     f"{max(istorija2.history['accuracy']):.4f}"],
    ['Model 3', '256->128->64->32->2', 'Dublja piramida', '0.3 (4x)', '4',
     f"{max(istorija3.history['val_accuracy']):.4f}",
     f"{max(istorija3.history['accuracy']):.4f}"],
]

tabela = ax5.table(cellText=redovi, colLabels=kolone, loc='center', cellLoc='center')
tabela.auto_set_font_size(False)
tabela.set_fontsize(9)
tabela.scale(1, 2)
ax5.set_title('Tabela parametara i rezultata modela', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('tabela_rezultata.png', dpi=150, bbox_inches='tight')
#plt.show()

print("\n REZULTATI ")
print(f"Model 1: max val. tacnost = {max(istorija1.history['val_accuracy']):.4f}")
print(f"Model 2: max val. tacnost = {max(istorija2.history['val_accuracy']):.4f}")
print(f"Model 3: max val. tacnost = {max(istorija3.history['val_accuracy']):.4f}")

#%%
# Sve metrike za sve epohe
istorija1_df = pd.DataFrame(istorija1.history)
istorija1_df.index.name = 'epoha'
istorija1_df.to_csv('istorija_model1.csv')

istorija2_df = pd.DataFrame(istorija2.history)
istorija2_df.index.name = 'epoha'
istorija2_df.to_csv('istorija_model2.csv')

istorija3_df = pd.DataFrame(istorija3.history)
istorija3_df.index.name = 'epoha'
istorija3_df.to_csv('istorija_model3.csv')

print("Istorija sacuvana")
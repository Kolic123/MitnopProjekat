import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

ime_fajla = "vectors.txt"


df = pd.read_csv(ime_fajla, sep=r'\s+', skiprows=[0], header=None, encoding="utf-8", dtype=str)

reci = df[0].astype(str).tolist()
ociscene_reci = []

for r in reci:
    ociscene_reci.append(r)

reci = ociscene_reci


matrica_vektora = df.iloc[:, 1:].to_numpy(dtype=np.float32)

validne = ~np.isnan(matrica_vektora).any(axis=1)
reci = [r for r, v in zip(reci, validne) if v]
matrica_vektora = matrica_vektora[validne]

recnik_vektora = {}
rec_to_idx = {}


for i, rec in enumerate(reci):
    recnik_vektora[rec] = matrica_vektora[i]

    rec_to_idx[rec] = i

print(f"Uspesno ucitano {len(recnik_vektora)} reci")


# Normalizacija vektora + PCA


reci = list(recnik_vektora.keys())
matrica_vektora = np.array(list(recnik_vektora.values()))

norme = np.linalg.norm(matrica_vektora, axis=1, keepdims=True)
matrica_vektora = matrica_vektora / (norme + 1e-10)
pca = PCA(n_components=2, random_state=42)
pca.fit(matrica_vektora)

pca3 = PCA(n_components=3, random_state=42)
pca3.fit(matrica_vektora)

with open("varijansa.txt", "w", encoding="utf-8") as f:
    f.write("PCA 2D varijansa: " + str(pca.explained_variance_ratio_) + "\n")
    f.write("PCA 3D varijansa: " + str(pca3.explained_variance_ratio_) + "\n")

rezultati_analogija = []


# Analogija u 200D prostoru i redukcija na 2D i 3D


potrebne_reci = ["king", "queen", "man", "woman"]
nedostaju = [r for r in potrebne_reci if r not in recnik_vektora]

if nedostaju:
    print(f"Greska: u recniku nema: {nedostaju}")
else:
    # Sabiranje/oduzimanje u 200D prostoru
    king  = matrica_vektora[rec_to_idx["king"]]
    queen = matrica_vektora[rec_to_idx["queen"]]
    man   = matrica_vektora[rec_to_idx["man"]]

    rezultat_200d = queen - king + man

    slicnosti = matrica_vektora @ rezultat_200d
    for rec in {"king", "queen", "man"}:
        slicnosti[rec_to_idx[rec]] = -1.0
    najbliza = reci[np.argmax(slicnosti)]
    rezultati_analogija.append(f"queen - king + man = '{najbliza}' (ocekivano: 'woman')")

    print(f"\nAnalogija: queen - king + man = ?")
    print(f"Najbliza rec: '{najbliza}' (ocekivano: 'woman')")
    
    recnik_2d = {r: pca.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_2d = pca.transform(rezultat_200d.reshape(1, -1))[0]
    fig, ax = plt.subplots(figsize=(12, 10))

    boje = {
        "king":  "royalblue",
        "queen": "crimson",
        "man":   "forestgreen",
        "woman": "darkorange",
    }
    
    sve_x = [recnik_2d[r][0] for r in potrebne_reci] + [rezultat_2d[0], 0.0]
    sve_y = [recnik_2d[r][1] for r in potrebne_reci] + [rezultat_2d[1], 0.0]
    raspon_x = max(sve_x) - min(sve_x) or 1.0
    raspon_y = max(sve_y) - min(sve_y) or 1.0
    pad = 0.25
    ax.set_xlim(min(sve_x) - pad * raspon_x, max(sve_x) + pad * raspon_x)
    ax.set_ylim(min(sve_y) - pad * raspon_y, max(sve_y) + pad * raspon_y)

    for rec in potrebne_reci:
        x, y = recnik_2d[rec]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=boje[rec], lw=2.5))
        ox = 0.04 * raspon_x * (1 if x >= 0 else -1)
        oy = 0.04 * raspon_y * (1 if y >= 0 else -1)
        ax.text(x + ox, y + oy, rec, fontsize=13, fontweight="bold",
                color=boje[rec], ha="center")

    kx, ky = recnik_2d["king"]
    qx, qy = recnik_2d["queen"]
    mx, my = recnik_2d["man"]
    wx, wy = recnik_2d["woman"]

    ax.annotate("", xy=(qx, qy), xytext=(kx, ky),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))
    ax.annotate("", xy=(wx, wy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))

    _dxkq, _dykq = qx - kx, qy - ky
    _nkq = max(np.sqrt(_dxkq**2 + _dykq**2), 1e-10)
    _dxmw, _dymw = wx - mx, wy - my
    _nmw = max(np.sqrt(_dxmw**2 + _dymw**2), 1e-10)
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _goff = 0.12 * max(raspon_x, raspon_y)
    _xm = 0.12 * (_xlim[1] - _xlim[0])
    _ym = 0.12 * (_ylim[1] - _ylim[0])
    def _in_bounds(x, y):
        return (_xlim[0]+_xm < x < _xlim[1]-_xm and _ylim[0]+_ym < y < _ylim[1]-_ym)
    _xt1 = (kx+qx)/2 + (-_dykq/_nkq)*_goff
    _yt1 = (ky+qy)/2 + (_dxkq/_nkq)*_goff
    if not _in_bounds(_xt1, _yt1):
        _xt1 = (kx+qx)/2 - (-_dykq/_nkq)*_goff
        _yt1 = (ky+qy)/2 - (_dxkq/_nkq)*_goff
    _xt2 = (mx+wx)/2 + (_dymw/_nmw)*_goff
    _yt2 = (my+wy)/2 + (-_dxmw/_nmw)*_goff
    if not _in_bounds(_xt2, _yt2):
        _xt2 = (mx+wx)/2 - (_dymw/_nmw)*_goff
        _yt2 = (my+wy)/2 - (-_dxmw/_nmw)*_goff
    ax.annotate("queen−king",
                xy=((kx+qx)/2, (ky+qy)/2),
                xytext=(_xt1, _yt1),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.annotate("woman−man",
                xy=((mx+wx)/2, (my+wy)/2),
                xytext=(_xt2, _yt2),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_title("Očekivano: queen − king + man = woman",
                 fontsize=14)
    ax.set_xlabel("Prva glavna komponenta")
    ax.set_ylabel("Druga glavna komponenta")
    plt.grid(True, alpha=0.3)
    plt.savefig("queen_king_man_woman1.png", dpi=150, bbox_inches="tight")

    
    # Grafik 2: queen-king+man = rezultat
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    q2d = recnik_2d["queen"]
    k2d = recnik_2d["king"]
    w2d = recnik_2d["woman"]

    # 4 temena cetvorougla 
    P0 = np.array([0.0, 0.0])   
    P1 = q2d.copy()              
    P2 = q2d - k2d               
    P3 = rezultat_2d.copy()      

    sve_x2 = [P0[0], P1[0], P2[0], P3[0], w2d[0]]
    sve_y2 = [P0[1], P1[1], P2[1], P3[1], w2d[1]]
    rx2 = max(sve_x2) - min(sve_x2) or 1.0
    ry2 = max(sve_y2) - min(sve_y2) or 1.0
    pad2 = 0.30
    ax2.set_xlim(min(sve_x2) - pad2 * rx2, max(sve_x2) + pad2 * rx2)
    ax2.set_ylim(min(sve_y2) - pad2 * ry2, max(sve_y2) + pad2 * ry2)

    def nacrtaj_strelu(start, end, color, lw=2.5, ls="solid"):
        ax2.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                     linestyle=ls))

    def oznaci_liniju(start, end, tekst, color, side=1, t=0.5, offset=0.30):
        s, e = np.array(start), np.array(end)
        pos = s + t * (e - s)
        d = e - s
        perp = np.array([-d[1], d[0]])
        norm = np.linalg.norm(perp)
        if norm > 1e-10:
            perp = perp / norm * offset * ry2 * side
        ax2.text(pos[0] + perp[0], pos[1] + perp[1], tekst,
                 fontsize=11, color=color, ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color,
                           alpha=1.0, lw=1.5))

    # 3 poznate strane cetvorougla
    nacrtaj_strelu(P0, P1, "crimson")
    oznaci_liniju(P0, P1, "queen", "crimson", side=1, t=0.5, offset=0.15)

    nacrtaj_strelu(P1, P2, "royalblue")
    oznaci_liniju(P1, P2, "−king", "royalblue", side=-1, t=0.2, offset=0.30)

    nacrtaj_strelu(P2, P3, "forestgreen")
    oznaci_liniju(P2, P3, "+man", "forestgreen", side=1, t=0.2, offset=0.30)

    # Nedostajuca 4. strana (0,0) do rezultata (isprekidana)
    nacrtaj_strelu(P0, P3, "darkorange", lw=2.8, ls="dashed")
    oznaci_liniju(P0, P3, f"queen−king+man = {najbliza}", "darkorange", side=-1, t=0.4, offset=0.18)

    
    cx = (min(sve_x2) + max(sve_x2)) / 2
    cy = (min(sve_y2) + max(sve_y2)) / 2
    for pt, col, sz in [
        (P0, "black", 100),
        (P1, "crimson", 120),
        (P2, "slategray", 100),
        (P3, "darkorange", 150),
    ]:
        ax2.scatter(*pt, color=col, s=sz, zorder=5)


    ax2.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Najbliza rec: queen − king + man = '{najbliza}'\n",
        fontsize=13
    )
    ax2.set_xlabel("Prva glavna komponenta")
    ax2.set_ylabel("Druga glavna komponenta")
    ax2.grid(True, alpha=0.3)
    plt.savefig("queen_king_man_woman2.png", dpi=150, bbox_inches="tight")

    
    # Grafik 3: 3D strelice
    
    recnik_3d = {r: pca3.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_3d = pca3.transform(rezultat_200d.reshape(1, -1))[0]

    fig3 = plt.figure(figsize=(14, 11))
    ax3 = fig3.add_subplot(111, projection='3d')

    for rec in potrebne_reci:
        x, y, z = recnik_3d[rec]
        ax3.plot([0, x], [0, y], [0, z], color=boje[rec], lw=2.5)

    mx3_lbl, my3_lbl, mz3_lbl = recnik_3d["man"]
    kx3_lbl, ky3_lbl, kz3_lbl = recnik_3d["king"]
    natpisi_3d = {
        "king":  (kx3_lbl * 1.05, ky3_lbl * 1.05, kz3_lbl * 1.05),
        "man":   (mx3_lbl * 1.05, my3_lbl * 1.05, mz3_lbl * 1.05),
        "queen": (recnik_3d["queen"][0] * 1.18, recnik_3d["queen"][1] * 1.18, recnik_3d["queen"][2] * 1.18),
        "woman": (recnik_3d["woman"][0] * 1.18, recnik_3d["woman"][1] * 1.18, recnik_3d["woman"][2] * 1.18),
    }
    for rec in potrebne_reci:
        lx, ly, lz = natpisi_3d[rec]
        ax3.text(lx, ly, lz, rec, fontsize=12, fontweight="bold", color=boje[rec])

    kx3, ky3, kz3 = recnik_3d["king"]
    qx3, qy3, qz3 = recnik_3d["queen"]
    mx3, my3, mz3 = recnik_3d["man"]
    wx3, wy3, wz3 = recnik_3d["woman"]

    ax3.plot([kx3, qx3], [ky3, qy3], [kz3, qz3],
             color="gray", lw=1.5, linestyle="dotted")
    ax3.plot([mx3, wx3], [my3, wy3], [mz3, wz3],
             color="gray", lw=1.5, linestyle="dotted")
    _all3v = list(recnik_3d.values())
    _rng3 = max(
        max(v[0] for v in _all3v) - min(v[0] for v in _all3v),
        max(v[1] for v in _all3v) - min(v[1] for v in _all3v),
        max(v[2] for v in _all3v) - min(v[2] for v in _all3v),
        0.1
    ) * 0.18
    _xoff3, _yoff3, _zoff3 = _rng3, _rng3 * 0.5, _rng3 * 0.5
    ax3.text((kx3+qx3)/2 + _xoff3, (ky3+qy3)/2 + _yoff3, (kz3+qz3)/2 + _zoff3,
             "queen−king", fontsize=9, color="gray", ha="center")
    ax3.text((mx3+wx3)/2 - _xoff3, (my3+wy3)/2 - _yoff3, (mz3+wz3)/2 - _zoff3,
             "woman−man", fontsize=9, color="gray", ha="center")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("Očekivano: queen − king + man = woman", fontsize=14)
    ax3.view_init(elev=20, azim=225)
    plt.savefig("queen_king_man_woman3.png", dpi=150, bbox_inches="tight")

    
    # Grafik 4: 3D zbir vektora
    
    fig4 = plt.figure(figsize=(14, 11))
    ax4 = fig4.add_subplot(111, projection='3d')

    P0_3d = np.array([0.0, 0.0, 0.0])
    P1_3d = recnik_3d["queen"].copy()
    P2_3d = recnik_3d["queen"] - recnik_3d["king"]
    P3_3d = rezultat_3d.copy()

    def nacrtaj_strelu3d(ax, start, end, color, lw=2.5):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw)

    def oznaci_sredinu3d(ax, start, end, tekst, color, dz=0.0):
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(mid[0], mid[1], mid[2] + dz, tekst, fontsize=11,
                color=color, fontweight="bold")

    _dz4 = 0.01 * (max(abs(P0_3d[2]), abs(P1_3d[2]), abs(P2_3d[2]), abs(P3_3d[2])) + 0.3)
    
    nacrtaj_strelu3d(ax4, P0_3d, P1_3d, "crimson")
    oznaci_sredinu3d(ax4, P0_3d, P1_3d, "queen", "crimson", dz=_dz4)

    
    nacrtaj_strelu3d(ax4, P1_3d, P2_3d, "royalblue")
    oznaci_sredinu3d(ax4, P1_3d, P2_3d, "−king", "royalblue", dz=_dz4*2)

    
    nacrtaj_strelu3d(ax4, P2_3d, P3_3d, "forestgreen")
    oznaci_sredinu3d(ax4, P2_3d, P3_3d, "+man", "forestgreen", dz=-_dz4)

    
    ax4.plot([P0_3d[0], P3_3d[0]], [P0_3d[1], P3_3d[1]], [P0_3d[2], P3_3d[2]],
             color="darkorange", lw=2.5, linestyle="dashed")
    oznaci_sredinu3d(ax4, P0_3d, P3_3d, f"  = {najbliza}", "darkorange", dz=-_dz4*2)

    # Tacke u temenima
    for pt, col, sz, label in [
        (P1_3d, "crimson",   120, "queen"),
        (P2_3d, "slategray", 100, "queen−king"),
        (P3_3d, "darkorange",150, najbliza),
    ]:
        ax4.scatter(pt[0], pt[1], pt[2], color=col, s=sz)

    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title(
        f"Najbliza rec: queen − king + man = '{najbliza}'",
        fontsize=13
    )
    ax4.view_init(elev=20, azim=225)
    plt.savefig("queen_king_man_woman4.png", dpi=150, bbox_inches="tight")


potrebne_reci = ["man", "men", "woman", "women"]
nedostaju = [r for r in potrebne_reci if r not in recnik_vektora]

if nedostaju:
    print(f"Greska: u recniku nema: {nedostaju}")
else:
    # Sabiranje/oduzimanje u 200D prostoru
    man   = matrica_vektora[rec_to_idx["man"]]
    men   = matrica_vektora[rec_to_idx["men"]]
    woman = matrica_vektora[rec_to_idx["woman"]]

    rezultat_200d = men - man + woman

    slicnosti = matrica_vektora @ rezultat_200d
    for rec in {"man", "men", "woman"}:
        slicnosti[rec_to_idx[rec]] = -1.0
    najbliza = reci[np.argmax(slicnosti)]
    rezultati_analogija.append(f"men - man + woman = '{najbliza}' (ocekivano: 'women')")

    print(f"\nAnalogija: men - man + woman = ?")
    print(f"Najbliza rec: '{najbliza}' (ocekivano: 'women')")

    recnik_2d = {r: pca.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_2d = pca.transform(rezultat_200d.reshape(1, -1))[0]
    
    
    fig, ax = plt.subplots(figsize=(12, 10))

    boje = {
        "man":   "royalblue",
        "men":   "crimson",
        "woman": "forestgreen",
        "women": "darkorange",
    }

    sve_x = [recnik_2d[r][0] for r in potrebne_reci] + [rezultat_2d[0], 0.0]
    sve_y = [recnik_2d[r][1] for r in potrebne_reci] + [rezultat_2d[1], 0.0]
    raspon_x = max(sve_x) - min(sve_x) or 1.0
    raspon_y = max(sve_y) - min(sve_y) or 1.0
    pad = 0.25
    ax.set_xlim(min(sve_x) - pad * raspon_x, max(sve_x) + pad * raspon_x)
    ax.set_ylim(min(sve_y) - pad * raspon_y, max(sve_y) + pad * raspon_y)

    # Strelice od (0,0) za svaku rec
    for rec in potrebne_reci:
        x, y = recnik_2d[rec]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=boje[rec], lw=2.5))
        ox = 0.04 * raspon_x * (1 if x >= 0 else -1)
        oy = 0.04 * raspon_y * (1 if y >= 0 else -1)
        ax.text(x + ox, y + oy, rec, fontsize=13, fontweight="bold",
                color=boje[rec], ha="center")

    # Strelice man-->men i woman-->women
    kx, ky = recnik_2d["man"]
    qx, qy = recnik_2d["men"]
    mx, my = recnik_2d["woman"]
    wx, wy = recnik_2d["women"]

    ax.annotate("", xy=(qx, qy), xytext=(kx, ky),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))
    ax.annotate("", xy=(wx, wy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))

    _dxkq, _dykq = qx - kx, qy - ky
    _nkq = max(np.sqrt(_dxkq**2 + _dykq**2), 1e-10)
    _dxmw, _dymw = wx - mx, wy - my
    _nmw = max(np.sqrt(_dxmw**2 + _dymw**2), 1e-10)
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _goff = 0.12 * max(raspon_x, raspon_y)
    _xm = 0.12 * (_xlim[1] - _xlim[0])
    _ym = 0.12 * (_ylim[1] - _ylim[0])
    def _in_bounds(x, y):
        return (_xlim[0]+_xm < x < _xlim[1]-_xm and _ylim[0]+_ym < y < _ylim[1]-_ym)
    _xt1 = (kx+qx)/2 + (-_dykq/_nkq)*_goff
    _yt1 = (ky+qy)/2 + (_dxkq/_nkq)*_goff
    if not _in_bounds(_xt1, _yt1):
        _xt1 = (kx+qx)/2 - (-_dykq/_nkq)*_goff
        _yt1 = (ky+qy)/2 - (_dxkq/_nkq)*_goff
    _xt2 = (mx+wx)/2 + (_dymw/_nmw)*_goff
    _yt2 = (my+wy)/2 + (-_dxmw/_nmw)*_goff
    if not _in_bounds(_xt2, _yt2):
        _xt2 = (mx+wx)/2 - (_dymw/_nmw)*_goff
        _yt2 = (my+wy)/2 - (-_dxmw/_nmw)*_goff
    ax.annotate("men−man",
                xy=((kx+qx)/2, (ky+qy)/2),
                xytext=(_xt1, _yt1),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.annotate("women−woman",
                xy=((mx+wx)/2, (my+wy)/2),
                xytext=(_xt2, _yt2),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_title("Očekivano: men − man + woman = women",
                 fontsize=14)
    ax.set_xlabel("Prva glavna komponenta")
    ax.set_ylabel("Druga glavna komponenta")
    plt.grid(True, alpha=0.3)
    plt.savefig("men_man_woman_women1.png", dpi=150, bbox_inches="tight")

    
    # Grafik 2: men-man+woman = rezultat
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    q2d = recnik_2d["men"]
    k2d = recnik_2d["man"]
    w2d = recnik_2d["women"]

    # 4 temena cetvorougla
    P0 = np.array([0.0, 0.0])  
    P1 = q2d.copy()             
    P2 = q2d - k2d               
    P3 = rezultat_2d.copy()      

    sve_x2 = [P0[0], P1[0], P2[0], P3[0], w2d[0]]
    sve_y2 = [P0[1], P1[1], P2[1], P3[1], w2d[1]]
    rx2 = max(sve_x2) - min(sve_x2) or 1.0
    ry2 = max(sve_y2) - min(sve_y2) or 1.0
    pad2 = 0.30
    ax2.set_xlim(min(sve_x2) - pad2 * rx2, max(sve_x2) + pad2 * rx2)
    ax2.set_ylim(min(sve_y2) - pad2 * ry2, max(sve_y2) + pad2 * ry2)

    def nacrtaj_strelu(start, end, color, lw=2.5, ls="solid"):
        ax2.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                     linestyle=ls))

    def oznaci_strelu(start, end, tekst, color, side=1):
        s, e = np.array(start), np.array(end)
        mid = (s + e) / 2
        d = e - s
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-10) * 0.025 * max(rx2, ry2) * side
        ax2.text(mid[0] + perp[0], mid[1] + perp[1], tekst,
                 fontsize=11, color=color, ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # 3 poznate strane cetvorougla
    nacrtaj_strelu(P0, P1, "crimson")
    oznaci_strelu(P0, P1, "men", "crimson")

    nacrtaj_strelu(P1, P2, "royalblue")
    oznaci_strelu(P1, P2, "−man", "royalblue")

    nacrtaj_strelu(P2, P3, "forestgreen")
    oznaci_strelu(P2, P3, "+woman", "forestgreen", side=-1)

    # Nedostajuca 4. strana (0,0) do rezultata (isprekidana)
    nacrtaj_strelu(P0, P3, "darkorange", lw=2.8, ls="dashed")
    oznaci_strelu(P0, P3, f"men−man+woman  =  {najbliza}", "darkorange")

    # Oznacene tacke u temenima
    cx = (min(sve_x2) + max(sve_x2)) / 2
    cy = (min(sve_y2) + max(sve_y2)) / 2
    for pt, col, sz in [
        (P0, "black", 100),
        (P1, "crimson", 120),
        (P2, "slategray", 100),
        (P3, "darkorange", 150),
    ]:
        ax2.scatter(*pt, color=col, s=sz, zorder=5)

    ax2.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Najbliza rec: men − man + woman = '{najbliza}'\n",
        fontsize=13
    )
    ax2.set_xlabel("Prva glavna komponenta")
    ax2.set_ylabel("Druga glavna komponenta")
    ax2.grid(True, alpha=0.3)
    plt.savefig("men_man_woman_women2.png", dpi=150, bbox_inches="tight")

    
    # Grafik 3: 3D strelice od (0,0)
    
    recnik_3d = {r: pca3.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_3d = pca3.transform(rezultat_200d.reshape(1, -1))[0]

    fig3 = plt.figure(figsize=(14, 11))
    ax3 = fig3.add_subplot(111, projection='3d')

    for rec in potrebne_reci:
        x, y, z = recnik_3d[rec]
        ax3.plot([0, x], [0, y], [0, z], color=boje[rec], lw=2.5)

    mx3_lbl, my3_lbl, mz3_lbl = recnik_3d["man"]
    kx3_lbl, ky3_lbl, kz3_lbl = recnik_3d["men"]
    natpisi_3d = {
        "man":   (mx3_lbl * 1.05, my3_lbl * 1.05, mz3_lbl * 1.05),
        "men":   (kx3_lbl * 1.18, ky3_lbl * 1.18, kz3_lbl * 1.18),
        "woman": (recnik_3d["woman"][0] * 1.05, recnik_3d["woman"][1] * 1.05, recnik_3d["woman"][2] * 1.05),
        "women": (recnik_3d["women"][0] * 1.18, recnik_3d["women"][1] * 1.18, recnik_3d["women"][2] * 1.18),
    }
    for rec in potrebne_reci:
        lx, ly, lz = natpisi_3d[rec]
        ax3.text(lx, ly, lz, rec, fontsize=12, fontweight="bold", color=boje[rec])

    kx3, ky3, kz3 = recnik_3d["man"]
    qx3, qy3, qz3 = recnik_3d["men"]
    mx3, my3, mz3 = recnik_3d["woman"]
    wx3, wy3, wz3 = recnik_3d["women"]

    ax3.plot([kx3, qx3], [ky3, qy3], [kz3, qz3],
             color="gray", lw=1.5, linestyle="dotted")
    ax3.plot([mx3, wx3], [my3, wy3], [mz3, wz3],
             color="gray", lw=1.5, linestyle="dotted")
    _all3v = list(recnik_3d.values())
    _rng3 = max(
        max(v[0] for v in _all3v) - min(v[0] for v in _all3v),
        max(v[1] for v in _all3v) - min(v[1] for v in _all3v),
        max(v[2] for v in _all3v) - min(v[2] for v in _all3v),
        0.1
    ) * 0.18
    _xoff3, _yoff3, _zoff3 = _rng3, _rng3 * 0.5, _rng3 * 0.5
    ax3.text((kx3+qx3)/2 + _xoff3, (ky3+qy3)/2 + _yoff3, (kz3+qz3)/2 + _zoff3,
             "men−man", fontsize=9, color="gray", ha="center")
    ax3.text((mx3+wx3)/2 - _xoff3, (my3+wy3)/2 - _yoff3, (mz3+wz3)/2 - _zoff3,
             "women−woman", fontsize=9, color="gray", ha="center")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("Očekivano: men − man + woman = women", fontsize=14)
    ax3.view_init(elev=20, azim=225)
    plt.savefig("men_man_woman_women3.png", dpi=150, bbox_inches="tight")

    
    # Grafik 4: 3D zbir vektora
    
    fig4 = plt.figure(figsize=(14, 11))
    ax4 = fig4.add_subplot(111, projection='3d')

    P0_3d = np.array([0.0, 0.0, 0.0])
    P1_3d = recnik_3d["men"].copy()
    P2_3d = recnik_3d["men"] - recnik_3d["man"]
    P3_3d = rezultat_3d.copy()

    def nacrtaj_strelu3d(ax, start, end, color, lw=2.5):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw)

    def oznaci_sredinu3d(ax, start, end, tekst, color, dz=0.0):
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(mid[0], mid[1], mid[2] + dz, tekst, fontsize=11,
                color=color, fontweight="bold")

    _dz4 = 0.01 * (max(abs(P0_3d[2]), abs(P1_3d[2]), abs(P2_3d[2]), abs(P3_3d[2])) + 0.3)
     
    nacrtaj_strelu3d(ax4, P0_3d, P1_3d, "crimson")
    oznaci_sredinu3d(ax4, P0_3d, P1_3d, "men", "crimson", dz=_dz4)

    
    nacrtaj_strelu3d(ax4, P1_3d, P2_3d, "royalblue")
    oznaci_sredinu3d(ax4, P1_3d, P2_3d, "−man", "royalblue", dz=0)

    

    nacrtaj_strelu3d(ax4, P2_3d, P3_3d, "forestgreen")
    oznaci_sredinu3d(ax4, P2_3d, P3_3d, "+woman", "forestgreen", dz=_dz4*1.5)

    
    ax4.plot([P0_3d[0], P3_3d[0]], [P0_3d[1], P3_3d[1]], [P0_3d[2], P3_3d[2]],
             color="darkorange", lw=2.5, linestyle="dashed")
    oznaci_sredinu3d(ax4, P0_3d, P3_3d, f"  = {najbliza}", "darkorange", dz=-_dz4*2)

    # Tacke u temenima
    for pt, col, sz, label in [
        (P1_3d, "crimson",   120, "men"),
        (P2_3d, "slategray", 100, "men−man"),
        (P3_3d, "darkorange",150, najbliza),
    ]:
        ax4.scatter(pt[0], pt[1], pt[2], color=col, s=sz)

    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title(
        f"Najbliza rec: men − man + woman = '{najbliza}'",
        fontsize=13
    )
    #promeni se ugao iz kog se gleda da bi se slika bolje videla
    ax4.view_init(elev=20, azim=225)
    plt.savefig("men_man_woman_women4.png", dpi=150, bbox_inches="tight")


potrebne_reci = ["good", "better", "bad", "worse"]
nedostaju = [r for r in potrebne_reci if r not in recnik_vektora]

if nedostaju:
    print(f"Greska: sledece reci nisu pronadjene u recniku: {nedostaju}")
else:
    # Sabiranje/oduzimanje u 200D prostoru
    good   = matrica_vektora[rec_to_idx["good"]]
    better = matrica_vektora[rec_to_idx["better"]]
    bad    = matrica_vektora[rec_to_idx["bad"]]

    rezultat_200d = better - good + bad


    slicnosti = matrica_vektora @ rezultat_200d
    for rec in {"good", "better", "bad"}:
        slicnosti[rec_to_idx[rec]] = -1.0
    najbliza = reci[np.argmax(slicnosti)]
    rezultati_analogija.append(f"better - good + bad = '{najbliza}' (ocekivano: 'worse')")

    print(f"\nAnalogija: better - good + bad = ?")
    print(f"Najbliza rec: '{najbliza}' (ocekivano: 'worse')")

    
    recnik_2d = {r: pca.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_2d = pca.transform(rezultat_200d.reshape(1, -1))[0]


    
    # Grafik: strelice od (0,0) 
    
    fig, ax = plt.subplots(figsize=(12, 10))

    boje = {
        "good":   "royalblue",
        "better": "crimson",
        "bad":    "forestgreen",
        "worse":  "darkorange",
    }

    sve_x = [recnik_2d[r][0] for r in potrebne_reci] + [rezultat_2d[0], 0.0]
    sve_y = [recnik_2d[r][1] for r in potrebne_reci] + [rezultat_2d[1], 0.0]
    raspon_x = max(sve_x) - min(sve_x) or 1.0
    raspon_y = max(sve_y) - min(sve_y) or 1.0
    pad = 0.25
    ax.set_xlim(min(sve_x) - pad * raspon_x, max(sve_x) + pad * raspon_x)
    ax.set_ylim(min(sve_y) - pad * raspon_y, max(sve_y) + pad * raspon_y)

    # Strelice od (0,0) za svaku rec
    for rec in potrebne_reci:
        x, y = recnik_2d[rec]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=boje[rec], lw=2.5))
        ox = 0.04 * raspon_x * (1 if x >= 0 else -1)
        oy = 0.04 * raspon_y * (1 if y >= 0 else -1)
        ax.text(x + ox, y + oy, rec, fontsize=13, fontweight="bold",
                color=boje[rec], ha="center")

    # Strelice good-->better i bad-->worse
    kx, ky = recnik_2d["good"]
    qx, qy = recnik_2d["better"]
    mx, my = recnik_2d["bad"]
    wx, wy = recnik_2d["worse"]

    ax.annotate("", xy=(qx, qy), xytext=(kx, ky),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))
    ax.annotate("", xy=(wx, wy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))

    ax.text((kx+qx)/2, (ky+qy)/2, "better−good",
            fontsize=10, color="dimgray", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray",
                      alpha=1.0, lw=1.0))
    ax.text((mx+wx)/2, (my+wy)/2, "worse−bad",
            fontsize=10, color="dimgray", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray",
                      alpha=1.0, lw=1.0))

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_title("Očekivano: better − good + bad = worse",
                 fontsize=14)
    ax.set_xlabel("Prva glavna komponenta")
    ax.set_ylabel("Druga glavna komponenta")
    plt.grid(True, alpha=0.3)
    plt.savefig("better_good_bad_worse1.png", dpi=150, bbox_inches="tight")

    
    # Grafik 2: better-good+bad = rezultat
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    q2d = recnik_2d["better"]
    k2d = recnik_2d["good"]
    w2d = recnik_2d["worse"]

    # 4 temena cetvorougla
    P0 = np.array([0.0, 0.0])   
    P1 = q2d.copy()              
    P2 = q2d - k2d               
    P3 = rezultat_2d.copy()      

    sve_x2 = [P0[0], P1[0], P2[0], P3[0], w2d[0]]
    sve_y2 = [P0[1], P1[1], P2[1], P3[1], w2d[1]]
    rx2 = max(sve_x2) - min(sve_x2) or 1.0
    ry2 = max(sve_y2) - min(sve_y2) or 1.0
    pad2 = 0.30
    ax2.set_xlim(min(sve_x2) - pad2 * rx2, max(sve_x2) + pad2 * rx2)
    ax2.set_ylim(min(sve_y2) - pad2 * ry2, max(sve_y2) + pad2 * ry2)

    def nacrtaj_strelu(start, end, color, lw=2.5, ls="solid"):
        ax2.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                     linestyle=ls))

    def oznaci_strelu(start, end, tekst, color, side=1):
        s, e = np.array(start), np.array(end)
        mid = (s + e) / 2
        d = e - s
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-10) * 0.025 * max(rx2, ry2) * side
        ax2.text(mid[0] + perp[0], mid[1] + perp[1], tekst,
                 fontsize=11, color=color, ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # 3 poznate strane cetvorougla
    nacrtaj_strelu(P0, P1, "crimson")
    oznaci_strelu(P0, P1, "better", "crimson")

    nacrtaj_strelu(P1, P2, "royalblue")
    oznaci_strelu(P1, P2, "−good", "royalblue")

    nacrtaj_strelu(P2, P3, "forestgreen")
    oznaci_strelu(P2, P3, "+bad", "forestgreen", side=-1)

    # Nedostajuca 4. strana (0,0) do rezultata (isprekidana)
    nacrtaj_strelu(P0, P3, "darkorange", lw=2.8, ls="dashed")
    _r0, _r3 = np.array(P0), np.array(P3)
    _rmid = (_r0 + _r3) / 2
    _rd = _r3 - _r0
    _rperp = np.array([-_rd[1], _rd[0]])
    _rperp = _rperp / (np.linalg.norm(_rperp) + 1e-10) * 0.04 * max(rx2, ry2)
    ax2.text(_rmid[0] + _rperp[0] - 0.09 * rx2, _rmid[1] + _rperp[1] + 0.06 * ry2,
             f"better−good+bad  =  {najbliza}",
             fontsize=11, color="darkorange", ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # Oznacene tacke u temenima
    cx = (min(sve_x2) + max(sve_x2)) / 2
    cy = (min(sve_y2) + max(sve_y2)) / 2
    for pt, col, sz in [
        (P0, "black", 100),
        (P1, "crimson", 120),
        (P2, "slategray", 100),
        (P3, "darkorange", 150),
    ]:
        ax2.scatter(*pt, color=col, s=sz, zorder=5)

    ax2.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Najbliza rec: better − good + bad = '{najbliza}'\n",
        fontsize=13
    )
    ax2.set_xlabel("Prva glavna komponenta")
    ax2.set_ylabel("Druga glavna komponenta")
    ax2.grid(True, alpha=0.3)
    plt.savefig("better_good_bad_worse2.png", dpi=150, bbox_inches="tight")

    
    # Grafik 3: 3D strelice od (0,0)
    
    recnik_3d = {r: pca3.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_3d = pca3.transform(rezultat_200d.reshape(1, -1))[0]

    fig3 = plt.figure(figsize=(14, 11))
    ax3 = fig3.add_subplot(111, projection='3d')

    for rec in potrebne_reci:
        x, y, z = recnik_3d[rec]
        ax3.plot([0, x], [0, y], [0, z], color=boje[rec], lw=2.5)

    mx3_lbl, my3_lbl, mz3_lbl = recnik_3d["good"]
    kx3_lbl, ky3_lbl, kz3_lbl = recnik_3d["bad"]
    natpisi_3d = {
        "good":   (mx3_lbl * 1.05, my3_lbl * 1.05, mz3_lbl * 1.05),
        "bad":    (kx3_lbl * 1.05, ky3_lbl * 1.05, kz3_lbl * 1.05),
        "better": (recnik_3d["better"][0] * 1.18, recnik_3d["better"][1] * 1.18, recnik_3d["better"][2] * 1.18),
        "worse":  (recnik_3d["worse"][0] * 1.05, recnik_3d["worse"][1] * 1.05, recnik_3d["worse"][2] * 1.05),
    }
    for rec in potrebne_reci:
        lx, ly, lz = natpisi_3d[rec]
        ax3.text(lx, ly, lz, rec, fontsize=12, fontweight="bold", color=boje[rec])

    kx3, ky3, kz3 = recnik_3d["good"]
    qx3, qy3, qz3 = recnik_3d["better"]
    mx3, my3, mz3 = recnik_3d["bad"]
    wx3, wy3, wz3 = recnik_3d["worse"]

    ax3.plot([kx3, qx3], [ky3, qy3], [kz3, qz3],
             color="gray", lw=1.5, linestyle="dotted")
    ax3.plot([mx3, wx3], [my3, wy3], [mz3, wz3],
             color="gray", lw=1.5, linestyle="dotted")
    _all3v = list(recnik_3d.values())
    _rng3 = max(
        max(v[0] for v in _all3v) - min(v[0] for v in _all3v),
        max(v[1] for v in _all3v) - min(v[1] for v in _all3v),
        max(v[2] for v in _all3v) - min(v[2] for v in _all3v),
        0.1
    ) * 0.18
    _xoff3, _yoff3, _zoff3 = _rng3, _rng3 * 0.5, _rng3 * 0.5
    ax3.text((kx3+qx3)/2 + _xoff3, (ky3+qy3)/2 + _yoff3, (kz3+qz3)/2 + _zoff3,
             "better−good", fontsize=9, color="gray", ha="center")
    ax3.text((mx3+wx3)/2 - _xoff3, (my3+wy3)/2 - _yoff3, (mz3+wz3)/2 - _zoff3,
             "worse−bad", fontsize=9, color="gray", ha="center")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("Očekivano: better − good + bad = worse", fontsize=14)
    ax3.view_init(elev=20, azim=225)
    plt.savefig("better_good_bad_worse3.png", dpi=150, bbox_inches="tight")

    
    # Grafik 4: 3D zbir vektora
    
    fig4 = plt.figure(figsize=(14, 11))
    ax4 = fig4.add_subplot(111, projection='3d')

    P0_3d = np.array([0.0, 0.0, 0.0])
    P1_3d = recnik_3d["better"].copy()
    P2_3d = recnik_3d["better"] - recnik_3d["good"]
    P3_3d = rezultat_3d.copy()

    def nacrtaj_strelu3d(ax, start, end, color, lw=2.5):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw)

    def oznaci_sredinu3d(ax, start, end, tekst, color, dz=0.0):
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(mid[0], mid[1], mid[2] + dz, tekst, fontsize=11,
                color=color, fontweight="bold")

    _dz4 = 0.01 * (max(abs(P0_3d[2]), abs(P1_3d[2]), abs(P2_3d[2]), abs(P3_3d[2])) + 0.3)
    
    nacrtaj_strelu3d(ax4, P0_3d, P1_3d, "crimson")
    oznaci_sredinu3d(ax4, P0_3d, P1_3d, "better", "crimson", dz=_dz4)

    
    nacrtaj_strelu3d(ax4, P1_3d, P2_3d, "royalblue")
    oznaci_sredinu3d(ax4, P1_3d, P2_3d, "−good", "royalblue", dz=_dz4*2)

    
    nacrtaj_strelu3d(ax4, P2_3d, P3_3d, "forestgreen")
    oznaci_sredinu3d(ax4, P2_3d, P3_3d, "+bad", "forestgreen", dz=-_dz4)

    
    ax4.plot([P0_3d[0], P3_3d[0]], [P0_3d[1], P3_3d[1]], [P0_3d[2], P3_3d[2]],
             color="darkorange", lw=2.5, linestyle="dashed")
    oznaci_sredinu3d(ax4, P0_3d, P3_3d, f"  = {najbliza}", "darkorange", dz=-_dz4*2)

    # Tacke u temenima
    for pt, col, sz, label in [
        (P1_3d, "crimson",   120, "better"),
        (P2_3d, "slategray", 100, "better−good"),
        (P3_3d, "darkorange",150, najbliza),
    ]:
        ax4.scatter(pt[0], pt[1], pt[2], color=col, s=sz)

    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title(
        f"Najbliza rec: better − good + bad = '{najbliza}'",
        fontsize=13
    )
    #promeni se ugao iz kog se gleda da bi se slika bolje videla
    ax4.view_init(elev=20, azim=225)
    plt.savefig("better_good_bad_worse4.png", dpi=150, bbox_inches="tight")


potrebne_reci = ["japan", "japanese", "spain", "spanish"]
nedostaju = [r for r in potrebne_reci if r not in recnik_vektora]

if nedostaju:
    print(f"Greska: sledece reci nisu pronadjene u recniku: {nedostaju}")
else:
    # Sabiranje/oduzimanje u 200D prostoru
    japan    = matrica_vektora[rec_to_idx["japan"]]
    japanese = matrica_vektora[rec_to_idx["japanese"]]
    spain    = matrica_vektora[rec_to_idx["spain"]]

    rezultat_200d = japanese - japan + spain


    slicnosti = matrica_vektora @ rezultat_200d
    for rec in {"japan", "japanese", "spain"}:
        slicnosti[rec_to_idx[rec]] = -1.0
    najbliza = reci[np.argmax(slicnosti)]
    rezultati_analogija.append(f"japanese - japan + spain = '{najbliza}' (ocekivano: 'spanish')")

    print(f"\nAnalogija: japanese - japan + spain = ?")
    print(f"Najbliza rec: '{najbliza}' (ocekivano: 'spanish')")

    
    recnik_2d = {r: pca.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_2d = pca.transform(rezultat_200d.reshape(1, -1))[0]


    
    # Grafik: strelice od (0,0) 
    
    fig, ax = plt.subplots(figsize=(12, 10))

    boje = {
        "japan":    "royalblue",
        "japanese": "crimson",
        "spain":    "forestgreen",
        "spanish":  "darkorange",
    }

    sve_x = [recnik_2d[r][0] for r in potrebne_reci] + [rezultat_2d[0], 0.0]
    sve_y = [recnik_2d[r][1] for r in potrebne_reci] + [rezultat_2d[1], 0.0]
    raspon_x = max(sve_x) - min(sve_x) or 1.0
    raspon_y = max(sve_y) - min(sve_y) or 1.0
    pad = 0.25
    ax.set_xlim(min(sve_x) - pad * raspon_x, max(sve_x) + pad * raspon_x)
    ax.set_ylim(min(sve_y) - pad * raspon_y, max(sve_y) + pad * raspon_y)

    # Strelice od (0,0) za svaku rec
    for rec in potrebne_reci:
        x, y = recnik_2d[rec]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=boje[rec], lw=2.5))
        ox = 0.04 * raspon_x * (1 if x >= 0 else -1)
        oy = 0.04 * raspon_y * (1 if y >= 0 else -1)
        ax.text(x + ox, y + oy, rec, fontsize=13, fontweight="bold",
                color=boje[rec], ha="center")

    # Strelice japan-->japanese i spain-->spanish
    kx, ky = recnik_2d["japan"]
    qx, qy = recnik_2d["japanese"]
    mx, my = recnik_2d["spain"]
    wx, wy = recnik_2d["spanish"]

    ax.annotate("", xy=(qx, qy), xytext=(kx, ky),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))
    ax.annotate("", xy=(wx, wy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))

    _dxkq, _dykq = qx - kx, qy - ky
    _nkq = max(np.sqrt(_dxkq**2 + _dykq**2), 1e-10)
    _dxmw, _dymw = wx - mx, wy - my
    _nmw = max(np.sqrt(_dxmw**2 + _dymw**2), 1e-10)
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _goff = 0.12 * max(raspon_x, raspon_y)
    _xm = 0.12 * (_xlim[1] - _xlim[0])
    _ym = 0.12 * (_ylim[1] - _ylim[0])
    def _in_bounds(x, y):
        return (_xlim[0]+_xm < x < _xlim[1]-_xm and _ylim[0]+_ym < y < _ylim[1]-_ym)
    _xt1 = (kx+qx)/2 + (-_dykq/_nkq)*_goff
    _yt1 = (ky+qy)/2 + (_dxkq/_nkq)*_goff
    if not _in_bounds(_xt1, _yt1):
        _xt1 = (kx+qx)/2 - (-_dykq/_nkq)*_goff
        _yt1 = (ky+qy)/2 - (_dxkq/_nkq)*_goff
    _xt2 = (mx+wx)/2 + (_dymw/_nmw)*_goff
    _yt2 = (my+wy)/2 + (-_dxmw/_nmw)*_goff
    if not _in_bounds(_xt2, _yt2):
        _xt2 = (mx+wx)/2 - (_dymw/_nmw)*_goff
        _yt2 = (my+wy)/2 - (-_dxmw/_nmw)*_goff
    ax.annotate("japanese−japan",
                xy=((kx+qx)/2, (ky+qy)/2),
                xytext=(_xt1, _yt1),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.annotate("spanish−spain",
                xy=((mx+wx)/2, (my+wy)/2),
                xytext=(_xt2, _yt2),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_title("Očekivano: japanese − japan + spain = spanish",
                 fontsize=14)
    ax.set_xlabel("Prva glavna komponenta")
    ax.set_ylabel("Druga glavna komponenta")
    plt.grid(True, alpha=0.3)
    plt.savefig("japanese_japan_spain_spanish1.png", dpi=150, bbox_inches="tight")

    
    # Grafik 2: japanese-japan+spain = rezultat
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    q2d = recnik_2d["japanese"]
    k2d = recnik_2d["japan"]
    w2d = recnik_2d["spanish"]

    # 4 temena cetvorougla
    P0 = np.array([0.0, 0.0])   
    P1 = q2d.copy()              
    P2 = q2d - k2d               
    P3 = rezultat_2d.copy()      

    sve_x2 = [P0[0], P1[0], P2[0], P3[0], w2d[0]]
    sve_y2 = [P0[1], P1[1], P2[1], P3[1], w2d[1]]
    rx2 = max(sve_x2) - min(sve_x2) or 1.0
    ry2 = max(sve_y2) - min(sve_y2) or 1.0
    pad2 = 0.30
    ax2.set_xlim(min(sve_x2) - pad2 * rx2, max(sve_x2) + pad2 * rx2)
    ax2.set_ylim(min(sve_y2) - pad2 * ry2, max(sve_y2) + pad2 * ry2)

    def nacrtaj_strelu(start, end, color, lw=2.5, ls="solid"):
        ax2.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                     linestyle=ls))

    def oznaci_strelu(start, end, tekst, color, side=1, dx=0.0, dy=0.0):
        s, e = np.array(start), np.array(end)
        mid = (s + e) / 2
        d = e - s
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-10) * 0.025 * max(rx2, ry2) * side
        ax2.text(mid[0] + perp[0] + dx, mid[1] + perp[1] + dy, tekst,
                 fontsize=11, color=color, ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # 3 poznate strane cetvorougla
    nacrtaj_strelu(P0, P1, "crimson")
    oznaci_strelu(P0, P1, "japanese", "crimson", dx=-0.05*rx2)

    nacrtaj_strelu(P1, P2, "royalblue")
    oznaci_strelu(P1, P2, "−japan", "royalblue", dx=0.04*rx2, dy=0.05*ry2)

    nacrtaj_strelu(P2, P3, "forestgreen")
    oznaci_strelu(P2, P3, "+spain", "forestgreen", side=-1)

    # Nedostajuca 4. strana (0,0) do rezultata (isprekidana)
    nacrtaj_strelu(P0, P3, "darkorange", lw=2.8, ls="dashed")
    oznaci_strelu(P0, P3, f"japanese−japan+spain  =  {najbliza}", "darkorange", dy=-0.05*ry2)

    # Oznacene tacke u temenima
    cx = (min(sve_x2) + max(sve_x2)) / 2
    cy = (min(sve_y2) + max(sve_y2)) / 2
    for pt, col, sz in [
        (P0, "black", 100),
        (P1, "crimson", 120),
        (P2, "slategray", 100),
        (P3, "darkorange", 150),
    ]:
        ax2.scatter(*pt, color=col, s=sz, zorder=5)

    ax2.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Najbliza rec: japanese − japan + spain = '{najbliza}'\n",
        fontsize=13
    )
    ax2.set_xlabel("Prva glavna komponenta")
    ax2.set_ylabel("Druga glavna komponenta")
    ax2.grid(True, alpha=0.3)
    plt.savefig("japanese_japan_spain_spanish2.png", dpi=150, bbox_inches="tight")

    
    # Grafik 3: 3D strelice od (0,0)
    
    recnik_3d = {r: pca3.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_3d = pca3.transform(rezultat_200d.reshape(1, -1))[0]

    fig3 = plt.figure(figsize=(14, 11))
    ax3 = fig3.add_subplot(111, projection='3d')

    for rec in potrebne_reci:
        x, y, z = recnik_3d[rec]
        ax3.plot([0, x], [0, y], [0, z], color=boje[rec], lw=2.5)

    mx3_lbl, my3_lbl, mz3_lbl = recnik_3d["japan"]
    kx3_lbl, ky3_lbl, kz3_lbl = recnik_3d["spain"]
    natpisi_3d = {
        "japan":    (mx3_lbl * 1.02, my3_lbl * 1.02, mz3_lbl * 1.02),
        "spain":    (kx3_lbl * 1.02, ky3_lbl * 1.02, kz3_lbl * 1.02),
        "japanese": (recnik_3d["japanese"][0] * 1.10, recnik_3d["japanese"][1] * 1.10, recnik_3d["japanese"][2] * 1.10),
        "spanish":  (recnik_3d["spanish"][0] * 1.10, recnik_3d["spanish"][1] * 1.10, recnik_3d["spanish"][2] * 1.10),
    }
    for rec in potrebne_reci:
        lx, ly, lz = natpisi_3d[rec]
        ax3.text(lx, ly, lz, rec, fontsize=12, fontweight="bold", color=boje[rec])

    kx3, ky3, kz3 = recnik_3d["japan"]
    qx3, qy3, qz3 = recnik_3d["japanese"]
    mx3, my3, mz3 = recnik_3d["spain"]
    wx3, wy3, wz3 = recnik_3d["spanish"]

    ax3.plot([kx3, qx3], [ky3, qy3], [kz3, qz3],
             color="gray", lw=1.5, linestyle="dotted")
    ax3.plot([mx3, wx3], [my3, wy3], [mz3, wz3],
             color="gray", lw=1.5, linestyle="dotted")
    _all3v = list(recnik_3d.values())
    _rng3 = max(
        max(v[0] for v in _all3v) - min(v[0] for v in _all3v),
        max(v[1] for v in _all3v) - min(v[1] for v in _all3v),
        max(v[2] for v in _all3v) - min(v[2] for v in _all3v),
        0.1
    ) * 0.18
    _xoff3, _yoff3, _zoff3 = _rng3, _rng3 * 0.5, _rng3 * 0.5
    ax3.text((kx3+qx3)/2 + _xoff3, (ky3+qy3)/2 + _yoff3, (kz3+qz3)/2 + _zoff3,
             "japanese−japan", fontsize=9, color="gray", ha="center")
    ax3.text((mx3+wx3)/2 - _xoff3, (my3+wy3)/2 - _yoff3, (mz3+wz3)/2 - _zoff3,
             "spanish−spain", fontsize=9, color="gray", ha="center")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("Očekivano: japanese − japan + spain = spanish", fontsize=14)
    ax3.view_init(elev=20, azim=225)
    plt.savefig("japanese_japan_spain_spanish3.png", dpi=150, bbox_inches="tight")

    
    # Grafik 4: 3D zbir vektora
    
    fig4 = plt.figure(figsize=(14, 11))
    ax4 = fig4.add_subplot(111, projection='3d')

    P0_3d = np.array([0.0, 0.0, 0.0])
    P1_3d = recnik_3d["japanese"].copy()
    P2_3d = recnik_3d["japanese"] - recnik_3d["japan"]
    P3_3d = rezultat_3d.copy()

    def nacrtaj_strelu3d(ax, start, end, color, lw=2.5):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw)

    def oznaci_sredinu3d(ax, start, end, tekst, color, dz=0.0):
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(mid[0], mid[1], mid[2] + dz, tekst, fontsize=11,
                color=color, fontweight="bold")

    _dz4 = 0.01 * (max(abs(P0_3d[2]), abs(P1_3d[2]), abs(P2_3d[2]), abs(P3_3d[2])) + 0.3)
    
    nacrtaj_strelu3d(ax4, P0_3d, P1_3d, "crimson")
    oznaci_sredinu3d(ax4, P0_3d, P1_3d, "japanese", "crimson", dz=-_dz4*2)

    
    nacrtaj_strelu3d(ax4, P1_3d, P2_3d, "royalblue")
    oznaci_sredinu3d(ax4, P1_3d, P2_3d, "−japan", "royalblue", dz=_dz4*2)

    
    nacrtaj_strelu3d(ax4, P2_3d, P3_3d, "forestgreen")
    oznaci_sredinu3d(ax4, P2_3d, P3_3d, "+spain", "forestgreen", dz=-_dz4)

    
    ax4.plot([P0_3d[0], P3_3d[0]], [P0_3d[1], P3_3d[1]], [P0_3d[2], P3_3d[2]],
             color="darkorange", lw=2.5, linestyle="dashed")
    oznaci_sredinu3d(ax4, P0_3d, P3_3d, f"  = {najbliza}", "darkorange", dz=_dz4)

    # Tacke u temenima
    for pt, col, sz, label in [
        (P1_3d, "crimson",   120, "japanese"),
        (P2_3d, "slategray", 100, "japanese−japan"),
        (P3_3d, "darkorange",150, najbliza),
    ]:
        ax4.scatter(pt[0], pt[1], pt[2], color=col, s=sz)

    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title(
        f"Najbliza rec: japanese − japan + spain = '{najbliza}'",
        fontsize=13
    )
    #promeni se ugao iz kog se gleda da bi se slika bolje videla
    ax4.view_init(elev=20, azim=225)
    plt.savefig("japanese_japan_spain_spanish4.png", dpi=150, bbox_inches="tight")


potrebne_reci = ["go", "went", "come", "came"]
nedostaju = [r for r in potrebne_reci if r not in recnik_vektora]

if nedostaju:
    print(f"Greska: sledece reci nisu pronadjene u recniku: {nedostaju}")
else:
    # Sabiranje/oduzimanje u 200D prostoru
    go   = matrica_vektora[rec_to_idx["go"]]
    went = matrica_vektora[rec_to_idx["went"]]
    come = matrica_vektora[rec_to_idx["come"]]

    rezultat_200d = went - go + come


    slicnosti = matrica_vektora @ rezultat_200d
    for rec in {"go", "went", "come"}:
        slicnosti[rec_to_idx[rec]] = -1.0
    najbliza = reci[np.argmax(slicnosti)]
    rezultati_analogija.append(f"went - go + come = '{najbliza}' (ocekivano: 'came')")

    print(f"\nAnalogija: went - go + come = ?")
    print(f"Najbliza rec: '{najbliza}' (ocekivano: 'came')")

    
    recnik_2d = {r: pca.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_2d = pca.transform(rezultat_200d.reshape(1, -1))[0]


    
    # Grafik: strelice od (0,0) 
    
    fig, ax = plt.subplots(figsize=(12, 10))

    boje = {
        "go":   "royalblue",
        "went": "crimson",
        "come": "forestgreen",
        "came": "darkorange",
    }

    sve_x = [recnik_2d[r][0] for r in potrebne_reci] + [rezultat_2d[0], 0.0]
    sve_y = [recnik_2d[r][1] for r in potrebne_reci] + [rezultat_2d[1], 0.0]
    raspon_x = max(sve_x) - min(sve_x) or 1.0
    raspon_y = max(sve_y) - min(sve_y) or 1.0
    pad = 0.25
    ax.set_xlim(min(sve_x) - pad * raspon_x, max(sve_x) + pad * raspon_x)
    ax.set_ylim(min(sve_y) - pad * raspon_y, max(sve_y) + pad * raspon_y)

    # Strelice od (0,0) za svaku rec
    for rec in potrebne_reci:
        x, y = recnik_2d[rec]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=boje[rec], lw=2.5))
        ox = 0.04 * raspon_x * (1 if x >= 0 else -1)
        oy = 0.04 * raspon_y * (1 if y >= 0 else -1)
        ax.text(x + ox, y + oy, rec, fontsize=13, fontweight="bold",
                color=boje[rec], ha="center")

    # Strelice go-->went i come-->came
    kx, ky = recnik_2d["go"]
    qx, qy = recnik_2d["went"]
    mx, my = recnik_2d["come"]
    wx, wy = recnik_2d["came"]

    ax.annotate("", xy=(qx, qy), xytext=(kx, ky),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))
    ax.annotate("", xy=(wx, wy), xytext=(mx, my),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, linestyle="dotted"))

    _dxkq, _dykq = qx - kx, qy - ky
    _nkq = max(np.sqrt(_dxkq**2 + _dykq**2), 1e-10)
    _dxmw, _dymw = wx - mx, wy - my
    _nmw = max(np.sqrt(_dxmw**2 + _dymw**2), 1e-10)
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _goff = 0.12 * max(raspon_x, raspon_y)
    _xm = 0.12 * (_xlim[1] - _xlim[0])
    _ym = 0.12 * (_ylim[1] - _ylim[0])
    def _in_bounds(x, y):
        return (_xlim[0]+_xm < x < _xlim[1]-_xm and _ylim[0]+_ym < y < _ylim[1]-_ym)
    _xt1 = (kx+qx)/2 + (-_dykq/_nkq)*_goff
    _yt1 = (ky+qy)/2 + (_dxkq/_nkq)*_goff
    if not _in_bounds(_xt1, _yt1):
        _xt1 = (kx+qx)/2 - (-_dykq/_nkq)*_goff
        _yt1 = (ky+qy)/2 - (_dxkq/_nkq)*_goff
    _xt2 = (mx+wx)/2 + (_dymw/_nmw)*_goff
    _yt2 = (my+wy)/2 + (-_dxmw/_nmw)*_goff
    if not _in_bounds(_xt2, _yt2):
        _xt2 = (mx+wx)/2 - (_dymw/_nmw)*_goff
        _yt2 = (my+wy)/2 - (-_dxmw/_nmw)*_goff
    ax.annotate("went−go",
                xy=((kx+qx)/2, (ky+qy)/2),
                xytext=(_xt1, _yt1),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.annotate("came−come",
                xy=((mx+wx)/2, (my+wy)/2),
                xytext=(_xt2, _yt2),
                fontsize=9, color="gray", ha="center",
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax.set_title("Očekivano: went − go + come = came",
                 fontsize=14)
    ax.set_xlabel("Prva glavna komponenta")
    ax.set_ylabel("Druga glavna komponenta")
    plt.grid(True, alpha=0.3)
    plt.savefig("went_go_come_came1.png", dpi=150, bbox_inches="tight")

    
    # Grafik 2: went-go+come = rezultat
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    q2d = recnik_2d["went"]
    k2d = recnik_2d["go"]
    w2d = recnik_2d["came"]

    # 4 temena cetvorougla
    P0 = np.array([0.0, 0.0])  
    P1 = q2d.copy()              
    P2 = q2d - k2d               
    P3 = rezultat_2d.copy()      

    sve_x2 = [P0[0], P1[0], P2[0], P3[0], w2d[0]]
    sve_y2 = [P0[1], P1[1], P2[1], P3[1], w2d[1]]
    rx2 = max(sve_x2) - min(sve_x2) or 1.0
    ry2 = max(sve_y2) - min(sve_y2) or 1.0
    pad2 = 0.30
    ax2.set_xlim(min(sve_x2) - pad2 * rx2, max(sve_x2) + pad2 * rx2)
    ax2.set_ylim(min(sve_y2) - pad2 * ry2, max(sve_y2) + pad2 * ry2)

    def nacrtaj_strelu(start, end, color, lw=2.5, ls="solid"):
        ax2.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                     linestyle=ls))

    def oznaci_strelu(start, end, tekst, color, side=1):
        s, e = np.array(start), np.array(end)
        mid = (s + e) / 2
        d = e - s
        perp = np.array([-d[1], d[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-10) * 0.04 * max(rx2, ry2) * side
        ax2.text(mid[0] + perp[0], mid[1] + perp[1], tekst,
                 fontsize=11, color=color, ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

    # 3 poznate strane cetvorougla
    nacrtaj_strelu(P0, P1, "crimson")
    oznaci_strelu(P0, P1, "went", "crimson")

    nacrtaj_strelu(P1, P2, "royalblue")
    oznaci_strelu(P1, P2, "−go", "royalblue")

    nacrtaj_strelu(P2, P3, "forestgreen")
    oznaci_strelu(P2, P3, "+come", "forestgreen", side=-1)

    # Nedostajuca 4. strana (0,0) do rezultata (isprekidana)
    nacrtaj_strelu(P0, P3, "darkorange", lw=2.8, ls="dashed")
    oznaci_strelu(P0, P3, f"went−go+come  =  {najbliza}", "darkorange")

    # Oznacene tacke u temenima
    cx = (min(sve_x2) + max(sve_x2)) / 2
    cy = (min(sve_y2) + max(sve_y2)) / 2
    for pt, col, sz in [
        (P0, "black", 100),
        (P1, "crimson", 120),
        (P2, "slategray", 100),
        (P3, "darkorange", 150),
    ]:
        ax2.scatter(*pt, color=col, s=sz, zorder=5)

    ax2.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="lightgray", linewidth=0.8, linestyle="--")
    ax2.set_title(
        f"Najbliza rec: went − go + come = '{najbliza}'\n",
        fontsize=13
    )
    ax2.set_xlabel("Prva glavna komponenta")
    ax2.set_ylabel("Druga glavna komponenta")
    ax2.grid(True, alpha=0.3)
    plt.savefig("went_go_come_came2.png", dpi=150, bbox_inches="tight")

    
    # Grafik 3: 3D strelice od (0,0)
    
    recnik_3d = {r: pca3.transform(matrica_vektora[rec_to_idx[r]].reshape(1, -1))[0]
                 for r in potrebne_reci}
    rezultat_3d = pca3.transform(rezultat_200d.reshape(1, -1))[0]

    fig3 = plt.figure(figsize=(14, 11))
    ax3 = fig3.add_subplot(111, projection='3d')

    for rec in potrebne_reci:
        x, y, z = recnik_3d[rec]
        ax3.plot([0, x], [0, y], [0, z], color=boje[rec], lw=2.5)

    mx3_lbl, my3_lbl, mz3_lbl = recnik_3d["go"]
    kx3_lbl, ky3_lbl, kz3_lbl = recnik_3d["come"]
    natpisi_3d = {
        "go":   (mx3_lbl * 1.02, my3_lbl * 1.02, mz3_lbl * 1.02),
        "come": (kx3_lbl * 1.02 + 0.04, ky3_lbl * 1.02, kz3_lbl * 1.02 + 0.02),
        "went": (recnik_3d["went"][0] * 1.10, recnik_3d["went"][1] * 1.10, recnik_3d["went"][2] * 1.10),
        "came": (recnik_3d["came"][0] * 1.02, recnik_3d["came"][1] * 1.02, recnik_3d["came"][2] * 1.02),
    }
    for rec in potrebne_reci:
        lx, ly, lz = natpisi_3d[rec]
        ax3.text(lx, ly, lz, rec, fontsize=12, fontweight="bold", color=boje[rec])

    kx3, ky3, kz3 = recnik_3d["go"]
    qx3, qy3, qz3 = recnik_3d["went"]
    mx3, my3, mz3 = recnik_3d["come"]
    wx3, wy3, wz3 = recnik_3d["came"]

    ax3.plot([kx3, qx3], [ky3, qy3], [kz3, qz3],
             color="gray", lw=1.5, linestyle="dotted")
    ax3.plot([mx3, wx3], [my3, wy3], [mz3, wz3],
             color="gray", lw=1.5, linestyle="dotted")
    _all3v = list(recnik_3d.values())
    _rng3 = max(
        max(v[0] for v in _all3v) - min(v[0] for v in _all3v),
        max(v[1] for v in _all3v) - min(v[1] for v in _all3v),
        max(v[2] for v in _all3v) - min(v[2] for v in _all3v),
        0.1
    ) * 0.18
    _xoff3, _yoff3, _zoff3 = _rng3, _rng3 * 0.5, _rng3 * 0.5
    ax3.text((kx3+qx3)/2 + _xoff3, (ky3+qy3)/2 + _yoff3, (kz3+qz3)/2 + _zoff3,
             "went−go", fontsize=9, color="gray", ha="center")
    ax3.text((mx3+wx3)/2 - _xoff3, (my3+wy3)/2 - _yoff3, (mz3+wz3)/2 - _zoff3,
             "came−come", fontsize=9, color="gray", ha="center")

    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_zlabel("PC3")
    ax3.set_title("Očekivano: went − go + come = came", fontsize=14)
    ax3.view_init(elev=20, azim=225)
    plt.savefig("went_go_come_came3.png", dpi=150, bbox_inches="tight")

    
    # Grafik 4: 3D zbir vektora
    
    fig4 = plt.figure(figsize=(14, 11))
    ax4 = fig4.add_subplot(111, projection='3d')

    P0_3d = np.array([0.0, 0.0, 0.0])
    P1_3d = recnik_3d["went"].copy()
    P2_3d = recnik_3d["went"] - recnik_3d["go"]
    P3_3d = rezultat_3d.copy()

    def nacrtaj_strelu3d(ax, start, end, color, lw=2.5):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=color, lw=lw)

    def oznaci_sredinu3d(ax, start, end, tekst, color, dz=0.0):
        mid = (np.array(start) + np.array(end)) / 2
        ax.text(mid[0], mid[1], mid[2] + dz, tekst, fontsize=11,
                color=color, fontweight="bold")

    _dz4 = 0.01 * (max(abs(P0_3d[2]), abs(P1_3d[2]), abs(P2_3d[2]), abs(P3_3d[2])) + 0.3)
    
    nacrtaj_strelu3d(ax4, P0_3d, P1_3d, "crimson")
    oznaci_sredinu3d(ax4, P0_3d, P1_3d, "went", "crimson", dz=_dz4*0.5)

    
    nacrtaj_strelu3d(ax4, P1_3d, P2_3d, "royalblue")
    oznaci_sredinu3d(ax4, P1_3d, P2_3d, "−go", "royalblue", dz=_dz4)

    
    nacrtaj_strelu3d(ax4, P2_3d, P3_3d, "forestgreen")
    oznaci_sredinu3d(ax4, P2_3d, P3_3d, "+come", "forestgreen", dz=-_dz4*0.5)

    
    ax4.plot([P0_3d[0], P3_3d[0]], [P0_3d[1], P3_3d[1]], [P0_3d[2], P3_3d[2]],
             color="darkorange", lw=2.5, linestyle="dashed")
    oznaci_sredinu3d(ax4, P0_3d, P3_3d, f"  = {najbliza}", "darkorange", dz=-_dz4)

    # Tacke u temenima
    for pt, col, sz, label in [
        (P1_3d, "crimson",   120, "went"),
        (P2_3d, "slategray", 100, "went−go"),
        (P3_3d, "darkorange",150, najbliza),
    ]:
        ax4.scatter(pt[0], pt[1], pt[2], color=col, s=sz)

    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_zlabel("PC3")
    ax4.set_title(
        f"Najbliza rec: went − go + come = '{najbliza}'",
        fontsize=13
    )
    #promeni se ugao iz kog se gleda da bi se slika bolje videla
    ax4.view_init(elev=20, azim=225)
    plt.savefig("went_go_come_came4.png", dpi=150, bbox_inches="tight")

    plt.show()

with open("poredjenja.txt", "w", encoding="utf-8") as f:
    for r in rezultati_analogija:
        f.write(r + "\n")
print(f"Rezultati zapisani u 'poredjenja.txt'")

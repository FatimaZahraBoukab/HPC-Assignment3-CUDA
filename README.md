# High Performance Computing – Assignment 3 : Programmation CUDA

**Étudiante :** BOUKAB Fatima Zahra
**Formation :** Master Sécurité IT & Big Data (SITBD) 
**Module :** High Performance Computing

---

## 1. Objectif

Ce TP introduit la programmation GPU avec CUDA : écriture de kernels, organisation des threads en blocs et en grilles (1D et 2D), calcul de l'indice global d'un thread, gestion de la mémoire hôte / device et gestion des threads inactifs.

---

## 2. Rappel : indice global d'un thread

- Grille 1D : `i = blockIdx.x * blockDim.x + threadIdx.x`
- Grille 2D : `col = blockIdx.x * blockDim.x + threadIdx.x` et `row = blockIdx.y * blockDim.y + threadIdx.y`
- Nombre de blocs pour couvrir N éléments : `blocks = (N + threads - 1) / threads` (division entière par excès).
- Threads inactifs = threads lancés − N. Un test `if (i < n)` dans le kernel les empêche d'écrire hors du tableau.

---

## Exercice 1 : kernel qui affiche `blockIdx.x` et `threadIdx.x`

**Énoncé :** écrire un kernel qui affiche « my cuda course » avec son `blockIdx.x` et son `threadIdx.x`, le lancer avec `<<<3, 5>>>`, puis l'exécuter deux fois et expliquer si l'ordre est identique.

### Code 

```cuda
#include <stdio.h>

__global__ void hello() {
    printf("my cuda course | blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<3, 5>>>();          // 3 blocs de 5 threads = 15 threads
    cudaDeviceSynchronize();    // attend la fin du GPU et vide le buffer de printf
    return 0;
}
```

### Résultat obtenu

```
my cuda course | blockIdx.x = 2, threadIdx.x = 0
my cuda course | blockIdx.x = 2, threadIdx.x = 1
my cuda course | blockIdx.x = 2, threadIdx.x = 2
my cuda course | blockIdx.x = 2, threadIdx.x = 3
my cuda course | blockIdx.x = 2, threadIdx.x = 4
my cuda course | blockIdx.x = 0, threadIdx.x = 0
my cuda course | blockIdx.x = 0, threadIdx.x = 1
my cuda course | blockIdx.x = 0, threadIdx.x = 2
my cuda course | blockIdx.x = 0, threadIdx.x = 3
my cuda course | blockIdx.x = 0, threadIdx.x = 4
my cuda course | blockIdx.x = 1, threadIdx.x = 0
my cuda course | blockIdx.x = 1, threadIdx.x = 1
my cuda course | blockIdx.x = 1, threadIdx.x = 2
my cuda course | blockIdx.x = 1, threadIdx.x = 3
my cuda course | blockIdx.x = 1, threadIdx.x = 4
```

### Analyse

**L'ordre des lignes n'est pas garanti identique d'une exécution à l'autre.** Dans l'exécution ci-dessus, les blocs apparaissent dans l'ordre 2, 0, 1 et non 0, 1, 2, ce qui le montre déjà. Raisons :

1. Les blocs sont distribués aux multiprocesseurs (SM) par le planificateur matériel, sans ordre imposé entre blocs : ils sont indépendants et peuvent s'exécuter dans n'importe quel ordre ou en même temps.
2. Les threads d'un bloc s'exécutent par warps (groupes de 32), donc en parallèle.
3. Le `printf` du GPU écrit dans un buffer, vidé vers la sortie standard de l'hôte de façon asynchrone. L'ordre d'affichage dépend donc de l'ordre d'exécution réel.

`cudaDeviceSynchronize()` est indispensable : le lancement d'un kernel est asynchrone, et sans cet appel le programme pourrait se terminer avant que les messages soient affichés.

---

## Exercice 2 : remplir un tableau avec `a[i] = i * i`

**Énoncé :** `threads = 256`, `blocks = (N + threads - 1) / threads`. Pour N = 256, 257 et 1000, calculer à la main les blocs, les threads lancés et les threads inactifs, puis vérifier par `printf`. Pour N = 1000, copier le tableau sur l'hôte et afficher les 5 dernières valeurs.

### Calcul à la main

| N | blocs = ⌈N/256⌉ | threads lancés = blocs × 256 | threads inactifs |
|---|---|---|---|
| 256 | 1 | 256 | 0 |
| 257 | 2 | 512 | 255 |
| 1000 | 4 | 1024 | 24 |

### Code 

```cuda
#include <stdio.h>
#include <stdlib.h>

__global__ void fill(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // indice global du thread
    if (i < n) a[i] = i * i;
}

int main() {
    int threads = 256;

    // ---- Partie 1 : vérification des calculs ----
    int Ns[3] = {256, 257, 1000};
    for (int k = 0; k < 3; k++) {
        int N = Ns[k];
        int blocks = (N + threads - 1) / threads;
        int launched = blocks * threads;
        printf("N=%4d : blocs=%d, threads lances=%d, threads inactifs=%d\n",
               N, blocks, launched, launched - N);
    }

    // ---- Partie 2 : N = 1000 ----
    int N = 1000;
    int blocks = (N + threads - 1) / threads;
    int *h = (int *)malloc(N * sizeof(int));   // tableau sur l'hôte (CPU)
    int *d;                                    // tableau sur le device (GPU)
    cudaMalloc(&d, N * sizeof(int));

    fill<<<blocks, threads>>>(d, N);
    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);   // GPU -> CPU

    for (int i = N - 5; i < N; i++) printf("a[%d] = %d\n", i, h[i]);

    cudaFree(d);
    free(h);
    return 0;
}
```

### Résultat obtenu

```
N= 256 : blocs=1, threads lances=256, threads inactifs=0
N= 257 : blocs=2, threads lances=512, threads inactifs=255
N=1000 : blocs=4, threads lances=1024, threads inactifs=24
a[995] = 990025
a[996] = 992016
a[997] = 994009
a[998] = 996004
a[999] = 998001
```

### Analyse

Les valeurs affichées par le programme correspondent exactement au calcul à la main. Pour N = 257, un seul élément de plus que 256 oblige à lancer un second bloc entier, d'où 255 threads inactifs. Les 5 dernières valeurs sont bien 995² à 999².

---

## Exercice 3 : addition de vecteurs `vecAdd`

**Énoncé :** `c[i] = a[i] + b[i]` pour N = 1000, avec `a[i] = i` et `b[i] = 2 * i`. Utiliser 3 tableaux sur le GPU, 2 copies hôte → device, 1 copie device → hôte, libérer la mémoire, puis vérifier que `c[i] == 3 * i`.

### Code 

```cuda
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1000, threads = 256;
    int blocks = (N + threads - 1) / threads;
    size_t bytes = N * sizeof(int);

    // 1) Tableaux sur l'hôte, initialisés : a[i] = i, b[i] = 2*i
    int *ha = (int *)malloc(bytes);
    int *hb = (int *)malloc(bytes);
    int *hc = (int *)malloc(bytes);
    for (int i = 0; i < N; i++) { ha[i] = i; hb[i] = 2 * i; }

    // 2) Les 3 tableaux sur le GPU
    int *da, *db, *dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    // 3) Les 2 copies hôte -> device
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

    // 4) Lancement du kernel
    vecAdd<<<blocks, threads>>>(da, db, dc, N);

    // 5) La copie device -> hôte
    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

    // 6) Vérification sur l'hôte : c[i] == 3*i pour tout i
    int bad = -1;
    for (int i = 0; i < N; i++) {
        if (hc[i] != 3 * i) { bad = i; break; }
    }
    if (bad < 0) printf("OK\n");
    else printf("Erreur a l'indice %d : c=%d, attendu=%d\n", bad, hc[bad], 3 * bad);

    // 7) Libérer toute la mémoire
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
```

### Résultat obtenu

```
OK
```

### Analyse

Les 1000 éléments vérifient `c[i] == 3 * i` (car `i + 2i = 3i`). Le schéma classique d'un programme CUDA est respecté : allocation hôte, allocation device, copie H→D, kernel, copie D→H, vérification, libération.

---

## Exercice 4 : produit élément par élément `Y[i] = X[i] * Y[i]`

**Énoncé :** deux vecteurs de N = 100 flottants, 30 threads par bloc, `X[i] = i`, `Y[i] = 2`.

### Réponses aux questions

1. **Nombre de blocs :** ⌈100 / 30⌉ = **4 blocs**. Threads lancés : 4 × 30 = 120, donc **20 threads inactifs**.
2. **Tableaux sur le GPU :** **2** (X et Y), car le résultat écrase Y et aucun tableau de sortie n'est nécessaire.
3. **Résultat attendu :** `Y[i] = X[i] * Y[i] = i * 2`, soit 0, 2, 4, …, 198.
4. **Pourquoi 30 threads par bloc est un mauvais choix :** le matériel exécute les threads par warps de **32 threads**. Un bloc de 30 threads occupe un warp dont 2 voies restent inutilisées (≈ 6 % de perte par bloc), et la taille n'est pas alignée sur la granularité du matériel. On choisit un **multiple de 32**, typiquement **128 ou 256** pour de grands tableaux (32 ou 64 suffiraient pour N = 100).

### Code 

```cuda
#include <stdio.h>
#include <stdlib.h>

__global__ void mulInPlace(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * y[i];     // le résultat écrase y
}

int main() {
    const int N = 100, threads = 30;
    int blocks = (N + threads - 1) / threads;
    printf("blocs=%d, threads lances=%d, inactifs=%d\n",
           blocks, blocks * threads, blocks * threads - N);

    size_t bytes = N * sizeof(float);
    float *hx = (float *)malloc(bytes);
    float *hy = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) { hx[i] = (float)i; hy[i] = 2.0f; }

    float *dx, *dy;                    // 2 tableaux GPU
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice);

    mulInPlace<<<blocks, threads>>>(dx, dy, N);

    cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("Y[%d] = %.1f\n", i, hy[i]);

    cudaFree(dx); cudaFree(dy);
    free(hx); free(hy);
    return 0;
}
```

### Résultat obtenu (extrait, la sortie complète est dans le notebook)

```
blocs=4, threads lances=120, inactifs=20
Y[0] = 0.0
Y[1] = 2.0
Y[2] = 4.0
Y[3] = 6.0
...
Y[48] = 96.0
Y[49] = 98.0
Y[50] = 100.0
...
Y[98] = 196.0
Y[99] = 198.0
```

### Analyse

La sortie confirme les valeurs attendues : `Y[i] = 2·i` pour i de 0 à 99, avec 4 blocs et 20 threads inactifs.

---

## Exercice 5 : matrice 20 × 40 en 2D

**Énoncé :** remplir une matrice de ROWS = 20 lignes et COLS = 40 colonnes, stockée en row-major, avec `A[row][col] = 100 * row + col`, en utilisant des blocs de 16 × 16 threads.

### Réponses aux questions

1. **Grille lancée :** la dimension x couvre les colonnes, ⌈40 / 16⌉ = 3, et la dimension y couvre les lignes, ⌈20 / 16⌉ = 2. La grille est donc `dim3 grid(3, 2)`, soit **6 blocs**. Threads lancés : 6 × 256 = **1536**. Threads utiles : 20 × 40 = 800. Threads inactifs : **736**.
2. **Quel thread a calculé `A[17][33]` ?**
   - Ligne 17 : 17 = 1 × 16 + 1, donc `blockIdx.y = 1` et `threadIdx.y = 1`.
   - Colonne 33 : 33 = 2 × 16 + 1, donc `blockIdx.x = 2` et `threadIdx.x = 1`.
   - Il s'agit du bloc **(2, 1)**, thread **(1, 1)**. Valeur : 100 × 17 + 33 = **1733**.

### Code 

```cuda
#include <stdio.h>
#include <stdlib.h>

#define ROWS 20
#define COLS 40

__global__ void fillMatrix(int *A, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x -> colonnes
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y -> lignes
    if (row < rows && col < cols)
        A[row * cols + col] = 100 * row + col;         // stockage row-major
}

int main() {
    dim3 block(16, 16);
    dim3 grid((COLS + block.x - 1) / block.x,          // 3
              (ROWS + block.y - 1) / block.y);         // 2
    int launched = grid.x * grid.y * block.x * block.y;
    printf("grille = (%d,%d), threads lances=%d, inactifs=%d\n",
           grid.x, grid.y, launched, launched - ROWS * COLS);

    size_t bytes = ROWS * COLS * sizeof(int);
    int *h = (int *)malloc(bytes);
    int *d;
    cudaMalloc(&d, bytes);

    fillMatrix<<<grid, block>>>(d, ROWS, COLS);
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) printf("%5d", h[r * COLS + c]);
        printf("\n");
    }
    printf("A[17][33] = %d\n", h[17 * COLS + 33]);

    cudaFree(d);
    free(h);
    return 0;
}
```

### Résultat obtenu (extrait, la matrice complète est dans le notebook)

```
grille = (3,2), threads lances=1536, inactifs=736
    0    1    2 ...   38   39
  100  101  102 ...  138  139
  200  201  202 ...  238  239
 ...
 1700 1701 1702 ... 1738 1739
 1800 1801 1802 ... 1838 1839
 1900 1901 1902 ... 1938 1939
A[17][33] = 1733
```

### Analyse

La matrice contient bien `100 * row + col` à chaque case : la première colonne vaut 0, 100, …, 1900 et la dernière ligne va de 1900 à 1939. Les 736 threads inactifs (hors de la matrice) sont neutralisés par le test `row < rows && col < cols`. La valeur `A[17][33] = 1733` correspond au calcul fait à la main.

---

## 5. Conclusion

| Exercice | Notion | Résultat |
|---|---|---|
| 1 | Lancement `<<<3, 5>>>`, `blockIdx`, `threadIdx`, synchronisation | Ordre non déterministe entre blocs |
| 2 | Indice global, calcul des blocs, threads inactifs | 0 / 255 / 24 threads inactifs, valeurs correctes |
| 3 | Cycle complet mémoire hôte/device, `vecAdd` | `OK` |
| 4 | Taille de bloc non alignée, warp de 32 | 4 blocs, 20 inactifs, `Y[i] = 2i` |
| 5 | Grille et blocs 2D, row-major | 6 blocs, 736 inactifs, `A[17][33] = 1733` |


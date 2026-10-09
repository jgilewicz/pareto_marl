# Etap 0: czy pojedynczy agent przestaje działać przy dużej liczbie aktuatorów?

Kontekst: v1 i v2 (Ant, 8 aktuatorów) dały NO-GO. Przy stałym K partycja nie ma
znaczenia (ICC ≈ 0), a K=1 wygrywa. Kierunek ma sens tylko tam, gdzie pojedynczy
agent nie daje rady. Etap 0 sprawdza, czy taka skala w ogóle istnieje.

## 1. Pytanie i reguła GO / NO-GO (ustalone przed runem)

> Czy istnieje liczba aktuatorów, przy której jakiś podział na agentów
> uczy się istotnie lepiej niż jeden agent?

- Porównania (zaplanowane z góry): dla każdego rozmiaru ciała i każdego z trzech
  podziałów (agent na segment, na nogę, na staw) test Welcha, jednostronny
  (podział > K=1), na zwrocie ewaluacyjnym, 5 seedów na stronę.
  4 rozmiary × 3 podziały = 12 porównań, korekta Holma.
- **GO**, jeśli dla co najmniej jednego rozmiaru ≥ 32 aktuatorów któreś
  porównanie przechodzi po korekcie (p < 0.05). Bez progu wielkości efektu;
  stosunek średnich raportujemy obok.
- W przeciwnym razie **NO-GO**: przy tych rozmiarach jeden agent wystarcza,
  przeszukiwanie partycji nie ma uzasadnienia.
- Raportowane bez wpływu na decyzję: krzywa zwrot(K=1) vs liczba aktuatorów,
  prędkość do przodu, krzywe uczenia, czas treningu.

## 2. Środowisko: ManySegmentAnt w MJX

- Ciało: `gen_asset(n_segs)` z MaMuJoCo (FACMAC), 4 aktuatory na segment
  (2 nogi × biodro + kostka). Fizyka i parametry jak `Ant-v5` na tym XML.
- Rozmiary: `n_segs ∈ {2, 4, 8, 16}` → **8, 16, 32, 64 aktuatory**.
- Obserwacja: `qpos[2:]`, `qvel` (bez sił kontaktu, jak v2).
- Nagroda: prędkość do przodu `torso_0` + 1 za „zdrowie” − 0.5·‖a‖².
  Koszt kontaktu (5e-4·‖cfrc‖²) pominięty: wymaga sił kontaktu w MJX, a jest
  zaniedbywalny. Liczby nie są więc 1:1 porównywalne z v2.
- Terminacja: z(`torso_0`) poza [0.2, 1.0]; epizod do 1000 kroków;
  frame_skip 5; szum resetu 0.1 (wartości domyślne `Ant-v5`).
- Kolizje tylko ciało–podłoga (`conaffinity=0` na ciele), czyli tanie w MJX.
- Sprawdzone: `mjx.put_model` przyjmuje model z RK4 (mujoco-mjx 3.15.0).

## 3. Warunki

| Warunek | K dla 8 / 16 / 32 / 64 aktuatorów |
| --- | --- |
| jeden agent | 1 / 1 / 1 / 1 |
| agent na segment | 2 / 4 / 8 / 16 |
| agent na nogę | 4 / 8 / 16 / 32 |
| agent na staw | 8 / 16 / 32 / 64 |

16 warunków × 5 seedów = **80 runów**. Obserwacje lokalne z MaMuJoCo
(`agent_obsk=1`, kategorie bez `cfrc_ext`), krytyk na pełnym stanie.

## 4. Algorytm: MAPPO w JAX

- Ta sama metoda co v2: CleanRL PPO, K aktorów, wspólny krytyk, ratio i clip
  per agent, suma po agentach, jeden Adam, jeden clip normy.
- Hiperparametry bez zmian: 8 env × 256 kroków, 10 epok, 32 minibatche,
  γ 0.99, λ 0.95, clip 0.2, lr 3e-4 z wygaszaniem, **T = 3M**.
- Normalizacja obserwacji i skalowanie nagrody jak wrappery CleanRL.
- Aktorzy przez `vmap` z paddingiem obserwacji i akcji do maksimum w partycji
  (maski), żeby K=64 nie oznaczało 64 osobnych grafów.
- `vmap` po seedach: jeden skompilowany program = jeden warunek × 5 seedów.
- Ewaluacja: 10 epizodów, akcja = średnia, statystyki obserwacji zamrożone.

## 5. Bramki portu (przed etapem 0)

1. **Fizyka:** trajektoria MJX vs MuJoCo C na tym samym modelu i akcjach,
   100 kroków, `n_segs` 2 i 16, zgodność w tolerancji float32.
2. **Obserwacje agentów:** indeksy z MaMuJoCo == wycinki stanu MJX dla każdej
   partycji z §3.
3. **Uczenie:** K=1 na `n_segs=2` rośnie w krótkim treningu lokalnie (CPU).
4. **Przepustowość:** pomiar na 1×H100 na WCSS. Na tej podstawie budżet.

Testy w pytest pomijamy (decyzja z v1). Bramki to skrypty weryfikacyjne
uruchamiane raz, wynik w opisie PR.

## 6. Struktura kodu (JAX, funkcyjnie)

```
pareto_marl/
  envs/many_segment_ant.py   XML, model MJX, reset/step, obs, nagroda
  networks/actor_critic.py   aktorzy (vmap + maski), krytyk
  losses/ppo.py              strata MAPPO
  training/mappo.py          rollout, GAE, update, ewaluacja, vmap po seedach
  utils/normalize.py         running mean/std
  partition.py               partycje dla n_segs, indeksy z MaMuJoCo
  stage0.py                  design + CLI: warunek → JSON wyników
  analysis.py, report.py     rozszerzone o regułę §1
slurm/stage0.sbatch          lem-gpu, 1×H100, 16 CPU
```

Kod torch (`mappo.py`, `run.py`, `reference/`) usuwamy. v1 i v2 są
odtwarzalne z commitów `a3d54a7` i `52a7a0a`.

## 7. Infrastruktura i budżet

- `lem-gpu`, `--gres=gpu:hopper:1`, `--cpus-per-task=16`, SLURM array po
  16 warunkach. Grant Solvro: ~3000 h GPU i ~21k h CPU zostało.
- Węzeł logowania nie ma AVX: venv budujemy tam, ale importów nie
  uruchamiamy. JAX z CUDA 12 w `.venv` w `$HOME`.
- Szacunek do potwierdzenia pomiarem: ~16 programów × 0.5–1 h GPU, czyli
  **~10–20 h GPU**. Zgoda na wysłanie po pomiarze z bramki 4.

## 8. Delegacja (orkiestrator)

| Fala | Task | Model | Zależy od | Pliki |
| --- | --- | --- | --- | --- |
| 1 | T0: zależności JAX/MJX w `pyproject.toml`, szkielet katalogów, kontrakt środowiska (`EnvState`, `reset`, `step`) | opus | — | `pyproject.toml`, `uv.lock` (gorące) |
| 2 | T1: środowisko MJX + bramki 1–2 | opus | T0 | `envs/`, `partition.py` |
| 2 | T2: MAPPO w JAX na kontrakcie + bramka 3 na atrapie | opus | T0 | `networks/`, `losses/`, `training/`, `utils/` |
| 3 | T3: integracja: `stage0.py`, sbatch GPU, reguła §1 w analizie i raporcie, usunięcie kodu torch, bramka 3 na prawdziwym środowisku | opus | T1, T2 | `stage0.py`, `slurm/`, `analysis.py`, `report.py` |
| — | Review każdego PR: świeża sesja opus w worktree autora | opus | PR | — |
| 4 | Bramka 4 + etap 0 na WCSS, raport | prezydent | T3 + zgoda na budżet | — |

`pyproject.toml` i `uv.lock` zmienia tylko T0; T1–T3 nie dodają zależności
bez uzgodnienia.

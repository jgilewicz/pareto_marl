# MVP: czy podział akcji między agentów MARL ma znaczenie? (MaMuJoCo Ant)

## 1. Cel MVP

Jedno pytanie, jedna odpowiedź GO / NO-GO:

> Czy przy **stałej liczbie agentów K** wybór partycji 8 aktuatorów Ant
> zmienia zwrot MAPPO **bardziej niż szum między seedami**?

- GO → pojęcie „optymalizacji partycji” ma sens, następny krok to
  przeszukiwanie (NSGA-II / pełny przegląd, sekcja 11).
- NO-GO → liczy się tylko K, przeszukiwanie partycji nie ma czego szukać.

### Dlaczego nie NSGA-II w MVP

- f2 (parametry aktorów) ma **rozłączne zakresy dla każdego K**
  (policzone dla wszystkich 4140 partycji, sekcja 2) → f2 to w praktyce K.
- Front Pareto = „najlepsza partycja dla każdego K”, ≤ 8 punktów.
- 1 seed na ocenę przy szumie PPO na Ant → NSGA-II wybiera szczęśliwe seedy.
- Zanim szukamy optimum, trzeba wiedzieć, czy jest sygnał. To robi próbkowanie
  warstwowe z powtórzeniami.

### Zakres

| W MVP | Poza MVP (później) |
| --- | --- |
| Ant, 8 aktuatorów, `agent_obsk=1` | inne roboty, pełna obserwacja |
| K ∈ {2, 4} + referencje K = 1, K = 8 | pozostałe K |
| partycje strukturalne + losowe (jednostajnie w obrębie K) | NSGA-II, MOBO, ograniczenia grafowe |
| MAPPO: osobni aktorzy, wspólny krytyk, hiperparametry CleanRL | HAPPO, IPPO, strojenie |
| 5 seedów na partycję, T = 1M | dłuższe budżety, multi-fidelity |
| SLURM array na CPU WCSS, 1 run = 1 task = 1 rdzeń | fizyka na GPU (MJX) |
| logowanie do wandb + JSON na dysku | — |

---

## 2. Formalizacja

**Przestrzeń akcji.** Ant ma m = 8 aktuatorów, `A = [-1, 1]^8`.

**Partycja.** `π = {B_1, …, B_K}` zbioru `{0,…,7}`, każdy blok steruje jeden
agent. Kodowanie: restricted growth string `x ∈ {0,…,7}^8`
(`x_0 = 0`, `x_{i+1} ≤ 1 + max_{j≤i} x_j`), `B_k = {i : x_i = k}`.

**Liczność.** B_8 = 4140 (zweryfikowane):

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liczba partycji | 1 | 127 | 966 | 1701 | 1050 | 266 | 28 | 1 |
| f2 min | 11 472 | 16 912 | 22 352 | 27 792 | 33 232 | 38 800 | 44 368 | 49 936 |
| f2 max | 11 472 | 18 832 | 24 400 | 29 968 | 35 152 | 40 080 | 45 008 | 49 936 |

f2 = Σ_k parametry aktora k (MLP `obs_k → 64 → 64 → |B_k|` + `log_std`).
Zakresy się nie nakładają: f2 wyznacza K.

**Wielkość mierzona.** `R(x, s)` = średni zwrot z 10 epizodów ewaluacyjnych
(akcja = średnia rozkładu) po T = 1M kroków, partycja x, seed s.
Dodatkowo `v(x, s)` = średnia prędkość `x_velocity` w ewaluacji (odróżnia
„stoi i zbiera healthy reward” od chodzenia).

---

## 3. Kryterium GO / NO-GO (ustalone przed runem)

Osobno dla K = 2 i K = 4, dane: `R(x, s)` dla partycji x z danego K
(strukturalne + losowe) i 5 seedów.

1. Jednoczynnikowa ANOVA, czynnik = partycja, powtórzenia = seedy.
2. ICC(1) = (MSB − MSW) / (MSB + (n − 1)·MSW), n = 5: udział wariancji zwrotu
   wyjaśniony przez partycję.

**GO**, jeśli w co najmniej jednym K: **p < 0.05 i ICC ≥ 0.2**.
W przeciwnym razie **NO-GO**.

Raportowane dodatkowo (bez wpływu na decyzję):

- średnia ± std per partycja, z podziałem strukturalne vs losowe,
- najlepsza partycja per K vs K = 1 (czy jakieś K > 1 jest konkurencyjne),
- wykres f2 vs zwrot (średnie po seedach, słupki = std),
- to samo dla `v(x, s)`.

---

## 4. Środowisko: MaMuJoCo Ant (zweryfikowane lokalnie)

Wersje: `gymnasium-robotics==1.4.2`, `gymnasium==1.4.0`, `mujoco==3.15.0`,
`Ant-v5` (obserwacja 105, epizod do 1000 kroków, terminacja przy
„niezdrowym” stanie).

**Aktuatory.** Nagroda liczy prędkość wzdłuż +x. Położenia nóg z `ant.xml`
(nazwy w XML są mylące, liczy się położenie):

| Indeks | Węzeł MaMuJoCo | Noga (XML) | Położenie (x, y) | Względem ruchu |
| --- | --- | --- | --- | --- |
| 0, 1 | hip4, ankle4 | `right_back_leg` | (+0.2, −0.2) | przód-prawo |
| 2, 3 | hip1, ankle1 | `front_left_leg` | (+0.2, +0.2) | przód-lewo |
| 4, 5 | hip2, ankle2 | `front_right_leg` | (−0.2, +0.2) | tył-lewo |
| 6, 7 | hip3, ankle3 | `back_leg` | (−0.2, −0.2) | tył-prawo |

**Własna faktoryzacja** działa dla dowolnej partycji (też niespójnej).
`agent_conf` musi być ≠ `None` (treść ignorowana przy `agent_factorization`),
etykieta dla `get_parts_and_edges` to `"Ant"`.

**Trik wektoryzacji.** Lokalna obserwacja agenta to podzbiór indeksów
globalnej obserwacji `Ant-v5`. Zweryfikowane: `obs[a] == state()[idx_a]` po
resecie i po 20 losowych krokach, a `state()` == obserwacja `Ant-v5`.
MaMuJoCo tworzymy raz na partycję tylko po indeksy, trenujemy na zwykłym
`gymnasium.make("Ant-v5")`.

Rozmiary obserwacji (`agent_obsk=1`, zmierzone):

| x | K | obs agentów |
| --- | --- | --- |
| `00000000` | 1 | 105 |
| `00112233` | 4 | 42, 42, 42, 42 |
| `01010101` | 2 | 53, 95 |
| `01234567` | 8 | 29, 32, 29, 32, 29, 32, 29, 32 |

---

## 5. Design eksperymentu

| Grupa | Partycje | Liczba |
| --- | --- | --- |
| referencja K = 1 | `00000000` | 1 |
| referencja K = 8 | `01234567` | 1 |
| K = 2 strukturalne | `00001111` przód/tył, `00111100` lewo/prawo, `00110011` przekątne, `01010101` biodra/kostki | 4 |
| K = 2 losowe | 8 partycji jednostajnie spośród 127, bez strukturalnych | 8 |
| K = 4 strukturalne | `00112233` agent na nogę | 1 |
| K = 4 losowe | 8 partycji jednostajnie spośród 1701, bez strukturalnych | 8 |
| **razem** | | **23** |

- Seedy 0–4 dla każdej partycji → **115 runów MAPPO**.
- Referencja CleanRL `ppo_continuous_action` na `Ant-v5`, seedy 0–4,
  8 env × 256 kroków → **5 runów**.
- Losowanie partycji: `numpy.random.default_rng(0)`, wynik zapisany raz do
  `configs/design.json` i commitowany (task arraya czyta z pliku).

---

## 6. MAPPO

Implementacja = CleanRL `ppo_continuous_action` z jedną zmianą: aktor jest
rozbity na K aktorów. Dla K = 1 to dokładnie CleanRL (test poprawności).

| Element | Wybór |
| --- | --- |
| Aktorzy | K osobnych MLP `obs_k → 64 → 64 → |B_k|`, tanh, init ortogonalny, `log_std` jako parametr |
| Krytyk | MLP `105 → 64 → 64 → 1` na pełnej obserwacji (CTDE) |
| Log-prob akcji wspólnej | suma log-prob agentów |
| Nagroda | wspólna, z `Ant-v5` |
| Wrappery (per env, jak CleanRL) | `RecordEpisodeStatistics`, `ClipAction`, `NormalizeObservation`, clip obs ±10, `NormalizeReward`, clip nagrody ±10 |
| Normalizacja obs | na globalnej obserwacji, przed podziałem na agentów |
| Wektoryzacja | `SyncVectorEnv`, 8 env, `autoreset_mode=SAME_STEP` (semantyka CleanRL) |
| Optymalizator | jeden Adam (lr 3e-4, eps 1e-5, liniowe wygaszanie), jeden clip normy gradientu 0.5 na wszystkich parametrach |
| Urządzenie | CPU, `torch.set_num_threads(1)` |

Hiperparametry (zamrożone, bez strojenia): γ 0.99, λ 0.95, 8 env × 256 kroków
(bufor 2048), 10 epok, 32 minibatche, clip 0.2, clip value loss, vf 0.5,
entropia 0.0, normalizacja przewag, T = 1M.

**Ewaluacja:** 10 epizodów, akcja = średnia, osobne `Ant-v5` z ziarnem
`seed + 10_000`, statystyki `NormalizeObservation` skopiowane z env 0 i
zamrożone. Zwrot surowy (bez normalizacji nagrody).

**Wynik runu** (`results/<run_id>/<x>-s<seed>.json`):

```
x, K, seed, obs_dims, act_dims, actor_params, total_steps,
eval_return, eval_return_std, eval_x_velocity,
learning_curve: [[global_step, episodic_return], ...],
wall_time_s
```

### Weryfikacja (bez testów jednostkowych)

- `just check` (ruff, ty) i `just smoke` (20k kroków, K = 1).
- K = 1 (pełny run) porównywalny z referencją CleanRL (w analizie: test
  Welcha na średnim zwrocie z ostatnich 10% treningu).

---

## 7. Infrastruktura: WCSS

- Partycja **`bem2-cpu-short`** (godziny CPU są w grancie), bez GPU.
- **SLURM array**, 1 task = 1 (partycja, seed), `--cpus-per-task=1`.
  Indeks tasku i → partycja `i // 5`, seed `i % 5` z `configs/design.json`.
- Brak puli procesów, cache i logiki wznawiania: brakujące wyniki = ponowne
  wysłanie brakujących indeksów (`--array=...`).
- Python 3.13 zarządzany przez `uv` (bez modułów), venv na PD
  (`UV_PROJECT_ENVIRONMENT`), budowany na węźle logowania.
- Wyniki: JSON per run do `$TMPDIR`, kopiowane na PD w pułapce `EXIT`.
- wandb: projekt `marl-partition-moo`, `group = RUN_ID`, run per task
  (`job_type` = `mappo` lub `cleanrl`). `WANDB_MODE=offline` jeśli węzły nie
  mają internetu, potem `wandb sync` z węzła logowania.
- Koszt: 120 tasków × 1 rdzeń × ~1 h ≈ 120 rdzeniogodzin (zmierzyć w smoke
  teście i poprawić).

---

## 8. Protokół runu

1. Lokalnie: `just check`, `just smoke` (zmierzone: 2.4–4k SPS na M-series,
   1M kroków ≈ 4–7 min).
2. WCSS: `uv` + klon w `$HOME`, `just venv <account>`, `preflight.sh`,
   `just smoke-wcss <account>` (3 taski MAPPO + 1 CleanRL × 50k).
3. Pełny run: `just submit <account> <run_id>` (115 + 5 tasków).
4. `just sync`, `just analyze <katalog wyników>` → `verdict.json`,
   `partitions.csv`, `front.png`, werdykt GO / NO-GO.

---

## 9. Struktura kodu

```
pareto_marl/
  partition.py   canon, enumeracja, losowanie w K, PartitionSpec, actor_params
  design.py      generowanie configs/design.json
  mappo.py       Actor, MultiAgentPolicy, Critic, trening, ewaluacja
  run.py         CLI: jeden run (indeks z designu, seed) → JSON + wandb
  analysis.py    ANOVA + ICC, werdykt, tabela, wykresy
reference/
  cleanrl_ppo_continuous_action.py   CleanRL przeniesiony na gymnasium 1.x
configs/design.json
slurm/array.sbatch
justfile
```

Zależności (dokładne wersje w `pyproject.toml`): gymnasium-robotics, gymnasium,
mujoco, torch (CPU), numpy, scipy, wandb, matplotlib; grupa `reference`: tyro,
tensorboard.

---

## 10. Kryteria ukończenia MVP

- [ ] `just check` czysty, `just smoke` lokalnie i na WCSS przechodzi.
- [ ] K = 1 uczy się porównywalnie z CleanRL (te same seedy i budżet).
- [ ] 115 + 5 runów zakończonych, wyniki na PD i w jednej grupie wandb.
- [ ] Werdykt GO / NO-GO wg sekcji 3 z liczbami (p, ICC dla K = 2 i K = 4).

---

## 11. Co dalej (tylko po GO)

1. Przeszukiwanie partycji: NSGA-II (operatory grupujące, kanonizacja)
   albo pełny przegląd spójnych partycji (240 dla Ant). Kryterium f2 zastąpić
   czymś, co nie jest proxy K (np. AUC, odporność na wyłączenie agenta).
2. Więcej seedów lub dominacja z przedziałami ufności zamiast 1 seeda na ocenę.
3. Izolacja efektu obserwacji: każdy aktor widzi pełny stan.
4. Multi-fidelity (successive halving) zamiast stałego T.
5. Fizyka na GPU (MJX / MuJoCo Playground).

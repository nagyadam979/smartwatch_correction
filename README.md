# Huawei Band 10 HR-korrekció MOX EKG-val

Ez a projekt a Gadgetbridge SQLite- vagy ZIP-export Huawei Band 10 pulzusadatait hasonlítja a vele párhuzamosan rögzített MOX Holter BDF EKG-jához. A ZIP kanonikus `database/Gadgetbridge` adatbázisa közvetlenül, memóriából és kibontás nélkül nyílik meg. A program automatikusan megkeresi az összes közös időszakot, kezeli az óraadat réseit, és 0,7 Hz-es alapvonal-eltávolítás + 5–30 Hz-es sávszűrés után 30 másodperces lokális blokkokban, template-, RR- és off-QRS műtermék-kapuval készít EKG-referencia HR-t. Ezután időoffsetet becsül, majd kronologikus train/holdout felosztáson ellenőrzi az identity, bias és affine korrekciót.

## Futtatás

Python 3, NumPy, SciPy és Matplotlib szükséges. A jelen fájlok alapértelmezett futtatása:

```bash
python3 analyze_hr.py
```

Explicit inputokkal és külön output könyvtárral:

```bash
python3 analyze_hr.py \
  --bdf 08_28/15673_20260828_164058.bdf \
  --watch-db 08_28/Gadgetbridge_20260828-213228 \
  --timezone Europe/Budapest \
  --output results
```

A 2026-09-01-i ZIP-export futtatása külön eredménykönyvtárba:

```bash
python3 -B analyze_hr.py \
  --bdf 09_01/15673_20260901_164946.bdf \
  --watch-db 09_01/gadgetbridge_20260901194648.zip \
  --timezone Europe/Budapest \
  --output 09_01/results
```

Regressziós ellenőrzés:

```bash
python3 -m unittest -v test_analyze_hr.py
```

## Kimenetek

- `results/report.md`: magyar összefoglaló a checksumokról, időillesztésről, jelminőségről, metrikákról és a korrekció kiválasztásáról.
- `results/comparison.csv`: minden óraminta, az illesztett referencia-idő, EKG HR, korrekció, hibák, minőség és train/validation jelölés. A MOX gyorsulásmérőjéből ezen kívül `mox_motion_count` (a 25 Hz-es X/Y/Z vektorhossz másodperces szórása) és `mox_posture` (`fekvő` / `álló/ülő nyugalom` / `álló + mozgás`, a MOX `Classification` csatornájából). Ezek finomabbak, mint az 1 Hz-es `activity_intensity`.
- `results/overview.png`: idősor, hibák, Bland–Altman jellegű kép, QRS-minőség és mozgásintenzitás.
- `results/qrs_diagnostics.png`: a két legerősebb, az új lokális kapuval visszaengedett magas-HR epizód reprezentatív nyers és zero-phase szűrt EKG-részlete, elfogadott R-csúcsokkal.

A `results/` reprodukálható generált output, ezért Gitben alapértelmezetten figyelmen kívül marad.

A `plot_hr_overlay.py` külön, nagy felbontású (alapértelmezetten 6000 px széles)
átfedésábrát rajzol, amin csak a Huawei nyers és a MOX EKG referencia HR
szerepel, korrekciós görbe nélkül. Az óra réseinél megszakítja a vonalat, hogy
a kizárt EKG-blokkokat ne kösse össze látszólagos folytonossággal.

```bash
python3 -B plot_hr_overlay.py \
  --session 2026-08-28=08_28/results_v2/comparison.csv \
  --session 2026-09-01=09_01/results/comparison.csv
```

Kimenet sessiononként a `comparison.csv` mellé: `hr_overlay_raw_vs_ecg.png`.
A `--dpi` és a `--width` kapcsolóval a felbontás állítható.

## Beat-szintű EKG-audit (egészségügyi átnézés)

Az `ecg_audit.py` nem pulzust számol, hanem a MOX EKG-t nézi át ritmusanomáliákra, és
minden találatot a nyers hullámformán ellenőriz. Külön, inkluzív detektálást használ
(morfológiai szűrés nélkül, T-hullám-elnyomással), mert a HR-pipeline szándékosan
eldobja az eltérő alakú ütéseket – vagyis pont azokat, amik itt érdekesek.

Minden esemény `valódi` / `bizonytalan` / `artifact` verdiktet kap mérhető bizonyíték
alapján: 2 s-os epochok jelminősége a saját QRS-amplitúdóhoz normalizálva, a szomszédos
ütések template-egyezése, szüneteknél kötelező alacsonyabb küszöbű újrakeresés a résben,
korai ütéseknél fiziológiai kapcsolási padló és QRS-szélesség-korlátok. A fokozatosan
rövidülő RR-sort légzési sinus arrhythmiaként különíti el az ektópiától.

```bash
python3 -B ecg_audit.py
```

Kimenet sessiononként (`08_28/audit`, `09_01/audit`): `report.md`, `beats.csv`,
`events.csv`, `audit_overview.png`, `audit_morphology.png` és `strips/` EKG-csíkok.

A `hrv_report.py` az audit `beats.csv`-jéből HRV-diagnosztikát számol az NN-soron
(ektópiák és kompenzációs szüneteik nélkül): Poincaré-plot SD1/SD2 ellipszissel,
NN-hisztogram és trianguláris index, Lomb–Scargle spektrum VLF/LF/HF bontással,
DFA α1/α2, sample entropy, Lorenz-diagram. A globális értékek mellett 5 perces
ablakokra is kiszámolja, mert a klinikai referenciák stacionárius szakaszra
vonatkoznak, a teljes felvétel SDNN-jét pedig a terhelési trend felfújja.

```bash
python3 -B hrv_report.py
```

Kimenet: `hrv_report.md` és `hrv_overview.png` az audit könyvtárában.

Ez kutatási/technikai eszköz, nem orvostechnikai eszköz, és a kimenete nem diagnózis.

## Személyre szabott korrekció tesztelése

A `personalize.py` a `comparison.csv` kimenetekből tanul korrekciót az óra HR-jére,
kizárólag óraoldali, kauzális jellemzőkből (HR, EWMA-eltérés, meredekség, gördülő
szórás, nyugalmi alapvonalhoz mért terhelés). Hét modellt hasonlít össze
(identity, bias, affine, HR-szakaszonkénti, FIR-dekonvolúció, ridge, gradient
boosting), és hét validációs séma alatt futtatja őket:

- **A** prefix-kalibráció: a session első X percén tanul, a maradékon értékel;
- **B** blokkolt keresztvalidáció 5 perces csoportokkal (tanulhatósági plafon);
- **C** napok közötti páronkénti transzfer;
- **C2** leave-one-session-out (legalább három sessiontől ad új információt);
- **D** populációs modell + rövid in-session eltolás;
- **E** HR-sávonkénti bontás;
- **F** tanulási görbe: k darab 5 perces blokkon tanít;
- **G** csak a magas-HR tartomány, ahol a hiba tömege keletkezik;
- **H** a MOX testhelyzet és mozgás mint kontextus: a modellek óraoldaliak maradnak,
  ez a séma azt méri, mennyit adna hozzá a Holter-oldali információ felső korlátként.

A modellek hiperparamétereit a tanítóhalmazon belüli blokkolt CV választja, a
szignifikanciát pedig 5 perces blokk-bootstrap adja, mert az 1 Hz-es minták
erősen autokorreláltak (integrált autokorrelációs idő 7–64 s), így a nyers
mintaszám félrevezetően nagy.

```bash
python3 -B personalize.py \
  --session 08_28=08_28/results_v2/comparison.csv \
  --session 09_01=09_01/results/comparison.csv
```

Argumentumok nélkül ugyanezt a két sessiont futtatja. Új mérés hozzáadása egy
további `--session NÉV=ÚT` kapcsoló. A teljes futás néhány perc, mert minden
séma minden foldjában újrahangolja a modelleket. Az `sklearn` opcionális;
hiánya esetén a gradient boosting kimarad, a többi séma fut.

## Következő felvételek

Az eredeti BDF-et és Gadgetbridge SQLite- vagy ZIP-exportot változatlanul őrizd meg, és minden új mérésnek adj egyedi fájlnevet. Az új felvételt külön output könyvtárba futtasd. A BDF nem hordoz időzónát, ezért az aktuális helyi zónát mindig add meg a `--timezone` argumentummal. Több külön nap és pulzustartomány szükséges ahhoz, hogy egy korrekció általánosíthatóságát meg lehessen ítélni.

Ez kutatási/technikai eszköz, nem orvostechnikai eszköz, és az eredménye nem diagnózis. Tünet vagy ritmuszavar gyanúja esetén fordulj egészségügyi szakemberhez.

A pipeline által kiválasztott bias vagy affine képlet is csak az adott mérési sessionre vonatkozó, ideiglenes (`session-specific, provisional`) jelölt. Általános Band 10 finomhangolásnak csak több külön felvételen végzett külső validáció után tekinthető.

## Adatok

A repó csak a kódot tartalmazza; nyers mérés (BDF EKG, Gadgetbridge-export, FIT/GPX) és generált eredmény nincs benne. A dátumozott mappák (`08_28/`, `09_01/` stb.), az `archive/` és a `results/` helyi, verziókezelésen kívüli könyvtárak. Az adatfüggő tesztek fixture hiányában kihagyódnak.

A `prepare_tour_data.py` a hivatalos Garmin FIT SDK-t a helyi `.vendor/` mappából tölti be:

```bash
python3 -m pip install --target .vendor garmin-fit-sdk
```

## Licenc

MIT – lásd a [LICENSE](LICENSE) fájlt.

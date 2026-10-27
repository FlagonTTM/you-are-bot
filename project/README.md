# Воспроизводимый native CatBoost baseline

Самостоятельный offline-проект для рубежного контроля. После установки зависимостей и распаковки официальных данных одна команда выполняет EDA → grouped 5-fold CV → полный refit → CSV → replay checkpoint → SQLite/JSON tracking. Сеть, LLM, загрузка весов и отправка submission в этом пути отсутствуют. HTTP-сервис **не реализован**; архитектурный эскиз сервиса в основном HTML не является работающим API.

## Запуск

Нужен **CPython 3.13.2**. Проверенная среда: macOS 15 / arm64. Все публичные runtime- и test-зависимости точно зафиксированы в `requirements.lock`; прямые корни — в `requirements.in`.

Обычная подготовка окружения (выполняется пользователем отдельно, не автоматически runner-ом):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r project/requirements.lock
```

Установка не выполнялась при сборке этого пакета: использованы уже установленные версии. Clean-environment installation и доступность wheel-файлов для другого компьютера не проверены. Для полностью offline-установки заранее подготовьте соответствующий этой платформе wheelhouse, затем используйте `pip install --no-index --find-links PATH -r project/requirements.lock`. Runner ничего не устанавливает.

В каталоге данных должны лежать ровно используемые пять официальных файлов:

```text
train.json
ytrain.csv
test.json
ytest.csv
sample_submission.csv
```

Raw data не входят в Git. Дополнительные файлы каталога runner не читает. `ytest.csv` содержит только ключи участников и ID, **не тестовые метки**. Заголовок `is_bot` в нём приводит к отказу до разбора значений. Пять SHA-256 сверяются с историческим официальным `runs/template-v1/report.json`; любые изменения байтов, даже переводов строк, требуют восстановить оригинальные файлы, а не ослабить проверку.

Из корня пакета:

```bash
./project/run.sh --data-dir /path/to/official-data --output-dir /path/to/new-runs --run-id native-reproduction-01
```

Или из самого `project/`:

```bash
./run.sh --data-dir /path/to/official-data
```

`--output-dir` по умолчанию — `project/outputs`. Без `--run-id` создаётся UTC timestamp + случайный suffix. Уже существующий namespace никогда не перезаписывается; для следующего запуска нужен другой ID. Wrapper определяет свой ROOT независимо от рабочего каталога, выставляет **все пять** BLAS/OpenMP-переменных в `4`, `PYTHONDONTWRITEBYTECODE=1` и вызывает один Python entry `run.py`. `-E -P -B` игнорируют ambient PYTHONPATH/current-directory imports, но сохраняют доступ к установленным user-site dependencies; `-I` в проверенной offline-среде скрыл бы установленный CatBoost. Это не sandbox против произвольного локального кода.

Для временных файлов используется `COURSEWORK_TMPDIR`, затем уже заданный `TMPDIR`; если обе переменные отсутствуют — собственный `project/outputs/.scratch`. В Hermes передавайте утверждённый scratch через `TMPDIR` или `COURSEWORK_TMPDIR`. Абсолютная scratch-локация в экспортированные метаданные не записывается.

## Замороженный научный контракт

- 1516 TRAIN / 379 TEST диалогов; 3032 / 758 participant rows.
- `configs/membership-v2.json` — неизменённое соответствие ID/order/dialog/component/fold: **2416 development** rows в folds `0..4`, **616 confirmation** rows с `fold=-1`. SHA-256: `e553e7bcdfa98b398b0a534a5d9fc095c5a19f60054f87c365cbe705d6b04dee`.
- Для каждого CV-fit используются только остальные development folds. Оба участника диалога, точные дубликаты содержимого и явные сериализованные ссылки `dialog_id=UUID(...)` объединены в connected components. Компонента целиком исключена из соответствующего fit. ID и group не являются модельными признаками.
- Confirmation не служит новым tuning/selection-контуром. Отдельного confirmation-fit нет. После пяти CV-fit полный финальный refit обучается на всех **3032 TRAIN labels**, включая бывшие 616 confirmation rows: это допустимый inference-fit замороженного рецепта, а не независимая оценка.
- Полный сильный template bank: **162 numeric + 14 categorical** features. Вход evaluation/inference явно очищен от `y`; supervised CatBoost statistics обучаются только внутри training части. Из текстовых categorical features маскируются UUID values.
- Параметры: `iterations=1000`, `depth=5`, `learning_rate=0.04`, `l2_leaf_reg=5`, `loss_function=Logloss`, `random_seed=1729`, `thread_count=4`, `verbose=False`, `allow_writing_files=False`. CPU по умолчанию CatBoost, без torch/E5/GPU. Нет early stopping, eval_set, HPO, model selection или изменения рецепта по метрике.
- Ожидаемый **исторический** OOF LogLoss: **0.16919850572003053**, только 2416 development rows. Каждый новый `report.json` содержит собственную фактически посчитанную метрику и разницу с исторической; reference не выдаётся за live результат. Сравнение `1e-12` диагностическое, не tuning gate.
- Исторический официальный **0.1303426569** относится к другому **LLM-ансамблю**, а **не** к CSV этого native baseline. Свежая LLM-reproduction не заявляется; интерфейс доступа к внешней языковой модели здесь не требуется. CSV **не отправляется** организатору.

`src/botbench.py` и `src/experiment.py` — полные byte-for-byte копии текущих аудированных модулей исходного research-проекта. `src/origins.json` хранит исходные относительные имена, размеры и SHA-256. Они могут отличаться от source SHA старого исторического run: это не выдаётся за равенство всего старого source snapshot. Default использует только чистые APIs `load_dataset`, `membership_folds`, `model_input`, `fit_predict`, `predict`, `write_predictions`. Исторические `experiment.run()` и `experiment.provenance()` сохранены как часть полного оригинала, но **никогда не вызываются default runner-ом**; их старый ROOT-ориентированный workflow не является standalone entry. Внешние project-code imports не нужны.

## Что появляется в `OUTPUT/RUN_ID/`

| Путь | Содержимое |
|---|---|
| `eda.json` | Вычисленные агрегаты: raw message schema, class/pair counts TRAIN, длины/гистограммы TRAIN vs TEST, same-speaker adjacency, whitespace/missingness, точные дубликаты, UUID/reference counts, components. Без сырых сообщений, ID и TEST labels. |
| `oof.csv` | `ID,is_bot,fold`, ровно 2416 development rows в frozen membership order. |
| `fold-receipts.json` | Состав fit/validation по ID и components, per-fold metrics, нулевое пересечение components; confirmation исключён из CV. Это локальный audit-artifact с ID, не публичный EDA. |
| `models/fold-0.cbm` … `fold-4.cbm` | Пять настоящих native CatBoost checkpoints. |
| `models/full.cbm` | Полный 3032-row refit, использованный для TEST. |
| `native-test.csv` | `ID,is_bot`, ровно **758** конечных вероятностей, точные ID/order/schema `sample_submission.csv`, finite `[0,1]`. |
| `checkpoint-replay.json` | Повторный inference из каждого собственного сохранённого `.cbm`; `numpy.array_equal` float64, без ослабления tolerance. Full model сравнивается с повторно прочитанным конечным CSV. |
| `report.json` | Собственные OOF/per-fold метрики, counts, время, replay и честные границы результата. |
| `tracking.sqlite3`, `tracking.json` | Реальный SQL tracker + JSON projection, не MLflow. |
| `index.html` | Статическая offline tracking-страница с параметрами, результатом и ссылками на артефакты. Окончательный статус и полный финальный manifest всегда в `tracking.json`/SQLite. |
| `run.log.jsonl` | Реальные UTC-события этапов. Ошибки записываются как type + stage, без приватного traceback/raw text. |
| `inputs-manifest.json`, `immutability.json` | Input hashes, scientific source/config/dependency hashes до/после; проверка неизменности. |
| `snapshot/` | Копия consumed runtime/config/lock closure, без raw data. |

## Трекинг всех параметров

SQLite schema: `runs`, `params`, `metrics`, `artifacts`. В каждой таблице есть `run_id`; один DB-файл принадлежит одному неизменяемому run namespace. `runs` содержит UTC start/end и **RUNNING → FINISHED / FAILED**. В `params` записаны полный fixed config, explicit scientific protocol, `get_params()` **и `get_all_params()` каждого из шести реально обученных CatBoost models**, categorical indices, feature counts, seed, Python/platform, версии **всей locked dependency closure**, PEP 508 dependency edges, thread policy, source/config/membership/data hashes и доступный read-only Git HEAD/dirty summary. Remotes/credentials/весь environment не читаются.

`metrics` хранит LogLoss/AUC по каждому fold и OOF. `artifacts` хранит run-relative path, SHA-256 и размер каждого payload artifact. SQLite и `tracking.json` — control envelopes; они исключены из собственной hash-таблицы, иначе возникла бы рекурсивная self-hash зависимость. Финальный JSON экспортируется из SQL readback. Частичные артефакты неуспешного run сохраняются и индексируются, когда безопасно прочитать их. Rollback незавершённой транзакции не удаляет заранее committed RUNNING receipt и конечный FAILED status. Повторное открытие того же run ID запрещено; автоматического resume нет. При kill -9/аварии ОС может честно остаться **RUNNING**, это не FINISHED.

Запросы по конкретному ID (локальный `sqlite3` CLI или Python stdlib):

```bash
sqlite3 /path/to/new-runs/native-reproduction-01/tracking.sqlite3 \
  "SELECT run_id,status,started_utc,ended_utc FROM runs WHERE run_id='native-reproduction-01';"
sqlite3 /path/to/new-runs/native-reproduction-01/tracking.sqlite3 \
  "SELECT name,value FROM metrics WHERE run_id='native-reproduction-01' ORDER BY name;"
sqlite3 /path/to/new-runs/native-reproduction-01/tracking.sqlite3 \
  "SELECT name,value_json FROM params WHERE run_id='native-reproduction-01' AND name='model/full';"
sqlite3 /path/to/new-runs/native-reproduction-01/tracking.sqlite3 \
  "SELECT path,sha256,bytes FROM artifacts WHERE run_id='native-reproduction-01' ORDER BY path;"
```

Scientific closure включает только `src/*.py`, `src/origins.json`, `run.py`, `run.sh`, оба consumed configs и оба dependency manifests; input closure — только пять названных файлов. `report.html`, README/docs, `tools/`, `evidence/`, history и outputs намеренно не являются scientific inputs: изменение HTML-отчёта не инвалидирует неизменённый fit. Tests discovery отдельно фиксируется при handoff. Artifact finder никогда не выходит за owned run directory и отвергает symlinks.

## Unit-проверки

Из корня пакета, с выделенным writable scratch:

```bash
SCRATCH=/path/to/approved-scratch
TMPDIR="$SCRATCH" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python3 -m pytest project/tests -q -p no:cacheprovider --basetemp="$SCRATCH/coursework-unit-unique"
```

Тесты используют **только искусственные tiny fixtures**, в отдельных native-fit tests — 2/3 iterations, а не официальный fit. Проверяются default recipe/full bank/source origins, fit-only labels, exclusion групп и confirmation, membership ID/order/dialog binding, test-target rejection, data/source immutability, отсутствие влияния mutable presentation, строгие CSV IDs/order/probabilities, checkpoint replay, реальный SQL/JSON failure/rollback/status/no-overwrite/path confinement и полная recursive dependency closure. Эти tiny результаты не являются competition metric/evidence. При реализации наблюдались RED → GREEN вертикальные срезы; финальные проверки выполняет также parent на зафиксированном standalone copy.

## Область воспроизводимости и ограничения lock

`requirements.lock` получен **не из global pip freeze**, а рекурсивным обходом installed `importlib.metadata` от семи публичных прямых корней с `packaging.Requirement` и оценкой PEP 508 markers (`extra=''`). Включены runtime + pytest transitives, в том числе matplotlib/pandas/plotly, требуемые CatBoost metadata; чужие/private packages не включены. Каждое имя имеет exact pin. При запуске closure и версии сверяются заново. На другой ОС/Python conditional dependencies и бинарные библиотеки могут отличаться; этот lock не является универсальным cross-platform или wheel-hash lock. Wheel SHA не были доступны без сети и **не выдуманы**. Строгое равенство результатов за пределами проверенной Python/platform/dependency среды не обещается. Число threads не является измерением жёсткого CPU quota; фактическое host resource monitoring — отдельная эксплуатационная проверка.

Self-contained означает отсутствие кода/весов/LLM-аккаунта из соседнего research-дерева. Официальные данные и предварительно установленные зависимости остаются явными внешними предпосылками.

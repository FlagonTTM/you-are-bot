# Полный CatBoost + LLM: один вход

Из корня `coursework/`, после размещения **пяти неизменённых официальных файлов** в `DATA_DIR`:

```bash
./project/run_ensemble.sh --data-dir "$DATA_DIR" --llm-mode cached --output-dir ./project/outputs --run-id ensemble-cached-v1
```

Новый внешний inference вместо исторических LLM-ответов:

```bash
./project/run_ensemble.sh --data-dir "$DATA_DIR" --llm-mode fresh --output-dir ./project/outputs --run-id ensemble-fresh-v1
```

`--llm-mode` обязателен. Имена запусков должны быть новыми: существующий каталог никогда не перезаписывается. `DATA_DIR` содержит `train.json`, `ytrain.csv`, `test.json`, `ytest.csv`, `sample_submission.csv`; SHA-256, схема, порядок ID, мощности и frozen membership проверяются до обучения. `ytest.csv` — только ID, не тестовые метки. Acquisition, установка окружения, отправка на соревнование и перестроение общего презентационного `project/report.html` **не входят** в эту команду. Локальный отчёт и индекс текущего запуска создаются автоматически.

## Что действительно выполняется

1. В обоих режимах запускается существующий `src/pipeline.py`: EDA, пять component-exclusive CV fits и отдельный full refit на всех 3032 TRAIN-участниках. Каждый CatBoost имеет 1000 деревьев, seed 1729, банк 162 numeric + 14 categorical, четыре вычислительных потока; HPO/ранней остановки/смены модели нет. Confirmation не участвует в CV, но входит в заранее замороженный полный refit.
2. Из RAW TRAIN строятся прежние char-wb TF-IDF references и точные first-message label counts. Все TRAIN-метки разрешены как labels других тренировочных примеров; TEST подаётся без labels. Для каждого из 379 TEST-диалогов используются новые full-refit priors. Чанки: отсортированные dialog IDs по 40, но исходный порядок строк внутри чанка, как в историческом `campaign.batches`; всего десять чанков. Текст, усечение и prompt не переизобретены.
3. **Cached:** native-модель обучается заново, это не replay CatBoost-файла. Сначала все новые native-вероятности сравниваются с историческим компонентом точно, без tolerance. Затем заново подготовленные `input.json` и `prompt.txt` каждого чанка сверяются с десятью историческими digest pairs. Только после этого разрешается использовать настоящие сохранённые LLM p0/p1 из `ensemble/components.csv`. SHA самого cache и manifest также фиксированы. Любое отличие — `FAILED`, не fallback на fresh, не ослабление допуска. Итоговый UTF-8/CRLF CSV обязан иметь SHA-256 `4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639`.
4. **Fresh:** до native fits проверяются фиксированная версия интерфейса запуска и авторизация доступа к внешней предобученной языковой модели. Точные требования сохранены в `llm/config.json` и `llm/adapter.py`. Существующий авторизованный аккаунт используется самим интерфейсом: pipeline не читает credential files, cookies или secrets. Исполняется прежний pure-inference вызов в отдельной read-only сессии с отключёнными tools/plugins/hooks/browser. Перед dispatch проверяются raw inputs, полный source/config/cache closure, exact prompt/input/schema/prepared bytes и бинарник CLI; проверенные prompt bytes передаются из памяти и повторно проверяются после возврата. Timeout 601 s на batch, максимум десять вызовов. Любой item event, кроме `reasoning`/`agent_message`, отклоняется. При timeout/неоднозначном результате сохраняются попытка и `FAILED`; существующие output/events **никогда автоматически не отправляются повторно**. Никакого скачивания внешних данных/весов, новых покупок или contest uploads нет. Этот режим передаёт подготовленные тексты существующему внешнему провайдеру и требует отдельного разрешения владельца на реальный запуск.
5. В обоих режимах сборка — ровно `0.5 * native + 0.5 * llm`, без clipping, округления, complement p1 или подбора весов. CSV соответствует официальному sample order. В fresh совпадение с историческим SHA — только диагностический факт, **не** условие успеха и не обещание идентичных ответов.

## Окружение и происхождение

Только существующий Python environment: CPython **3.13.2**, macOS arm64 и 25-пакетный installed closure из неизменённых `requirements.in` / `requirements.lock`. Новых Python dependencies нет; wrapper сам выставляет пять BLAS/OpenMP переменных в `4`, запрещает bytecode и изолирует import paths. Scratch: `COURSEWORK_TMPDIR`, иначе установленный `TMPDIR`, иначе `project/outputs/.scratch`; зависимости не устанавливаются автоматически. Интерфейс доступа к внешней LLM — отдельная от Python lock предпосылка, только для fresh. Модель и доступность аккаунта не объявляются проверенными по unit mocks.

`llm/bridge.py` — полная byte-identical копия audited guard-corrected bridge, SHA `63821f37719cae16c335f20a39862a8e53cc5a662948701f122d326acc925f6d`. `llm/historical_bridge.py` сохраняет полную generation source копию scored final-v1, SHA `dc95650cc3cf3081e87c2849a03d3f332c6b43e0fbdca5e95167d57253b118d6`. До работы проверяются оба SHA и AST parity `MODEL`, `PROMPT`, `first_text`, `prepare_batch`, `validate_output`; old screen/main entrypoints не вызываются. Adapter сохраняет audited inference argv, подставляет проверенный абсолютный CLI executable, отдельный пустой cwd и timeout 601. Сырые dialogue instructions рассматриваются как quoted untrusted data, а не команды.

Cache portable: `components.csv` SHA `ed36be23cfb492ac2dfdd77298bf998b64c900703db3b06f97af2a4e4bb6ab6e`, manifest SHA `382bc9a1381599993ee6784f287f09f8f99a8afab2a99dc70a8f7c40df6cc6e8`. Manifest несёт исторические input/prompt/output digests и generation-source identity; полная старая receipt chain проверена при создании cache. Для запуска не нужны файлы исходного research-каталога вне пакета.

## Артефакты и границы выводов

В `OUTPUT_DIR/RUN_ID/`: `submission.csv`, `llm-test.csv`, `report.json`, `submission-validation.json`, `cache-binding.json` (cached), `llm-payload-locks.json`, `inputs-manifest.json`, `immutability.json`, snapshot реально потреблённых source/config/lock/cache, локальные LLM payloads/receipts/events, `index.html`, `tracking.sqlite3`, `tracking.json`. Вложенный `native/catboost/` содержит реальные шесть native checkpoints, OOF, native TEST CSV, EDA, fold metrics/receipts, defaults и собственный FINISHED SQL/JSON receipt. Родитель перечитывает SQL/JSON и хеши этих артефактов, наследует все параметры и явно называет метрики `catboost_oof/*` / `catboost_fold-*/*`, а не ensemble CV.

Это настоящий stdlib SQLite + JSON tracker, **не MLflow** и не сервер. `index.html` содержит режим, offline параметры/метрики/ссылки; авторитетный terminal status — SQL/JSON. `FINISHED` устанавливается лишь после финальных readback/hash checks, исключения оставляют `FAILED`. Mutable общий HTML, документы и evidence не входят в scientific closure.

Исторический public LogLoss **0.1303426569** относится к submission **10588** и не пересчитывается локально. Полная ensemble CV здесь не измеряется: новые CatBoost OOF нельзя выдавать за OOF CatBoost+LLM. Fresh не получает автоматически официального score, гарантии прежнего score или нового submission.

Сырые payloads/events остаются только в локальных ignored `project/outputs/` либо во внешнем приватном output directory. Запись их в другие каталоги внутри coursework отклоняется. Не добавляйте raw текст/events в Git, presentation evidence или distributable ZIP; экспорт доказательств — отдельный curated allowlist. Минимальный `project/ensemble/run.py` остаётся отдельным standard-library replay **обоих** сохранённых компонентов: он не обучает и не классифицирует заново.

## Локальные тесты без официального fit и model calls

```bash
TMPDIR="$COURSEWORK_TMPDIR" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python3 -B -m pytest -p no:cacheprovider project/ensemble_tests project/tests -q
bash -n project/run_ensemble.sh
shellcheck project/run_ensemble.sh
```

Для тестов задайте `COURSEWORK_TMPDIR` как существующий разрешённый scratch directory. Новая suite использует tiny artificial dialogues, явно synthetic native receipts и mock CLI subprocess; её числа **не являются** competition metrics. Реальный cached fit и десять fresh batches запускает владелец после заморозки исходников, каждый в своём новом namespace.

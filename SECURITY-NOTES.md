# Локальные проверки безопасности

Пакет — учебный offline-прототип, **не production release**. Наличие тестов и сканеров не доказывает эксплуатационную безопасность.

## Что фактически выполнено

- Основной runtime `project/src/` + `project/run.py`: Bandit с порогом medium/high — **0 findings**, ошибок разбора нет.
- Gitleaks для основного runtime, архива и полного пакета: срабатывания на публичных контрольных суммах разобраны; исключения ограничены точными строками с этими digest.
- Shared baseline `/Users/Ivan/PycharmProjects/security/check.sh`: Git whitespace/conflict check, candidate Gitleaks и ShellCheck прошли. Общий скрипт завершился **exit 1** из-за Bandit warnings; это **не** полный зелёный gate.
- Online Semgrep rules и Trivy advisory lookup не запускались. Актуальные CVE установленных зависимостей этим пакетом не проверены.

## Разобранные предупреждения

1. **Gitleaks generic-api-key в архивном `research/encoder_assets.py:18` и `evidence/history/encoder-pilot-report.json:88–90`.** Значения — SHA-256 публичных файлов `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, не credentials. Контрольные суммы сверены с локальными разрешёнными model assets. `.gitleaks.toml` разрешает только полные строки с конкретными путями/именами и этими digest; архивные исходники и отчёты не менялись. Произвольные токены и остальные строки не исключены.
2. **Bandit B615, четыре medium warnings в архивном `research/encoder_probe.py:204,243,247,250`.** Эти вызовы читают локальные директории с `local_files_only=True`, `trust_remote_code=False`; веса — safetensors. Загрузка pretrained assets предварительно проверяет фиксированный revision и publisher hashes. Это не объявляется OS network sandbox: программный `no_network()` сам по себе таким барьером не является. Архив не вызывается default runner-ом; сами веса в Git отсутствуют. Предупреждения сохранены, а не подавлены ради зелёного отчёта.
3. **Low warnings:** `assert` в старых helper-функциях/контентных self-checks и импорт/вызовы `subprocess`. Default pipeline отдельно проверяет данные и splits через явные exceptions. Его subprocess-вызовы — фиксированные read-only `git rev-parse` / `git status`, без shell и без полей диалогов в аргументах. Историческая LLM-ветка не входит в default offline run. Это triage текущего ограниченного сценария, не blanket waiver на произвольный запуск архивов.

## Границы

Не включены raw competition data, pretrained-веса, credentials, `.env`, сторонние session logs. CSV в evidence — собственные вероятности модели, не скрытые test labels. Полный локальный run содержит собственные `.cbm` checkpoints; curated Git evidence намеренно их не включает. Новых пакетов и весов при сборке не устанавливалось. Baseline не обращался к внешним API/LLM и не отправлял CSV на конкурс. Создание и push приватного GitHub-репозитория — отдельно разрешённая пользователем операция, не часть pipeline.

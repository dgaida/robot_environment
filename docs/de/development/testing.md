# Testen

Das Framework verwendet `pytest` für automatisierte Tests.

## Tests ausführen
Um alle Tests auszuführen, verwenden Sie:
```bash
python3 -m pytest
```

## Coverage-Bericht
Der Coverage-Bericht wird automatisch generiert:
```bash
python3 -m pytest --cov=robot_environment
```

## Integrationstests
Integrationstests erfordern oft eine aktive Redis-Instanz.
```bash
docker run -p 6379:6379 redis:alpine
python3 -m pytest tests/integration
```

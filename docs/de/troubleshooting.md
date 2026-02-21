# Fehlerbehebung (Troubleshooting)

## Häufige Probleme

### ModuleNotFoundError: No module named 'text2speech.engines'
Dies ist ein bekanntes Problem bei der Installation des `text2speech`-Pakets, bei dem Unterpakete wie `engines` nicht korrekt mitinstalliert werden.

**Lösung:**
Installieren Sie das `text2speech`-Paket im **Editable-Modus** aus seinem Quellcode-Verzeichnis:
```bash
cd /pfad/zu/text2speech/repository
pip install -e .
```

Als Maintainer des `text2speech`-Repositories sollten Sie die `pyproject.toml` anpassen, um alle Unterpakete einzuschließen:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["text2speech*"]
```

### Keine Objekte erkannt
- Überprüfen Sie die Redis-Verbindung.
- Stellen Sie sicher, dass der Visions-Dienst (`vision_detect_segment`) läuft.
- Überprüfen Sie die Beleuchtung des Arbeitsbereichs.

### Roboter bewegt sich nicht
- Überprüfen Sie die IP-Adresse des Roboters in der Konfiguration.
- Stellen Sie sicher, dass der Roboter kalibriert ist (besonders bei Niryo).
- Prüfen Sie, ob die Zielkoordinaten innerhalb des erreichbaren Bereichs liegen.

### Ungenaue Positionierung
- Überprüfen Sie die Workspace-Kalibrierung.
- Stellen Sie sicher, dass die Kamera fest montiert ist.
- Verifizieren Sie die Koordinatentransformationen in `robot_workspace`.

## Debug-Modus
Aktivieren Sie das ausführliche Logging beim Initialisieren der Umgebung:
```python
env = Environment(..., verbose=True)
```

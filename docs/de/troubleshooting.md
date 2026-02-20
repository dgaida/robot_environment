# Fehlerbehebung (Troubleshooting)

## Häufige Probleme

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

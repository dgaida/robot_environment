# Docstring-Styleguide

Wir verwenden den **Google Python Style Guide** für alle Docstrings in diesem Projekt.

## Format

```python
def funktion(parameter: int) -> bool:
    """
    Kurze Zusammenfassung der Funktion.

    Detaillierte Beschreibung der Funktion und ihrer Logik,
    falls erforderlich.

    Args:
        parameter: Beschreibung des Parameters.

    Returns:
        bool: Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wenn der Parameter ungültig ist.
    """
```

## Anforderungen
- Alle öffentlichen Klassen, Methoden und Funktionen müssen dokumentiert werden.
- Modul-Docstrings am Anfang jeder Datei sind erforderlich.
- Wir erzwingen eine API-Dokumentationsabdeckung von mindestens 95% mittels `interrogate`.

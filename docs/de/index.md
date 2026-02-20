# Roboter-Umgebung (Robot Environment)

**Ein umfassendes Python-Framework für robotergestützte Pick-and-Place-Operationen mit visionsbasierter Objekterkennung und Manipulationsfähigkeiten.**

Die Roboter-Umgebung ist ein modulares Python-Framework, das die Steuerung von Roboterarmen für präzise **Pick-and-Place-Aufgaben** automatisiert. Durch die Integration von **KI-basierter Objekterkennung** und fortschrittlichem Workspace-Management können Roboter wie der Niryo Ned2 oder WidowX Objekte unabhängig identifizieren und manipulieren.

## Hauptmerkmale

- 🤖 **Multi-Roboter-Unterstützung** - Modulare Architektur zur Unterstützung von Niryo Ned2 und WidowX Roboterarmen.
- 👁️ **Visionsbasierte Objekterkennung** - Integration verschiedener Detektionsmodelle über [vision_detect_segment](https://github.com/dgaida/vision_detect_segment).
- 🗺️ **Workspace-Management** - Flexible Workspace-Definition mit Koordinatentransformation von Kamera zu Welt über [robot_workspace](https://github.com/dgaida/robot_workspace).
- 📡 **Redis-Kommunikation** - Effizientes Image-Streaming und Objektdatenaustausch via Redis über [redis_robot_comm](https://github.com/dgaida/redis_robot_comm).
- 🔊 **Text-to-Speech** - Natürliches Sprachfeedback mit [text2speech](https://github.com/dgaida/text2speech).
- 🧵 **Thread-sichere Operationen** - Gleichzeitige Kamera-Updates und Robotersteuerung mit korrektem Locking.
- 🎮 **Simulationsunterstützung** - Kompatibel mit realen Robotern und der Gazebo-Simulation.
- 💾 **Objektspeicher-Management** - Intelligente Verfolgung erkannter Objekte mit Workspace-bezogenen Updates.

## Schnellstart

```python
from robot_environment.environment import Environment
import time

# Umgebung für Niryo-Roboter initialisieren
env = Environment(
    el_api_key="ihr_elevenlabs_key",
    use_simulation=False,
    robot_id="niryo",
    verbose=True
)

# Zur Beobachtungspose bewegen
env.robot_move2home_observation_pose()
time.sleep(2)

# Objekte erkennen
robot = env.robot()
success = robot.pick_place_object(
    object_name="pencil",
    pick_coordinate=[-0.1, 0.01],
    place_coordinate=[0.1, 0.11]
)

if success:
    print("Aufgabe abgeschlossen!")

env.cleanup()
```

## Installation

```bash
pip install -e .
```

Stellen Sie sicher, dass ein Redis-Server läuft:
```bash
docker run -p 6379:6379 redis:alpine
```

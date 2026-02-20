# Architektur

Dieses Dokument beschreibt die Systemarchitektur der Roboter-Umgebung.

## Systemübersicht

Die Roboter-Umgebung ist in mehreren Schichten aufgebaut, um eine klare Trennung der Zuständigkeiten zu gewährleisten.

```mermaid
graph TD
    Env[Environment Layer] --> Robot[Robot Control Layer]
    Env --> Vision[Vision Layer]
    Env --> Workspace[Workspace Layer]

    Robot --> RC[RobotController Abstract]
    RC --> NiryoRC[NiryoRobotController]
    RC --> WidowXRC[WidowXRobotController]

    Vision --> FG[FrameGrabber Abstract]
    FG --> NiryoFG[NiryoFrameGrabber]
    FG --> WidowXFG[WidowXFrameGrabber]

    Workspace --> WS[Workspace Abstract]
    WS --> NiryoWS[NiryoWorkspace]
```

## Komponenten

### Environment Layer
Der `Environment`-Zentralorchestrator koordiniert alle Subsysteme. Er verwaltet den Objektspeicher (`ObjectMemoryManager`) und stellt sicher, dass Kamera-Updates und Roboterbefehle thread-sicher ausgeführt werden.

### Robot Control Layer
Stellt eine High-Level-API für Pick-and-Place-Operationen bereit. Die abstrakte Klasse `RobotController` ermöglicht die Unterstützung verschiedener Hardware-Backends.

### Vision Layer
Verantwortlich für die Erfassung von Bildern und deren Streaming über Redis. Er integriert sich mit dem `vision_detect_segment`-Paket für die KI-gestützte Objekterkennung.

### Workspace Layer
Verwaltet die räumlichen Grenzen und bietet Transformationen zwischen Kamerakoordinaten (Pixel) und Weltkoordinaten (Meter).

## Datenfluss

Der folgende Diagramm zeigt den Datenfluss während eines typischen Update-Zyklus:

```mermaid
sequenceDiagram
    participant E as Environment
    participant F as FrameGrabber
    participant R as Redis
    participant V as Vision Service
    participant M as Memory Manager

    loop Kamera-Update
        E->>F: get_current_frame()
        F->>R: Publish Image
        R->>V: Process Image
        V->>R: Publish Detections
        R->>E: get_detected_objects()
        E->>M: update(detections)
    end
```

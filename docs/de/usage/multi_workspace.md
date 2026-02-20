# Multi-Workspace Guide

Die Roboter-Umgebung unterstützt die Verwaltung mehrerer Arbeitsbereiche gleichzeitig.

## Beispiel: Transfer zwischen Workspaces

```python
left_ws_id = env.workspaces().get_workspace_left_id()
right_ws_id = env.workspaces().get_workspace_right_id()

# Von links aufnehmen und rechts platzieren
robot.pick_place_object_across_workspaces(
    object_name="cube",
    pick_workspace_id=left_ws_id,
    pick_coordinate=[0.2, 0.05],
    place_workspace_id=right_ws_id,
    place_coordinate=[0.25, -0.05]
)
```

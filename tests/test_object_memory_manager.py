"""
Unit tests for ObjectMemoryManager class

Tests cover:
- Initialization and workspace management
- Object memory updates with visibility tracking
- Manual updates (pick/place operations)
- Thread safety
- Memory clearing and statistics
- Multi-workspace operations
"""

import pytest
import time
import threading
from unittest.mock import Mock
from robot_environment.object_memory_manager import ObjectMemoryManager
from robot_workspace import Objects, Object, PoseObjectPNP


@pytest.fixture
def memory_manager():
    """Create a fresh ObjectMemoryManager"""
    return ObjectMemoryManager(manual_update_timeout=5.0, position_tolerance=0.05, verbose=False)


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock()
    workspace.id.return_value = "test_ws"
    workspace.img_shape.return_value = (640, 480, 3)
    return workspace


def create_mock_object(label, x, y, width=0.05, height=0.05):
    """Helper to create properly mocked Object"""
    obj = Mock(spec=Object)
    obj.label.return_value = label
    obj.x_com.return_value = x
    obj.y_com.return_value = y
    obj.xy_com.return_value = (x, y)
    obj.width_m.return_value = width
    obj.height_m.return_value = height
    obj.coordinate.return_value = [x, y]
    obj._x_com = x
    obj._y_com = y
    return obj


class TestObjectMemoryManagerInitialization:
    """Test initialization and basic setup"""

    def test_initialization_default_params(self):
        """Test initialization with default parameters"""
        manager = ObjectMemoryManager()

        assert manager._manual_update_timeout == 5.0
        assert manager._position_tolerance == 0.05
        assert manager._verbose is False
        assert len(manager._memories) == 0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters"""
        manager = ObjectMemoryManager(manual_update_timeout=10.0, position_tolerance=0.1, verbose=True)

        assert manager._manual_update_timeout == 10.0
        assert manager._position_tolerance == 0.1
        assert manager._verbose is True

    def test_initialization_creates_lock(self, memory_manager):
        """Test that thread lock is created"""
        assert memory_manager._lock is not None
        assert hasattr(memory_manager._lock, "acquire")
        assert hasattr(memory_manager._lock, "release")

    def test_initialize_workspace(self, memory_manager):
        """Test initializing a new workspace"""
        memory_manager.initialize_workspace("ws1")

        assert "ws1" in memory_manager._memories
        assert isinstance(memory_manager._memories["ws1"], Objects)
        assert len(memory_manager._memories["ws1"]) == 0

    def test_initialize_workspace_twice(self, memory_manager):
        """Test initializing same workspace twice doesn't reset"""
        memory_manager.initialize_workspace("ws1")

        # Add mock object
        mock_obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(mock_obj)

        # Initialize again
        memory_manager.initialize_workspace("ws1")

        # Should not be reset (only initializes if not exists)
        assert len(memory_manager._memories["ws1"]) == 1


class TestObjectMemoryManagerBasicOperations:
    """Test basic memory operations"""

    def test_get_empty_workspace(self, memory_manager):
        """Test getting objects from empty workspace"""
        memory_manager.initialize_workspace("ws1")

        objects = memory_manager.get("ws1")

        assert isinstance(objects, Objects)
        assert len(objects) == 0

    def test_get_uninitialized_workspace(self, memory_manager):
        """Test getting objects from uninitialized workspace"""
        objects = memory_manager.get("nonexistent")

        assert isinstance(objects, Objects)
        assert len(objects) == 0

    def test_get_returns_copy(self, memory_manager):
        """Test that get() returns a copy, not reference"""
        memory_manager.initialize_workspace("ws1")

        # Add object to internal memory
        mock_obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(mock_obj)

        # Get copy
        copy1 = memory_manager.get("ws1")
        copy2 = memory_manager.get("ws1")

        # Modify copy
        copy1.clear()

        # Original and second copy should be unchanged
        assert len(memory_manager._memories["ws1"]) == 1
        assert len(copy2) == 1

    def test_clear_specific_workspace(self, memory_manager):
        """Test clearing specific workspace"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        # Add objects
        memory_manager._memories["ws1"].append(create_mock_object("pencil", 0.25, 0.05))
        memory_manager._memories["ws2"].append(create_mock_object("pen", 0.30, 0.10))

        # Clear ws1
        memory_manager.clear("ws1")

        assert len(memory_manager._memories["ws1"]) == 0
        assert len(memory_manager._memories["ws2"]) == 1

    def test_clear_all_workspaces(self, memory_manager):
        """Test clearing all workspaces"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        memory_manager._memories["ws1"].append(create_mock_object("pencil", 0.25, 0.05))
        memory_manager._memories["ws2"].append(create_mock_object("pen", 0.30, 0.10))

        # Clear all
        memory_manager.clear()

        assert len(memory_manager._memories["ws1"]) == 0
        assert len(memory_manager._memories["ws2"]) == 0

    def test_get_all_workspaces(self, memory_manager):
        """Test getting objects from all workspaces"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        memory_manager._memories["ws1"].append(create_mock_object("pencil", 0.25, 0.05))
        memory_manager._memories["ws2"].append(create_mock_object("pen", 0.30, 0.10))

        all_objects = memory_manager.get_all()

        assert len(all_objects) == 2
        assert "ws1" in all_objects
        assert "ws2" in all_objects
        assert len(all_objects["ws1"]) == 1
        assert len(all_objects["ws2"]) == 1


class TestObjectMemoryManagerUpdate:
    """Test memory update logic"""

    def test_update_adds_new_objects(self, memory_manager):
        """Test that update adds new detected objects"""
        memory_manager.initialize_workspace("ws1")

        detected = Objects([create_mock_object("pencil", 0.25, 0.05), create_mock_object("pen", 0.30, 0.10)])

        added, updated = memory_manager.update("ws1", detected, at_observation_pose=True, robot_in_motion=False)

        assert added == 2
        assert updated == 0
        assert len(memory_manager._memories["ws1"]) == 2

    def test_update_skips_when_not_at_observation(self, memory_manager):
        """Test update is skipped when not at observation pose"""
        memory_manager.initialize_workspace("ws1")

        detected = Objects([create_mock_object("pencil", 0.25, 0.05)])

        added, updated = memory_manager.update("ws1", detected, at_observation_pose=False, robot_in_motion=False)

        assert added == 0
        assert updated == 0
        assert len(memory_manager._memories["ws1"]) == 0

    def test_update_skips_when_robot_in_motion(self, memory_manager):
        """Test update is skipped when robot is moving"""
        memory_manager.initialize_workspace("ws1")

        detected = Objects([create_mock_object("pencil", 0.25, 0.05)])

        added, updated = memory_manager.update("ws1", detected, at_observation_pose=True, robot_in_motion=True)

        assert added == 0
        assert updated == 0

    def test_update_ignores_duplicates(self, memory_manager):
        """Test that duplicate objects are not added"""
        memory_manager.initialize_workspace("ws1")

        # First update
        detected1 = Objects([create_mock_object("pencil", 0.25, 0.05)])
        memory_manager.update("ws1", detected1, True, False)

        # Second update with same object
        detected2 = Objects([create_mock_object("pencil", 0.25, 0.05)])
        added, updated = memory_manager.update("ws1", detected2, True, False)

        assert added == 0
        assert len(memory_manager._memories["ws1"]) == 1

    def test_update_clears_memory_when_workspace_regained(self, memory_manager):
        """Test memory is cleared when workspace becomes visible again"""
        memory_manager.initialize_workspace("ws1")

        # Initial update
        detected1 = Objects([create_mock_object("pencil", 0.25, 0.05)])
        memory_manager.update("ws1", detected1, True, False)

        # Lose visibility
        memory_manager.update("ws1", Objects(), False, True)

        # Regain visibility - should clear old data
        detected2 = Objects([create_mock_object("pen", 0.30, 0.10)])
        added, updated = memory_manager.update("ws1", detected2, True, False)

        # Should have cleared and added new
        assert len(memory_manager._memories["ws1"]) == 1
        assert memory_manager._memories["ws1"][0].label() == "pen"


class TestObjectMemoryManagerManualUpdates:
    """Test manual update tracking for pick/place operations"""

    def test_mark_manual_update(self, memory_manager):
        """Test marking an object as manually updated"""
        memory_manager.initialize_workspace("ws1")

        # Add object
        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        # Mark as manually updated
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)
        result = memory_manager.mark_manual_update("ws1", "pencil", [0.25, 0.05], new_pose)

        assert result is True
        assert "pencil" in memory_manager._manual_updates["ws1"]

    def test_mark_manual_update_object_not_found(self, memory_manager):
        """Test marking manual update for non-existent object"""
        memory_manager.initialize_workspace("ws1")

        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)
        result = memory_manager.mark_manual_update("ws1", "nonexistent", [0.25, 0.05], new_pose)

        assert result is False

    def test_manual_updates_expire(self, memory_manager):
        """Test that manual updates expire after timeout"""
        # Use short timeout
        manager = ObjectMemoryManager(manual_update_timeout=0.1)
        manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        manager._memories["ws1"].append(obj)

        # Mark manual update
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)
        manager.mark_manual_update("ws1", "pencil", [0.25, 0.05], new_pose)

        # Wait for expiry
        time.sleep(0.15)

        # Update should trigger cleanup
        manager.update("ws1", Objects(), True, False)

        # Manual update should be expired
        assert "pencil" not in manager._manual_updates["ws1"]

    def test_update_respects_manual_updates(self, memory_manager):
        """Test that updates respect manual position changes"""
        memory_manager.initialize_workspace("ws1")

        # Add object at original position
        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        # Manually move it
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)
        memory_manager.mark_manual_update("ws1", "pencil", [0.25, 0.05], new_pose)

        # Detection at old position should be ignored
        detected = Objects([create_mock_object("pencil", 0.25, 0.05)])
        added, updated = memory_manager.update("ws1", detected, True, False)

        # Should not add duplicate (manual update protected it)
        assert added == 0


class TestObjectMemoryManagerRemoveObject:
    """Test object removal operations"""

    def test_remove_object_success(self, memory_manager):
        """Test successfully removing an object"""
        memory_manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        result = memory_manager.remove_object("ws1", "pencil", [0.25, 0.05])

        assert result is True
        assert len(memory_manager._memories["ws1"]) == 0

    def test_remove_object_not_found(self, memory_manager):
        """Test removing non-existent object"""
        memory_manager.initialize_workspace("ws1")

        result = memory_manager.remove_object("ws1", "pencil", [0.25, 0.05])

        assert result is False

    def test_remove_object_wrong_coordinates(self, memory_manager):
        """Test removing object with wrong coordinates"""
        memory_manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        # Try to remove with far-off coordinates
        result = memory_manager.remove_object("ws1", "pencil", [0.50, 0.50])

        assert result is False
        assert len(memory_manager._memories["ws1"]) == 1

    def test_remove_object_clears_manual_update(self, memory_manager):
        """Test that removing object clears manual update tracking"""
        memory_manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        # Mark manual update
        memory_manager._manual_updates["ws1"] = {"pencil": time.time()}

        # Remove object
        memory_manager.remove_object("ws1", "pencil", [0.25, 0.05])

        assert "pencil" not in memory_manager._manual_updates["ws1"]


class TestObjectMemoryManagerMoveObject:
    """Test moving objects between workspaces"""

    def test_move_object_success(self, memory_manager):
        """Test successfully moving object between workspaces"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        result = memory_manager.move_object("ws1", "ws2", "pencil", [0.25, 0.05], [0.30, 0.10])

        assert result is True
        assert len(memory_manager._memories["ws1"]) == 0
        assert len(memory_manager._memories["ws2"]) == 1
        assert memory_manager._memories["ws2"][0].label() == "pencil"

    def test_move_object_creates_target_workspace(self, memory_manager):
        """Test moving to uninitialized workspace initializes it"""
        memory_manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        result = memory_manager.move_object("ws1", "ws2", "pencil", [0.25, 0.05], [0.30, 0.10])

        assert result is True
        assert "ws2" in memory_manager._memories

    def test_move_object_not_found(self, memory_manager):
        """Test moving non-existent object"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        result = memory_manager.move_object("ws1", "ws2", "pencil", [0.25, 0.05], [0.30, 0.10])

        assert result is False

    def test_move_object_updates_coordinates(self, memory_manager):
        """Test that moved object has new coordinates"""
        memory_manager.initialize_workspace("ws1")
        memory_manager.initialize_workspace("ws2")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        memory_manager.move_object("ws1", "ws2", "pencil", [0.25, 0.05], [0.30, 0.10])

        moved_obj = memory_manager._memories["ws2"][0]
        assert moved_obj._x_com == 0.30
        assert moved_obj._y_com == 0.10


class TestObjectMemoryManagerStatistics:
    """Test memory statistics and reporting"""

    def test_get_memory_stats_empty(self, memory_manager):
        """Test getting stats for empty memory"""
        memory_manager.initialize_workspace("ws1")

        stats = memory_manager.get_memory_stats()

        assert "ws1" in stats
        assert stats["ws1"]["object_count"] == 0
        assert stats["ws1"]["manual_updates"] == 0

    def test_get_memory_stats_with_objects(self, memory_manager):
        """Test getting stats with objects"""
        memory_manager.initialize_workspace("ws1")

        memory_manager._memories["ws1"].append(create_mock_object("pencil", 0.25, 0.05))
        memory_manager._memories["ws1"].append(create_mock_object("pen", 0.30, 0.10))

        stats = memory_manager.get_memory_stats()

        assert stats["ws1"]["object_count"] == 2
        assert len(stats["ws1"]["objects"]) == 2

    def test_get_memory_stats_includes_manual_updates(self, memory_manager):
        """Test that stats include manual update info"""
        memory_manager.initialize_workspace("ws1")

        obj = create_mock_object("pencil", 0.25, 0.05)
        memory_manager._memories["ws1"].append(obj)

        # Mark manual update
        memory_manager._manual_updates["ws1"] = {"pencil": time.time()}

        stats = memory_manager.get_memory_stats()

        assert stats["ws1"]["manual_updates"] == 1
        assert stats["ws1"]["objects"][0]["manually_updated"] is True


class TestObjectMemoryManagerThreadSafety:
    """Test thread safety of operations"""

    def test_concurrent_updates(self, memory_manager):
        """Test concurrent updates are safe"""
        memory_manager.initialize_workspace("ws1")

        def update_memory():
            detected = Objects([create_mock_object("pencil", 0.25, 0.05)])
            memory_manager.update("ws1", detected, True, False)

        threads = [threading.Thread(target=update_memory) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have added objects (but deduped)
        assert len(memory_manager._memories["ws1"]) >= 1

    def test_concurrent_reads(self, memory_manager):
        """Test concurrent reads are safe"""
        memory_manager.initialize_workspace("ws1")
        memory_manager._memories["ws1"].append(create_mock_object("pencil", 0.25, 0.05))

        results = []

        def read_memory():
            objects = memory_manager.get("ws1")
            results.append(len(objects))

        threads = [threading.Thread(target=read_memory) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert len(results) == 10
        assert all(r == 1 for r in results)

    def test_concurrent_remove_operations(self, memory_manager):
        """Test concurrent remove operations"""
        memory_manager.initialize_workspace("ws1")

        # Add multiple objects
        for i in range(5):
            memory_manager._memories["ws1"].append(create_mock_object(f"obj{i}", 0.2 + i * 0.05, 0.05))

        def remove_object(idx):
            memory_manager.remove_object("ws1", f"obj{idx}", [0.2 + idx * 0.05, 0.05])

        threads = [threading.Thread(target=remove_object, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All objects should be removed
        assert len(memory_manager._memories["ws1"]) == 0


class TestObjectMemoryManagerEdgeCases:
    """Test edge cases and error conditions"""

    def test_update_uninitialized_workspace_initializes_it(self, memory_manager):
        """Test updating uninitialized workspace initializes it"""
        detected = Objects([create_mock_object("pencil", 0.25, 0.05)])

        memory_manager.update("ws1", detected, True, False)

        assert "ws1" in memory_manager._memories

    def test_position_tolerance_matching(self, memory_manager):
        """Test position tolerance for duplicate detection"""
        memory_manager.initialize_workspace("ws1")

        # Add object
        detected1 = Objects([create_mock_object("pencil", 0.250, 0.050)])
        memory_manager.update("ws1", detected1, True, False)

        # Try to add same object with slightly different coordinates (within tolerance)
        detected2 = Objects([create_mock_object("pencil", 0.252, 0.051)])
        added, _ = memory_manager.update("ws1", detected2, True, False)

        # Should not add duplicate (within 0.05m tolerance)
        assert added == 0
        assert len(memory_manager._memories["ws1"]) == 1

    def test_clear_nonexistent_workspace(self, memory_manager):
        """Test clearing non-existent workspace doesn't crash"""
        memory_manager.clear("nonexistent")
        # Should not crash
        assert True

    def test_get_stats_empty_manager(self, memory_manager):
        """Test getting stats when no workspaces initialized"""
        stats = memory_manager.get_memory_stats()

        assert isinstance(stats, dict)
        assert len(stats) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

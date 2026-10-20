"""
Vehicle Profile Registry.
Loads and caches vehicle configuration profiles from model/vehicles/*.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .vehicle_schema import VehicleProfile


class VehicleRegistry:
    """
    Registry for managing supported vehicle profiles and platform specifications.
    """

    def __init__(self, vehicles_dir: Optional[str] = None):
        if vehicles_dir is None:
            vehicles_dir = str(Path(__file__).parent / "vehicles")

        self.vehicles_dir = Path(vehicles_dir)
        self._profiles: Dict[str, VehicleProfile] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        if not self.vehicles_dir.exists():
            return

        for yaml_file in sorted(self.vehicles_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and "id" in data:
                    profile = VehicleProfile(**data)
                    self._profiles[profile.id] = profile
            except Exception as e:
                # Log or handle corrupted file without breaking registry
                continue

    def get_profile(self, vehicle_id: str) -> Optional[VehicleProfile]:
        """
        Get vehicle profile by ID. Falls back to 'generic_petrol' if not found.
        """
        if vehicle_id in self._profiles:
            return self._profiles[vehicle_id]
        return self._profiles.get("generic_petrol")

    def list_profiles(self) -> List[VehicleProfile]:
        """
        List all available vehicle profiles.
        """
        return list(self._profiles.values())

    def list_vehicle_ids(self) -> List[str]:
        """
        List available vehicle IDs.
        """
        return list(self._profiles.keys())


# Global singleton instance
vehicle_registry = VehicleRegistry()

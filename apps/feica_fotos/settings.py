"""Feica Fotos settings with a one-time, read-only legacy import."""
from __future__ import annotations

import os
from PySide6.QtCore import QSettings

SETTINGS_ORGANIZATION = 'Feica Fotos'
SETTINGS_APPLICATION = 'Feica Fotos'
MIGRATION_MARKER = 'migration/local_looks_v1_complete'
# Only preferences consumed by the current App are imported. In particular,
# migration markers or unknown legacy keys never determine the new state.
MIGRATED_KEYS = ('window_geometry', 'resource_dir', 'last_dir',
                 'export_dir', 'reduce_transparency')
RESOURCE_ENV = 'FEICA_FOTOS_RESOURCE_DIR'
LEGACY_RESOURCE_ENV = 'LOCAL_LOOKS_RESOURCE_DIR'


def migrate_legacy_settings(target: QSettings, legacy: QSettings) -> bool:
    """Import absent current preferences once, without modifying legacy storage.

    Existing target values win, including empty strings and false booleans.
    A persisted completion marker prevents deliberately removed new preferences
    from being restored from legacy storage on a later launch.
    """
    if target.value(MIGRATION_MARKER, False, type=bool):
        return False
    for key in MIGRATED_KEYS:
        if not target.contains(key) and legacy.contains(key):
            target.setValue(key, legacy.value(key))
    target.sync()
    if target.status() != QSettings.Status.NoError:
        raise RuntimeError('Feica Fotos preferences could not be saved')
    target.setValue(MIGRATION_MARKER, True)
    target.sync()
    if target.status() != QSettings.Status.NoError:
        raise RuntimeError('Feica Fotos preference migration could not be saved')
    return True


def application_settings(*, factory=QSettings) -> QSettings:
    """Open the branded store; an injected factory can provide isolated stores."""
    target = factory(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
    if not target.value(MIGRATION_MARKER, False, type=bool):
        # Compatibility names identify existing user preferences, not UI branding.
        legacy = factory('LocalLooks', 'LocalLooks')
        migrate_legacy_settings(target, legacy)
    return target


def resolve_resource_dir(explicit, settings: QSettings, default, environ=None):
    """Resolve explicit path, current ENV, legacy ENV, saved path, then default."""
    env = os.environ if environ is None else environ
    return (explicit or env.get(RESOURCE_ENV) or env.get(LEGACY_RESOURCE_ENV)
            or settings.value('resource_dir') or default)

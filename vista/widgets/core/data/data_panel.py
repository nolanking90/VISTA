"""Shared base class for data manager panels."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class DataPanel(QWidget):
    """Base class for the data manager tab panels.

    Holds the viewer reference and the signals every panel shares. Panels
    emit status_message instead of reaching for the main window's status bar
    through the widget hierarchy (panels are reparented by QTabWidget.addTab,
    so self.parent() chains do not lead where they appear to).
    """

    data_changed = pyqtSignal()  # Signal when data is modified
    files_dropped = pyqtSignal(list)  # Emits list of file paths dropped onto the panel
    status_message = pyqtSignal(str, int)  # Message text + timeout ms for the status bar
    edit_mode_changed = pyqtSignal(bool)  # True while an exclusive edit mode is active

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.edit_mode_active = False

    def begin_edit_mode(self, exempt=()):
        """Lock the panel for an exclusive edit mode.

        Disables every control in the panel except the `exempt` widgets.
        Exempt widgets are force-enabled: they are the only way out of the
        mode, so they must be operable regardless of prior selection rules.
        """
        self.edit_mode_active = True
        for child in self.children():
            if isinstance(child, QWidget):
                child.setEnabled(child in exempt)
        self.edit_mode_changed.emit(True)

    def end_edit_mode(self):
        """Unlock the panel and recompute selection-dependent control states."""
        self.edit_mode_active = False
        for child in self.children():
            if isinstance(child, QWidget):
                child.setEnabled(True)
        self.refresh_control_states()
        self.edit_mode_changed.emit(False)

    def refresh_control_states(self):
        """Recompute selection-dependent enabled states after end_edit_mode.

        Default does nothing; panels with selection-dependent buttons
        override this with their selection-changed handler.
        """

    def cancel_edit_mode(self):
        """Discard the in-progress edit.

        Invoked by the main window when Escape is pressed while this panel
        has edit_mode_active set. Default does nothing; panels with edit
        modes override.
        """

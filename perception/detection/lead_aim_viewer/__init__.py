"""Lead-aim debugging tools: per-run logger + 3D/timeseries viewer.

Modules:
  recorder    - LeadAimLogger: accumulates per-frame tracker state and
                event markers, dumps to a single .npz.
  data_source - LiveSource / ReplaySource: common interface the viewer
                pulls FrameSnapshots from.
  scene_3d    - 3D subplot renderer (plate, trajectory, aim point, safety).
  scene_z     - z(t) timeseries subplot renderer.
  scene_state - text status panel renderer.
  viewer      - LeadAimViewer: figure + animation + widgets.

CLI entry: ``python -m perception.detection.lead_aim_viewer``
"""

"""Steps 4-6: the support (operation) surface.

* ``surface.plane``    (:mod:`r2s.surface.plane_fit`)    horizontal tabletop box from the collider mesh
* ``surface.trim``     (:mod:`r2s.surface.trim`)         drop floater gaussians over that surface
* ``surface.receiver`` (:mod:`r2s.surface.receiver_fit`) smooth polygon receiver from the trimmed splat

:mod:`r2s.surface.runtime` turns the descriptors into Eden entities (sim assembly).
"""

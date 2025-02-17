# Viral Room Sim

by Peter Sharpe

-----

Part of an old experiment for using potential flows as fast approximators for flowfields in indoor environments. The goal here was to do rapid real-time air quality forecasting for large rooms with mostly-quiescent air. Potential flows are mathematically-convenient here because they have no recirculation regions (e.g., closed streamlines that do not intersect a forcing singularity). This means that we can do integration of advected quantities along streamlines starting at singularities and be guaranteed to traverse the entire domain. Notably, real flows can and do have recirculation regions, so this is not strictly realistic.

## Examples

### Torus

![torus](./figures/case_1.png)

### Room

![room](./figures/case_2.png)

### Room with Multiple Fans

![room-with-fans](./figures/case_3.png)

#### Room with Open Windows

![room-with-windows](./figures/case_4.png)

import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from scipy import interpolate


class RoomSimulation:
    def __init__(self,
                 inner_radius=2,
                 outer_radius=3,
                 ):
        theta = np.linspace(0, 2 * np.pi, 300)
        unit_circle = np.stack([
            np.cos(theta),
            np.sin(theta),
        ], axis=1)

        from potentialflowvisualizer import Flowfield, Doublet, Vortex

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.inside_ring = inner_radius * unit_circle
        self.outside_ring = outer_radius * unit_circle
        self.field = Flowfield(
            objects=[
                Vortex(
                    strength=1,
                    x=0,
                    y=0,
                )
            ]
        )

    def get_velocity(self,
                     x: float | np.ndarray,
                     y: float | np.ndarray
                     ) -> tuple[float | np.ndarray, float | np.ndarray]:
        points = np.stack([
            np.reshape(x, -1),
            np.reshape(y, -1),
        ], axis=1)
        u = np.reshape(self.field.get_x_velocity_at(points), np.shape(x))
        v = np.reshape(self.field.get_y_velocity_at(points), np.shape(y))

        return u, v

    def point_is_inside_room(self,
                             x: float | np.ndarray,
                             y: float | np.ndarray
                             ) -> bool | np.ndarray:
        points = np.stack([
            np.reshape(x, -1),
            np.reshape(y, -1),
        ], axis=1)

        from matplotlib.path import Path
        is_within_inner_ring = Path(self.inside_ring).contains_points(points)
        is_within_outer_ring = Path(self.outside_ring).contains_points(points)
        is_inside_room = np.logical_and(
            is_within_outer_ring,
            np.logical_not(is_within_inner_ring)
        )

        return np.reshape(is_inside_room, np.shape(x))

    def plot_room(self, show=True, set_equal=True):
        plt.plot(*self.inside_ring.T, "-k", linewidth=3)
        plt.plot(*self.outside_ring.T, "-k", linewidth=3)
        if set_equal:
            p.equal()
        if show:
            p.show_plot(None, "$x$ [m]", "$y$ [m]")

    def plot_flowfield(self, show=True, set_equal=True):
        xlim = np.array([-1, 1]) * self.outer_radius
        ylim = np.array([-1, 1]) * self.outer_radius

        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        from matplotlib import cm

        # Extract colors from RdBu and coolwarm colormaps
        color_blue = "royalblue"  # "#cm.RdBu(float(0))
        color_gray = "darkgray"
        color_red = "crimson"  # cm.RdBu(float(1))

        # Create custom colormap using LinearSegmentedColormap.from_list
        colors = [color_blue, color_gray, color_red]
        positions = [1e-1, 1e-2, 1e-3]
        cmap = LinearSegmentedColormap.from_list('Custom_Colormap', colors, N=256)

        # Logarithmic normalization
        norm = LogNorm(vmin=1e-3, vmax=1)

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(
            np.linspace(*xlim, 300),
            np.linspace(*ylim, 300),
        )
        U, V = self.get_velocity(X, Y)
        velmag = np.sqrt(U ** 2 + V ** 2)

        inside_room = self.point_is_inside_room(X, Y)
        U = np.where(
            inside_room,
            U,
            np.nan
        )
        V = np.where(
            inside_room,
            V,
            np.nan
        )

        plt.streamplot(
            X, Y, U, V,
            color=velmag,
            linewidth=1,
            minlength=0.02,
            cmap=cmap,
            norm=norm,
            broken_streamlines=False
        )
        # plt.colorbar(label="Velocity Magnitude [m/s]")

        plt.plot(*self.inside_ring.T, "-k", linewidth=3)
        plt.plot(*self.outside_ring.T, "-k", linewidth=3)

        # ax.add_patch(
        #     plt.Circle(
        #         xy=(self.fan.x, self.fan.y),
        #         radius=0.3,
        #         color="k",
        #         zorder=4,
        #     )
        # )
        # ax.add_patch(
        #     plt.arrow(
        #         x=self.fan.x - 0.2 * np.cos(self.fan.alpha),
        #         y=self.fan.y - 0.2 * np.sin(self.fan.alpha),
        #         dx=0.2 * np.cos(self.fan.alpha),
        #         dy=0.2 * np.sin(self.fan.alpha),
        #         width=0.02,
        #         head_width=0.1,
        #         color="w",
        #         zorder=5,
        #     )
        # )

        plt.xlim(xlim + np.array([-0.1, 0.1]))
        plt.ylim(ylim + np.array([-0.1, 0.1]))

        if set_equal:
            p.equal()
        if show:
            p.show_plot("Potential Flow in a Toroidal Room", "$x$ [m]", "$y$ [m]")


if __name__ == '__main__':
    sim = RoomSimulation()

    sim.plot_flowfield(show=False)
    p.show_plot("Flow in a Toroidal Room", "$x$ [m]", "$y$ [m]",
                savefig=["figures/case_1.svg",
                         "figures/case_1.png"]
                )
